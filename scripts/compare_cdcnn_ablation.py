#!/usr/bin/env python3
"""Build an auditable comparison from validated B0 and completed A1-A3 runs."""
from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pca_analysis import save_json, sha256


def verify_ablation(path: Path):
    summary=json.loads((path/"summary.json").read_text()); metrics=pd.read_csv(path/"batch_metrics.csv")
    pred=pd.read_csv(path/"predictions.csv.gz")
    if len(pred)!=13465 or sorted(pred.batch.unique())!=list(range(2,11)):
        raise ValueError(f"{path.name}: target prediction coverage failure")
    rebuilt=pred.groupby("batch").correct.mean().reindex(range(2,11)).to_numpy()
    if not np.allclose(rebuilt,metrics.set_index("batch").reindex(range(2,11)).accuracy):
        raise ValueError(f"{path.name}: batch metrics disagree with predictions")
    if not np.isclose(rebuilt.mean(),summary["target_unweighted_mean_accuracy"]):
        raise ValueError(f"{path.name}: target mean disagrees with predictions")
    if not np.isclose(pred.correct.mean(),summary["target_pooled_accuracy"]):
        raise ValueError(f"{path.name}: pooled accuracy disagrees with predictions")
    cv=pd.read_csv(path/"cv_predictions.csv.gz")
    cv_unweighted=cv.groupby("cv_fold").correct.mean().mean()
    if len(cv)!=445 or cv.sample_id.nunique()!=445 or not np.isclose(cv_unweighted,summary["mean_source_cv_accuracy"]):
        raise ValueError(f"{path.name}: source CV predictions/summary disagree")
    return summary,metrics


def main():
    root=Path.cwd().resolve()
    paths={"B0":root/"runs/20260906T091956Z_resnet", "A1":root/"runs/20260907T191438Z_a1",
           "A2":root/"runs/20260907T200808Z_a2", "A3":root/"runs/20260907T214925Z_a3"}
    validation=root/"runs/20260907T184942Z_resnet_validation"
    b0v=json.loads((validation/"summary.json").read_text()); b0m=pd.read_csv(paths["B0"]/"batch_metrics.csv")
    b0p=pd.read_csv(paths["B0"]/"predictions.csv.gz")
    if len(b0p)!=13465 or not np.isclose(b0p.correct.mean(),b0v["target_sample_weighted_accuracy"]):
        raise ValueError("Validated B0 prediction evidence failed comparison verification")
    summaries={"B0":{"mean_source_cv_accuracy":b0v["mean_source_cv_accuracy"],
        "target_unweighted_mean_accuracy":b0v["target_unweighted_mean_accuracy"],
        "target_pooled_accuracy":b0v["target_sample_weighted_accuracy"]}}
    metrics={"B0":b0m}
    for exp in ("A1","A2","A3"): summaries[exp],metrics[exp]=verify_ablation(paths[exp])
    rows=[]
    for exp in ("B0","A1","A2","A3"):
        row={"experiment":exp,"batch1_cv_accuracy":summaries[exp]["mean_source_cv_accuracy"]}
        by=metrics[exp].set_index("batch").accuracy
        row.update({f"batch{b}_accuracy":float(by[b]) for b in range(2,11)})
        row["target_unweighted_mean_accuracy"]=summaries[exp]["target_unweighted_mean_accuracy"]
        row["target_pooled_accuracy"]=summaries[exp]["target_pooled_accuracy"]; rows.append(row)
    comparison=pd.DataFrame(rows); deltas=[]
    for component,left,right in (("input augmentation","B0","A1"),("feature generation","A1","A2"),("contrastive objective","A2","A3")):
        l=comparison.set_index("experiment").loc[left]; r=comparison.set_index("experiment").loc[right]
        deltas.append({"ablation":f"{right} - {left}","component":component,
            "delta_batch1_cv_accuracy":r.batch1_cv_accuracy-l.batch1_cv_accuracy,
            "delta_target_unweighted_mean_accuracy":r.target_unweighted_mean_accuracy-l.target_unweighted_mean_accuracy,
            "delta_target_pooled_accuracy":r.target_pooled_accuracy-l.target_pooled_accuracy})
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_cdcnn_comparison"; out=root/"runs"/run_id; out.mkdir(parents=True,exist_ok=False)
    comparison.to_csv(out/"comparison.csv",index=False); pd.DataFrame(deltas).to_csv(out/"ablation_deltas.csv",index=False)
    config={"reference":"validated B0; not retrained","experiment_order":["A1","A2","A3"],"runs":{k:str(v.relative_to(root)) for k,v in paths.items()},"b0_validation_run":str(validation.relative_to(root)),"selection_scope":"Batch 1 only; target results were not used for tuning","primary_metric":"unweighted mean of Batch 2-10 accuracies"}
    save_json(out/"configuration.json",config)
    inputs=[]
    for exp,path in paths.items():
        for filename in ("summary.json","batch_metrics.csv","predictions.csv.gz"):
            p=path/filename; inputs.append({"experiment":exp,"path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":sha256(p)})
    for filename in ("summary.json","report.md"):
        p=validation/filename; inputs.append({"experiment":"B0_validation","path":str(p.relative_to(root)),"bytes":p.stat().st_size,"sha256":sha256(p)})
    save_json(out/"input_manifest.json",inputs); save_json(out/"software_versions.json",{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"platform":platform.platform()})
    report(comparison,pd.DataFrame(deltas),out,paths)
    save_json(out/"run_manifest.json",{"run_id":run_id,"created_utc":datetime.now(timezone.utc).isoformat(),"status":"completed","preprocessing_fit_scope":"A1-A3: fold-local Batch 1 scalers for CV; all-Batch-1 scaler/model for final fit; Batches 2-10 evaluation only.","selection_scope":"All assumptions fixed before target evaluation; no target-guided tuning.","verification":"All A1-A3 target and CV accuracies recomputed from sample predictions; B0 linked to its separate validation run.","exclusions":[]})
    save_json(out/"output_inventory.json",sorted(p.name for p in out.iterdir() if p.name!="output_inventory.json")); print(out)


def report(comp,deltas,out,paths):
    headers=["Model","Batch 1 CV"]+[f"B{i}" for i in range(2,11)]+["Target mean","Pooled"]
    lines=["| "+" | ".join(headers)+" |","|"+"---|"*len(headers)]
    for r in comp.itertuples(index=False):
        vals=[r.experiment,f"{r.batch1_cv_accuracy:.6f}"]+[f"{getattr(r,f'batch{i}_accuracy'):.6f}" for i in range(2,11)]+[f"{r.target_unweighted_mean_accuracy:.6f}",f"{r.target_pooled_accuracy:.6f}"]
        lines.append("| "+" | ".join(vals)+" |")
    dlines=["| Ablation | Component | Δ CV | Δ target mean | Δ pooled |","|---|---|---:|---:|---:|"]
    for r in deltas.itertuples(index=False): dlines.append(f"| {r.ablation} | {r.component} | {r.delta_batch1_cv_accuracy:+.6f} | {r.delta_target_unweighted_mean_accuracy:+.6f} | {r.delta_target_pooled_accuracy:+.6f} |")
    text=f"""# B0–A3 CDCNN ablation comparison\n\n## Outcome\n\nAll A1–A3 experiments completed successfully in the required sequence. B0 was not retrained; values come from its validated prediction-level evidence. The best primary metric among these fixed runs is B0 at **{comp.target_unweighted_mean_accuracy.max():.6f}**. Among A1–A3, A2 has the highest primary metric at **{comp.set_index('experiment').loc['A2','target_unweighted_mean_accuracy']:.6f}**; all three remain below B0.\n\n## Configuration summary\n\nAll models use the 5,116,518-parameter five-block ResNet, fold-local Batch 1 scaling for five-fold CV, and a final scaler/model fit on all 445 Batch 1 rows. Adam, lr 0.001, weight decay 1e-4, batch 64, and 100 epochs were fixed. A1 adds one Gaussian-noise copy per training row (standardized-space σ=0.05). A2 adds one same-class Beta(2,2) interpolation per augmented row after ResNet3. A3 adds probability MSE (weight 0.1) and paired cosine-distance contrastive loss (weight 0.1) to explicit CE. These underdetermined paper details were declared before execution and were not tuned from targets.\n\n## Comparison\n\n{chr(10).join(lines)}\n\nThe target mean is the unweighted mean of the nine batch accuracies; pooled is sample-weighted across 13,465 target rows.\n\n## Component contributions\n\n{chr(10).join(dlines)}\n\nInput augmentation reduced both target summaries versus B0 under this fixed Gaussian-noise assumption. Feature generation recovered some performance. The contrastive objective increased pooled accuracy and Batch 1 CV over A2, but reduced the primary unweighted target-batch mean because its gains were concentrated in larger batches (especially Batch 10). Because the assumed augmentation/generation/loss definitions were not uniquely specified by the available figure/specification, these results characterize this documented implementation, not every possible CDCNN realization.\n\n## Integrity and interpretation\n\nAll 13,910 source files' rows were validated before each experiment; no samples or batches were excluded, deduplicated, clipped, or imputed. Each A1–A3 metric was recomputed from 445 source-CV and 13,465 target sample predictions. Batches 2–10 were only transformed by the all-Batch-1 scaler and evaluated after training. The cross-batch changes are measured distribution-generalization results and do not uniquely establish sensor drift as their cause.\n\n## Run locations\n\n- B0: `{paths['B0'].relative_to(Path.cwd())}` (validated by `runs/20260907T184942Z_resnet_validation`)\n- A1: `{paths['A1'].relative_to(Path.cwd())}`\n- A2: `{paths['A2'].relative_to(Path.cwd())}`\n- A3: `{paths['A3'].relative_to(Path.cwd())}`\n"""
    (out/"report.md").write_text(text,encoding="utf-8")


if __name__=="__main__": main()
