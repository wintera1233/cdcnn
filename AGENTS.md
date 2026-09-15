# Project Instructions for AI Agents

**1. General Principles & Dataset Rules**
- Dataset: UCI Gas Sensor Array Drift Dataset (13,910 measurements, Batches 1–10, 6 gas classes, 128 extracted features)[cite: 11].
- Data Immutability: Treat raw data as strictly immutable[cite: 11]. Never modify, rename, move, delete, or clip raw data[cite: 11]. Do not introduce automatic outlier removal or value imputation[cite: 11].
- Metadata Exclusion: Do not assume concentration labels, timestamps, or device IDs exist[cite: 11]. Keep metadata out of model feature inputs[cite: 11].

**2. CDCNN v6 Protocol & Source-Only Rules**
- Target Data Isolation: All model training, scaler fitting (`StandardScaler`), tuning, and CV must strictly use Batch 1[cite: 12]. Batches 2–10 must remain inaccessible until final model checkpoints are written and frozen[cite: 12].
- Unified Hyperparameters: All stages (B0, A1, A2-semantic, A3) must share identical protocol[cite: 12]:
  - Optimizer: SGD (lr=0.001, momentum=0.9, weight_decay=1e-4)[cite: 12].
  - Scheduler: `StepLR(step_size=25, gamma=0.5)`[cite: 12].
  - Epochs & Batch Size: Exactly 100 epochs (Early Stopping disabled), Batch Size = 64[cite: 12].
  - Seeds: Fixed 5 seeds (`1042`, `2024`, `3407`, `42`, `123`)[cite: 12].
- Model Implementations:
  - `A1`: Variance mixing with standard deviation API mapping (`scale = np.sqrt(v_mix)`)[cite: 12].
  - `A2-semantic`: MaxPool/Upsample is $z_{\text{low}}$ (low-frequency style); residual is $z_{\text{high}}$ (high-frequency detail)[cite: 12]. $\sigma'$ must use positive Log-Normal sampling, and $L_{\text{MSE}}$ acts on Softmax probabilities[cite: 12].

**3. Execution Safeguards & Execution Commands**
- Single-Worker CUDA: For GPU runs, CUDA launchers must enforce `--max-workers 1` to prevent VRAM collisions[cite: 10].
- GPU Pre-flight Gate: The launcher must reject full runs unless a passing GPU smoke artifact (`gpu-smoke`) exists[cite: 10].
- Execution Command: Launch full ablation runs strictly via `scripts/run_cdcnn_v6_full.py`[cite: 10].

**4. Run Directory Hygiene & Traceability**
- Immutable Output Artifacts: Every run must write to a unique `runs/<timestamp>_<name>/` directory[cite: 10, 11]. Never overwrite existing run outputs[cite: 10, 11].
- Quarantine Protocol: Never delete obsolete/failed run directories directly[cite: 9, 10]. Move them to `.trash/run-cleanup-<date>/` and record the rationale in `docs/run-cleanup-<date>.md`[cite: 9, 10].

**5. Legacy 2D Framework Ban**
- Never restore, add, import, or invoke a Conv2d gas-classification model or a 128-feature-to-16x8 reshape.
- `src/resnet_baseline.py`, `scripts/run_resnet.py`, and `configs/resnet.json` are disabled tombstones, not training interfaces.
- The canonical architecture is Conv1d over `[N, 1, 128]`; full training is permitted only through `scripts/run_cdcnn_v6_full.py` with `configs/cdcnn_v6.json`.
