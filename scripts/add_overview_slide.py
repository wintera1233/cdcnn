#!/usr/bin/env python
"""Insert a one-page overview of the half year's side projects into 10-22.pptx.

The deck is the user's own edited copy of the merged report. One slide is
added right after the cover, on the deck's layout, summarising each project
in reports/: 07-16, 07-23, 08-05, 08-26, 09-22/10-22 and concentration.pdf.
Edits the file in place; the unedited deck is in git history.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pptx import Presentation
from pptx.util import Inches

from scripts.make_merged_deck import BODY, Deck, kicker, table, textbox  # noqa: E402

DECK = ROOT / "reports" / "10-22.pptx"
POSITION = 1   # zero-based index the new slide moves to: right after the cover

ROWS = [
    ["07-16", "特徵擷取與特徵空間分析",
     "Zenodo 長期漂移資料集：62 顆感測器、12 個月、三種 VOC",
     "Savitzky–Golay 濾波；每顆感測器取 5 種特徵；用 PCA 看濃度與日間漂移的影響",
     "建立 372 維特徵表，定出分類流程"],
    ["07-23", "特徵篩選",
     "同上",
     "移除 R0、Rmax；Pearson 相關性分群；RANSAC 找各特徵的 inlier",
     "inlier 比例最高約 0.68；固定中心的 RANSAC 描述不了隨時間漂移的特徵"],
    ["08-05", "兩階段 MDIP 特徵選擇",
     "同上；Day 1–25 訓練、Day 26–30 驗證",
     "復現 MDIP（Liu & Tang 2020），加第二階段專門分 EtOH 與 Phenylethanol",
     "14 個特徵達到 0.98，特徵數減少 95% 以上"],
    ["08-26", "通用資料擴充評估",
     "TI 漂移資料集：3 批、5 台裝置、14 顆感測器、5 種氣體",
     "三層 1D CNN 基線；高斯噪聲、通道遮蔽、視窗切片三種擴充",
     "跨批次 Macro-F1 0.74 是瓶頸；通用擴充幫助有限 → 轉向 CDCNN"],
    ["09-22 → 10-22", "CDCNN 復現（本報告）",
     "UCI 漂移資料集：16 顆感測器、36 個月、6 種氣體",
     "照 Fig. 2 重建、只用 Batch 1；三個元件逐一加入；投影延伸；文獻對照",
     "基線 0.5556；三個元件都無增益；文獻中不看 target 的方法都在 60% 以下"],
    ["另一條線", "濃度為錨的漂移校正遷移（CDC，IEEE Sensors Letters 2020）",
     "UCI 漂移資料集",
     "復現 CB 模型（自編碼器 + 多任務 MLP，同時預測標籤與濃度）與遞迴式 CDC 遷移流程，每類 5 筆 transfer sample",
     "CB 平均 91.55（論文 91.85）；CDC 流程 Batch 10 到 92.54，靜態微調只有 60.75"],
]


def main() -> int:
    prs = Presentation(DECK)
    deck = Deck(prs)
    s, _ = deck.slide("這半年的工作概覽", keep_body=False)
    kicker(s, "六條線：三條在特徵工程，三條在漂移補償；資料集從 Zenodo、TI 走到 UCI。")
    X, _, W, _ = BODY
    table(s, ["時間", "主題", "資料集", "做了什麼", "結果"], ROWS,
          X, Inches(1.62), W, col_fracs=(0.1, 0.17, 0.2, 0.3, 0.23), size=9,
          row_h=Inches(0.72), bold_rows=(5,), aligns=["left"] * 5)
    # Move the new slide to its place after the cover.
    id_list = prs.slides._sldIdLst
    new = list(id_list)[-1]
    id_list.remove(new)
    id_list.insert(POSITION, new)
    prs.save(DECK)
    print(DECK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
