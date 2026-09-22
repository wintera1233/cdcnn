# CDCNN Baseline 最終設計

重建論文（*Sensors and Actuators: A. Physical* 372 (2024) 115314）Table 3 的
ResNet baseline。只有 `L_ce`，沒有資料擴充、特徵生成、對比損失。

分支 `exp/v7-redesign`，2026-09-23 定案。

## 架構

輸入是一筆量測的 128 維向量，以 `[N, 1, 128]` 進入（1 個 channel，長度 128）。
128 = 16 個 sensor × 每個 8 個統計量；**不得** reshape 成 16×8 影像。

```
Normal（per-sample 正規化：每筆樣本用自己 128 個值的均值與標準差標準化）
   │
Resnet1   1 → 32     3 Conv 32  → 3 Conv 32     shortcut 1 Conv 32
Resnet2  32 → 64     3 Conv 64  → 3 Conv 64     shortcut 1 Conv 64
Resnet3  64 → 128    3 Conv 128 → 3 Conv 128    shortcut 1 Conv 128
Resnet4 128 → 256    3 Conv 256 → 3 Conv 256    shortcut 1 Conv 256
Resnet5 256 → 128    3 Conv 512 → 3 Conv 128    shortcut 1 Conv 128
   │  （block 內部無任何正規化層；全程無 pooling，長度維持 128）
Flatten (128×128 = 16384)
Linear(16384 → 128) → BatchNorm1d(128) → Linear(128 → 6)
```

參數 **3,156,390**（backbone 1,058,080 + head 2,098,310）。

每個 block 是 `Conv1d(k=3) → ReLU → Conv1d(k=3)` 與 `Conv1d(k=1)` shortcut 相加後
再 ReLU，`padding=1` 維持長度。

## 訓練

| 項目 | 值 |
|---|---|
| 損失 | `L_ce`，照 Eq. (S2)：批內平均後跨批平均 |
| 最佳化 | SGD，lr **0.0003**，momentum 0.9，weight decay 1e-4 |
| 排程 | `StepLR(step_size=25, gamma=0.5)` |
| Epochs | 100（固定，不可選——見下） |
| Batch size | 64（445 筆 = 6×64 + 1×61） |
| Seeds | 1042, 2024, 3407 |

## 協定

Batch 1（445 筆）是唯一的來源。所有訓練、正規化擬合、調參、交叉驗證只用它。
Batches 2–10 在全部 checkpoint 寫入並雜湊之前不得開啟；`load_target()` 沒有
`TargetAccessLog` 就拋錯，每次 run 都寫 leakage audit。

指標是 **target mean**：九個 batch 各自準確率的**未加權平均**，與論文 Table 3 同一個統計量。

## 結果

| | target mean |
|---|---:|
| **本設計** | **0.5222** ± 0.0277 |
| 論文 ResNet | 0.6344 |
| 論文 CDWC | 0.6705 |
| 論文 CDCNN | 0.7230 |

Batch 1 準確率 0.9633。逐 batch：

| B2 | B3 | B4 | B5 | B6 | B7 | B8 | B9 | B10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.827 | 0.729 | 0.673 | 0.501 | 0.688 | 0.466 | 0.171 | 0.255 | 0.390 |

**九個 batch 有六個追平或超過論文的 ResNet**（B2 +0.058、B3 +0.067、B4 +0.030，
B6/B7/B10 差距在 0.04 以內）。整個 0.112 的缺口集中在 B5、B8、B9。

## 為什麼是這些選擇

14 個 cell 的量測，粗體為統計上可分辨（2×SE）：

| 選擇 | 對 target mean 的效應 |
|---|---:|
| per-sample 輸入取代 Batch-1 擬合的 `StandardScaler` | **+0.112** |
| flatten head 取代 global average pooling | **+0.078** |
| Fig. 2 的通道寬度取代內文的「封頂 128」 | +0.029 |
| lr 0.001 → 0.0003 | +0.045 |
| head 用 BatchNorm 而非 LayerNorm（在 per-sample 之下） | **+0.075** |

- **per-sample 與 LayerNorm 是替代品**，交互作用 −0.141。兩者都是「用樣本自己的
  統計量取代存下來的 Batch 1 統計量」，做兩次等於把同樣的資訊除掉兩次。單獨用
  per-sample（輸入端）最好。
- **GAP 有害**。論文「this network doesn't apply the pooling layer」是有分量的：
  每組 sensor 的第 1 個是量級 10⁴ 的穩態電阻、其餘 6 個是量級 10⁰ 的瞬態特徵，
  平均會抹掉氣體辨識所依賴的跨 sensor 比例。
- **記憶化不是病因**。flatten head 把 Batch 1 背到 1.0000，GAP 只到 0.955，
  但前者 target 反而高。

## 兩個已知的結構性限制

**1. Epoch 數無法選擇。** 兩次 5-fold 交叉驗證（共 120 folds）證明：held-out
準確率與 held-out 損失都**無法**定位 target 峰值。最清楚的一格 target 在 epoch 7
達峰、到 epoch 100 掉 0.164，而同一段窗口內 held-out 準確率上升 0.050、held-out
損失下降 1.298 且從未回升。兩個 source-only 訊號與 target **反向相關**。

原因是結構性的：交叉驗證量的是「對同一個 domain 的泛化」，我們要偵測的是「對漂移後
domain 的泛化」，把 Batch 1 擬合得更好會改善前者並摧毀後者。因此 epoch 數只能事先
以慣例訂死（論文訂 100，本設計沿用）。代價可量化：若能停在各自峰值，最佳格值 +0.057。
細節見 `docs/early-stopping.md`。

**2. Ethylene 完全無法遷移。** 在 Batch 1 上以 `StandardScaler` 可學到 recall
1.000，在 Batches 2–10 上是 0.001，本專案訓練過的所有設定都不超過 0.027。
**類別加權無效**——它在 source 已經滿分。Ethylene 在原始 PCA 空間中是一條外伸的臂，
辨識靠的是整體強度，而那正是感測器三年老化所摧毀的量。

逐類別 recall（本設計 vs 論文 CDCNN）：

| | Acetone | Acetaldehyde | Ethanol | Ethylene | Ammonia | Toluene |
|---|---:|---:|---:|---:|---:|---:|
| 本設計 | 0.668 | **0.858** | 0.616 | **0.000** | 0.514 | 0.388 |
| 論文 | 0.94 | **0.00** | 0.97 | **0.95** | 0.91 | 0.92 |

兩邊死在**相反的類別**上：論文把 100% 的 Acetaldehyde 判成 Ethanol（兩者皆為 C2
含氧化合物），卻能救回 Ethylene。論文在六個類別中有五個領先，所以剩餘差距不是單一
類別的問題。詳見 `docs/dead-class.md`。

## 重現

```bash
docker build -t cdcnn:cu121 -f docker/Dockerfile .
DOCKER="docker run --rm --gpus all --user $(id -u):$(id -g) -e HOME=/tmp \
  -v $PWD:/workspace -w /workspace cdcnn:cu121"

$DOCKER python -m unittest discover -s tests
$DOCKER python scripts/run_baseline.py gpu-smoke --config configs/head_normalisation.json
$DOCKER python scripts/run_baseline.py launch --config configs/head_normalisation.json \
  --max-workers 1 --gpu-smoke-run runs/<passing_gpu_smoke>
```

環境必須是 torch 2.5.1+cu121。torch 2.8.0+cu126 對這台機器的 CUDA 12.2 驅動
（535.309.01）會觸發 Xid 31 MMU fault 並汙染整台機器的 CUDA 狀態。

`R-fig-ps@lr0.0003` 在三次獨立的 run 中都回傳 0.5222，小數點後四位相同。

## 尚未解決

1. **Ammonia 在 B6 之後崩進 Toluene**（B8 的 429 筆中有 417 筆），單獨值 0.117。
2. **論文為何在五個類別上領先**——不是調參能補的差距。
3. **以人工漂移擾動的 Batch 1 樣本做驗證**，是唯一未試過的合法早停途徑，
   同時也會是論文資料擴充模組第一個獨立存在的理由。
