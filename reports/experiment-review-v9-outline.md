# 照論文實作 CDCNN 時發現的問題：簡報大綱

> 論文本身的介紹由講者另備，本大綱只涵蓋「照著論文做之後撞到什麼」。
> 每頁標明要放的圖：**論文的**在 `docs/paper/`，**我們的**在 `reports/figures/`。
> 論文的 PCA 總圖截在 `docs/paper/fig4a.png`（圖說印的是 Fig. 3，簡報請用論文實際編號），
> Fig. S1 截在 `docs/paper/figs1.png`。
> 數字全部出自 `runs/` 下通過 leakage audit 的執行（2026-09-22 至 09-23），
> 依據文件：`baseline.md`、`docs/v8-augmentation.md`、`docs/v9-feature-generation.md`、
> `docs/drift-geometry.md`、`docs/label-mapping.md`、`docs/early-stopping.md`。

---

## 第 1 頁：我們重現了什麼、沒重現什麼

| 論文的東西 | 重現狀態 | 我們的數字 | 論文 |
|---|---|---:|---:|
| 純 ResNet 基線（Fig. 2） | 架構照圖重建，source-only 訓練 | 0.5556 | 0.6346 |
| 資料擴充（Eq. 7，等向噪聲） | 照做 | −0.0216 | 含在 +0.036 內 |
| 特徵生成（Eqs. 8–16）+ `L_MSE`（S4） | 照做，Fig. 5 可畫出同樣的圖 | +0.0004 | 含在 +0.036 內 |
| 對比損失 `L_con`（S5） | 尚未 | — | +0.053 |
| Acetaldehyde 完全誤判（Fig. S3） | **重現了**，和論文一樣 0.00 | 0.000 | 0.00 |

- 協定：只用 Batch 1（445 筆）訓練與選擇，Batch 2–10 在所有 checkpoint 凍結後才開，每次執行有稽核
- 五個 seed、附 SD、可分辨門檻 0.009–0.022
- 本簡報接下來每頁一個實作時發現的問題

---

## 第 2 頁：問題一 —— 論文文字與圖互相矛盾，資料支持圖

**圖：`docs/paper/fig2_cdcnn.png`（論文 Fig. 2）**

| 論文說的 | 文字版 | 圖版 | 實測 |
|---|---|---|---:|
| Channel 數 | §5.2「1 到 128 逐步增加」→ 32/64/128/128/128 | Fig. 2 印 32/64/128/256/128，內層 512 | 圖版高 +0.03 到 +0.05 |
| Pooling | 「this network doesn't apply the pooling layer」 | — | 加 GAP 掉 **−0.078**，這句是關鍵句 |
| Eq. (S2) | 對 minibatch 取未加權平均 | — | 最後一批 61 筆，權重不同，實作照 S2 並記錄差異 |
| Eq. (S4) | 只有 1/B、1/n，沒有 1/C | — | `F.mse_loss` 預設會除以 6，等於 λ/6 |
| Eqs. (10)–(11) 的統計量維度 | 說結果在 R^{B×CH}，那會讓重上色變恆等 | — | 三種讀法（逐純量／channel／位置）都做了，都約零 |
| Eq. (14) | 高斯可抽到負的尺度 | — | 照論文不截斷，逐位置負值比例 0.000 |

- 結論：能照做的都照做了；每個歧義都記錄了兩種讀法的差，最大的一個是 pooling

---

## 第 3 頁：問題二 —— 資料集的標籤對不上論文的 Table 2

**圖：我們的 `batch1_pca.png`（只標整數標籤）對照論文 `docs/paper/fig4a.png` 的 batch1 面板**

- 資料集文件說 `1=Ethanol, 2=Ethylene, 3=Ammonia, 4=Acetaldehyde, 5=Acetone, 6=Toluene`
- 論文 Table 2 的 Batch 1 各氣體筆數（83/30/70/98/90/74）照欄位順序讀，要求標籤 1 有 83 筆；實際是 90
- 用 Batch 1 的 PCA 形狀對論文 Fig. 4(a)，判定標籤 2 與 3 互換；Table 2 六欄只有一欄對得上
- 這不影響任何準確率（模型只看整數），但影響所有「某氣體怎樣」的敘述，前一分支的對應表是錯的

---

## 第 4 頁：問題三 —— Fig. S1 的曲線畫的是什麼？

**圖：我們的 `figS1_overlay.png`；論文 `docs/paper/figs1.png`（a 是 accuracy，b 是 loss）**

- 論文 Fig. S1 的 ResNet／CDWC／CDCNN 曲線在 0.58／0.65／0.70 進入平台
- 我們的 Batch 1 訓練準確率 10 個 epoch 內到 0.95 以上，最後 1.000；target mean 落在 0.35–0.55
- 兩邊都對不上：若 S1 是訓練曲線，0.58 太低；若是 target 曲線，它暗示訓練時一直在看 target
- S1(b) 的 loss 也值得一提：CDCNN 的 loss 停在 1.5，ResNet 與 CDWC 到 0.5。CDCNN 多了 `L_MSE` 與 `L_con` 兩項，總 loss 高是合理的，但 1.0 的差距遠大於我們量到的 `L_MSE`（最終佔 loss 3%）
- 延伸問題：**source-only 下沒有停止規則**。5-fold CV 120 個 fold，held-out accuracy 與 held-out loss 的峰值都落在 epoch 60–100，而 target 的峰值在 7／40／70，反相關。epoch 只能固定為 100

---

## 第 5 頁：問題四 —— 論文自己的 CDCNN 也死在 Acetaldehyde

**圖：論文 `docs/paper/figS3_confusion.png` 對照我們的 `baseline-confusion_best.png`**

| 類別 | 論文 CDCNN | 論文 CDWC | 我們的基線 |
|---|---:|---:|---:|
| Ethanol | 0.97 | 0.88 | 0.67 |
| Ethylene | 0.95 | 0.87 | 0.62 |
| Ammonia | 0.91 | 0.87 | 0.86 |
| **Acetaldehyde** | **0.00** | **0.00** | **0.00** |
| Acetone | 0.94 | 0.85 | 0.51 |
| Toluene | 0.92 | 0.27 | 0.39 |

- 三個模型都把 Acetaldehyde 送進 Ethanol（論文 100%、92%；我們 38%）
- 原因是漂移，不是化學相似：它在 Batch 1 上 recall 1.000，target 上每個 batch 的質心都落到 Batch 1 的 Ethanol 位置
- Acetaldehyde 佔 target 14%，論文的 pooled 準確率因此只有約 0.80；target mean 0.7230 是在它為零的情況下達成的
- **所以我們與論文的差距全在其他五類**，尤其 Toluene（0.92 對 0.39）與 Acetone（0.94 對 0.51）

---

## 第 6 頁：問題五 —— 論文的資料擴充是負的，給了方向才是正的，但方向來自 target

**圖：我們的 `augmentation_pca.png`、`drift_dimension.png`**

| 變體 | 擴充 | target mean | vs 基準 | 可分辨 |
|---|---|---:|---:|---|
| `R-aug-t2` | 沿漂移方向，T=2，無噪聲 | 0.5770 | **+0.0214** | 是 |
| `R-fig-logps` | 無 | 0.5556 | — | — |
| `R-aug-paper` | 論文 Eq. (7) 等向噪聲 | 0.5340 | **−0.0216** | 否 |

- 論文寫的等向噪聲讓結果變差；同樣的位移給了方向就是本專案唯一可分辨的增益
- 但方向是看過 target 才選的，T 與去噪聲也是 → target-informed，只能算上界
- 增益全是 Acetone +0.255 換 Ethylene −0.272；死類別仍是 0.003，因為它與 Ethanol 幾乎一起漂（cos 0.761），共用方向平移解不開
- 漂移本身：感測器現象、約四維、PC1 佔 70.6%；Batch 1 自己的兩個採集區塊偏移能估到方向（cos 0.850），這是 source-only 的替代方案，尚未拿來訓練

---

## 第 7 頁：問題六（重點）—— 論文 Fig. 5 的畫法在 source-only 下違規

**圖：論文 `docs/paper/fig5-feature-generation.png` 對照我們的 `feature_generation.png`**

- Fig. 5 的 caption：「The outcome of the feature generation block **in one of the test domains (batch2)**」。藍點是 batch 2 的原始特徵，橘點是把區塊套在 batch 2 上得到的人工特徵
- 我們照同樣的畫法畫，形狀對得上：橘色雲比藍色雲寬、往外延伸
- **兩個問題：**
  1. **這張圖用了 target 資料。** 區塊套在 batch 2 上才有橘點；若每個 batch 各自這樣畫，就是每個 batch 都獨立進了模型。訓練時區塊只看得到 Batch 1，推論時區塊根本不作用（Fig. 2），所以這張圖畫的不是訓練中發生的事
  2. **這是循環論證。** 從 batch 2 生成的特徵當然靠近 batch 2，不管區塊做什麼；它不能證明區塊「涵蓋到未見過的域」
- 正確的問法：區塊只作用在 Batch 1，生成的雲有沒有到後面的 batch？→ 下一頁

---

## 第 8 頁：問題七（重點）—— 從 Batch 1 生成的特徵沒有覆蓋任何後續 batch

**圖：我們的 `generation_drift_aligned_2d.png`（主圖）、`generation_drift_projection.png`（一維數字）**

- 畫法：每個（氣體, batch）取 X 軸 = 真實漂移方向 u（Batch 1 質心 → target 質心），Y 軸 = 扣掉 u 之後殘差的第一主成分。漂移軸是建構出來的，雲沒有沿它走就藏不起來
- 三片雲：灰 = Batch 1 原始、橘 = 從 Batch 1 生成、藍 = target 原始
- **51 格裡橘色幾乎完全疊在灰色上，藍色停在 target 質心那條線，中間一段空隙**

| 一維量（沿 u，source 質心 0、target 質心 1） | 中位 | 範圍 |
|---|---:|---:|
| 橘藍直方圖重疊 | **0.00** | 0.00–0.42 |
| 橘雲寬度 ÷ 灰雲寬度 | **1.01** | 0.80–1.16 |
| 區塊沿 u 的配對位移（漂移距離為單位） | **0.03** | 0.007–0.166 |

- 重疊 ≥ 0.10 的六格全是 target 自己散回 source 附近（B2 的 Ethanol、Ammonia、Acetone 等），不是橘雲走過去
- 換另一個 checkpoint（`R-aug-t2`）重跑，每格差在小數第三位
- 之前試過的三種量法（2D PCA 凸包、全維半徑、質心距離）互相矛盾，各有盲點；這個一維＋對齊二維才是誠實的量法

---

## 第 9 頁：為什麼會這樣 —— Eq. (14) 沒有正負號，沿軸幅度太小

- 區塊的擾動確實偏好漂移的子空間：|cos| 是隨機虛無值的 3.1–4.6 倍，7.6% 能量落在 51 維漂移張成空間（虛無 0.3%）
- 但 Eq. (14) 的高斯是**對稱**的：順著漂移和逆著漂移機率相同，逐類有號分量平均 −7.5%
- 全空間模長 0.95 個半徑看起來「只差 2 倍」，投影到任一條真實漂移軸只剩 1–3%，差 30 倍
- 網路確實學到了區塊要求的不變性（`L_MSE` 降 2–5 倍、λ 拉到 1.0 也不動），但真實漂移有 94% 原封不動穿過 backbone
- 準確率：四格五 seed，**+0.0004**，效應下限的十分之一
- 與問題五是同一件事的兩面：無方向的位移是負的或零，方向就是全部效應，而 Eq. (14) 把它丟掉了

---

## 第 9b 頁：問題八 —— 把正負號補上也沒用（v10.0）

**圖：無；表格為主。數字出自 `runs/20260929T031927475103Z_baseline_ladder_full`。**

- 方向用 Batch 1 自己兩個採集區塊的偏移，在 block 3 的 style 空間估，每個 epoch 重估，不碰 target
- `R-gen-sign`：只修正負號，幅度照論文；`R-gen-shift`：論文噪聲之外再加 2 個區塊偏移的定向位移

| 變體 | target mean | SD | vs `R-gen` | 可分辨 |
|---|---:|---:|---:|---|
| `R-gen-shift` | 0.5796 | 0.0077 | +0.0023 | 否 |
| `R-gen` | 0.5773 | 0.0069 | — | — |
| `R-gen-sign` | 0.5767 | 0.0115 | −0.0006 | 否 |

- **方向是對的、幅度也對**：事後在 style 空間量，方向與真實漂移 cos +0.593（虛無值 0.088），六類全為正；位移是真實漂移的 0.73 到 1.41 倍
- 每個 batch、每個類別的差都在 0.012 內；同樣的定向位移在輸入端（v8.1）能搬 Acetone、Ethylene 各 0.25 以上
- 事前否證條件觸發。**特徵生成區塊作為機制關掉了，與 Eq. (14) 有沒有正負號無關**：它能碰到的子空間不是決策邊界會回應的那一個

---

## 第 9c 頁：最後一個元件 `L_con`（Eq. S5）—— 已實作，決定不跑

- S5 是標準的 Supervised Contrastive Loss，作用在 `z_f ∪ z̄_f`，生成特徵沿用原標籤，`f` 投影到單位球；τ 與 λ_con 論文都沒給
- 實作在 `src/loss.py`，四格 config（`R-con`、`R-con-shift`、`R-con-t5` 對 `R-gen`）與事前預測都寫好了，單元測試與單一 minibatch 前向都通過
- **為什麼不跑：** 它是唯一要求不變性的損失，但它要求的是「`z_f` 對 `z̄_f` 不變」。v9、v10 已經量過 `z̄_f`：從 Batch 1 生成的雲沒有離開 Batch 1（沿漂移軸重疊 0.00、寬度比 1.01），連方向正確、幅度對的 `z̄_f` 都只值 +0.0023。對一個沒離開 source 的擾動做不變性，不會變成對真實漂移的不變性
- 前一分支在另一個 backbone 上量了五次：−0.005 到 +0.003
- 附帶一個尺度問題：`L_con` 起始 ≈ log(127) ≈ 4.8，是 `L_ce` 的兩倍多，乘 0.5 後主導前期梯度；與論文 Fig. S1(b) CDCNN 的 loss 停在 1.5 一致
- 結論：論文三個元件中，兩個量到零、第三個的前提已被量掉。復現到此為止；config 留著，隨時可以 12 分鐘跑完

---

## 第 10 頁：總結與下一步

**重現了的：** 架構、Eq. (7) 擴充、Eqs. (8)–(16) 生成、Fig. 5 的畫法、Acetaldehyde 的死亡。
**重現不了的：** 論文的準確率增益 —— 兩個元件在 source-only 下各約零。

| 元件 | 本分支（五 seed） | 前一分支（另一 backbone） | 論文宣稱 |
|---|---:|---:|---:|
| 擴充（論文等向） | −0.0216 | −0.013 到 −0.009 | ResNet→CDWC +0.036 |
| 特徵生成 + `L_MSE`（論文原版） | +0.0004 | +0.001 到 +0.014 | （含在上列） |
| 特徵生成，Eq. (14) 加正負號與幅度（v10.0） | +0.0023 | — | — |
| 對比損失 `L_con` | 已實作，未跑 | −0.005 到 +0.003 | CDWC→CDCNN +0.053 |

- v10.0 已做：正負號與幅度都補上仍是 +0.0023，特徵生成關掉
- `L_con` 已實作未跑：前提（`z̄_f` 要離開 Batch 1）已被 v9、v10 量掉；config 與預測留檔，12 分鐘可補
- 復現結論：論文三個元件在 source-only 協定下沒有一個能重現增益；與論文的差距在 Acetaldehyde 之外的五類，來源不在這三個元件
- 任何後續工作都分「五類」與「六類」報告，因為死類別是所有共用方向方法的天花板

---

## 附錄：圖檔清單

| 頁 | 論文的圖 | 我們的圖 |
|---|---|---|
| 2 | `docs/paper/fig2_cdcnn.png` | — |
| 3 | `docs/paper/fig4a.png`（batch1 面板） | `batch1_pca.png`、`batch1_pca_zoom23.png` |
| 4 | `docs/paper/figs1.png` | `figS1_overlay.png` |
| 5 | `docs/paper/figS3_confusion.png` | `baseline-confusion_best.png` |
| 6 | — | `augmentation_pca.png`、`drift_dimension.png` |
| 7 | `docs/paper/fig5-feature-generation.png` | `feature_generation.png` |
| 8 | — | `generation_drift_aligned_2d.png`、`generation_drift_projection.png` |
| 9b | — | 表格，`runs/20260929T031927475103Z_baseline_ladder_full` |
| 9c | — | `configs/contrastive.json`、`src/loss.py` |

Run 目錄：v7.5 基線 `runs/20260923T024922889421Z_input_norm_5seed`、
v8.0 `runs/20260923T033747817275Z_augmentation_full`、
v8.1 `runs/20260923T045843638020Z_displacement_full`、
v9.0 `runs/20260923T071856507630Z_baseline_ladder_full`、
v10.0 `runs/20260929T031927475103Z_baseline_ladder_full`。
seeds 1042 / 2024 / 3407 / 42 / 123，lr 0.0003、100 epochs、batch 64。
