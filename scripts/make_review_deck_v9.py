#!/usr/bin/env python
"""Build reports/cdcnn-implementation-issues.pptx from the v9 review outline.

The outline is `reports/experiment-review-v9-outline.md`; slide wording is
written here so each slide can be terser than the outline. Figures come from
`reports/figures/` (ours) and `docs/paper/` (the paper's). python-pptx is used
because this machine has no node runtime.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
PAPER = ROOT / "docs" / "paper"
OUT = ROOT / "reports" / "cdcnn-implementation-issues.pptx"

CHARCOAL = RGBColor(0x2B, 0x2D, 0x42)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1C, 0x1C, 0x1E)
MUTED = RGBColor(0x66, 0x66, 0x6E)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
TINT = RGBColor(0xF3, 0xF4, 0xF6)
LINE = RGBColor(0xD9, 0xD8, 0xD4)
HEAD_FONT, BODY_FONT, EA_FONT = "Cambria", "Calibri", "Microsoft JhengHei"
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.6)


# ---------------------------------------------------------------- primitives
def _ea(run_, name: str = EA_FONT) -> None:
    rpr = run_._r.get_or_add_rPr()
    ea = rpr.find(qn("a:ea"))
    if ea is None:
        ea = rpr.makeelement(qn("a:ea"), {})
        rpr.append(ea)
    ea.set("typeface", name)


def run(par, text, *, size=16, bold=False, color=INK, font=BODY_FONT, italic=False):
    r = par.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    _ea(r)
    return r


def textbox(slide, x, y, w, h, *, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tf


def rect(slide, x, y, w, h, fill, *, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def picture(slide, path: Path, x, y, max_w, max_h, *, align="left"):
    """Fit an image inside a box, preserving aspect ratio."""
    with Image.open(path) as im:
        ar = im.size[0] / im.size[1]
    w, h = max_w, int(max_w / ar)
    if h > max_h:
        h, w = max_h, int(max_h * ar)
    if align == "center":
        x = x + (max_w - w) // 2
    elif align == "right":
        x = x + (max_w - w)
    return slide.shapes.add_picture(str(path), x, y, w, h)


def caption(slide, text, x, y, w, *, align=PP_ALIGN.LEFT):
    tf = textbox(slide, x, y, w, Inches(0.3), align=align)
    run(tf.paragraphs[0], text, size=10.5, color=MUTED, italic=True)


def blank(prs, dark=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if dark:
        rect(slide, 0, 0, W, H, CHARCOAL)
    return slide


def badge(slide, x, y, label, *, size=Inches(0.62), fill=ORANGE, text_size=15):
    s = rect(slide, x, y, size, size, fill, shape=MSO_SHAPE.OVAL)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p, label, size=text_size, bold=True, color=WHITE, font=HEAD_FONT)
    return s


def content_slide(prs, title, *, number=None, kicker=None):
    slide = blank(prs)
    x = M
    if number is not None:
        badge(slide, M, Inches(0.62), str(number))
        x = M + Inches(0.85)
    tf = textbox(slide, x, Inches(0.5), W - x - M, Inches(0.9))
    run(tf.paragraphs[0], title, size=30, bold=True, color=CHARCOAL, font=HEAD_FONT)
    if kicker:
        p = tf.add_paragraph()
        p.space_before = Pt(4)
        run(p, kicker, size=14, color=MUTED)
    return slide


def bullets(slide, items, x, y, w, h, *, size=15, gap=8):
    tf = textbox(slide, x, y, w, h)
    for i, item in enumerate(items):
        text, style = item if isinstance(item, tuple) else (item, "plain")
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        if style == "key":
            run(p, "▍ ", size=size, bold=True, color=ORANGE)
            run(p, text, size=size, bold=True, color=INK)
        elif style == "sub":
            run(p, "     ", size=size - 2)
            run(p, text, size=size - 2, color=MUTED)
        elif style == "head":
            run(p, text, size=size + 1, bold=True, color=CHARCOAL)
        else:
            run(p, "•  ", size=size, color=ORANGE)
            run(p, text, size=size, color=INK)
    return tf


def table(slide, headers, rows, x, y, w, *, col_fracs=None, size=12,
          row_h=Inches(0.36), bold_rows=(), aligns=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), x, y, w,
                                   row_h * (len(rows) + 1))
    t = shape.table
    tbl_pr = shape._element.graphic.graphicData.tbl
    # Drop the default banded style so the cells take our own fills.
    style_id = tbl_pr.find(qn("a:tblPr"))
    if style_id is not None:
        style_id.set("bandRow", "0")
        style_id.set("firstRow", "0")
    if col_fracs:
        for i, f in enumerate(col_fracs):
            t.columns[i].width = Emu(int(w * f))
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)

    def fill_cell(cell, text, *, bold=False, color=INK, bg=WHITE, align="left", sz=size):
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
        cell.margin_left = cell.margin_right = Inches(0.08)
        cell.margin_top = cell.margin_bottom = Inches(0.03)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = {"left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT,
                       "center": PP_ALIGN.CENTER}[align]
        run(p, text, size=sz, bold=bold, color=color)

    for c, head in enumerate(headers):
        fill_cell(t.cell(0, c), head, bold=True, color=WHITE, bg=CHARCOAL,
                  align=aligns[c])
    for r, row in enumerate(rows, start=1):
        bg = TINT if r in bold_rows else WHITE
        for c, text in enumerate(row):
            fill_cell(t.cell(r, c), text, bold=(r in bold_rows), bg=bg,
                      align=aligns[c])
    for r in range(len(rows) + 1):
        t.rows[r].height = row_h
    return shape


def stat(slide, x, y, w, value, label, *, color=ORANGE, value_size=30):
    rect(slide, x, y, w, Inches(1.7), TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = textbox(slide, x + Inches(0.2), y + Inches(0.12), w - Inches(0.4), Inches(0.65),
                 anchor=MSO_ANCHOR.MIDDLE)
    run(tf.paragraphs[0], value, size=value_size, bold=True, color=color, font=HEAD_FONT)
    tf = textbox(slide, x + Inches(0.2), y + Inches(0.85), w - Inches(0.4), Inches(0.8))
    run(tf.paragraphs[0], label, size=11.5, color=MUTED)


# ---------------------------------------------------------------- slides
def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 0. Title -----------------------------------------------------------------
    s = blank(prs, dark=True)
    tf = textbox(s, M, Inches(2.1), W - 2 * M, Inches(3))
    run(tf.paragraphs[0], "照論文實作 CDCNN 時發現的問題", size=42, bold=True,
        color=WHITE, font=HEAD_FONT)
    p = tf.add_paragraph(); p.space_before = Pt(16)
    run(p, "Source-only 協定下的復現：哪些對上了、哪些對不上、以及為什麼", size=20,
        color=RGBColor(0xCF, 0xD3, 0xDE))
    p = tf.add_paragraph(); p.space_before = Pt(30)
    run(p, "UCI Gas Sensor Array Drift，Batch 1 訓練、Batch 2–10 評估  ·  五個 seed  ·  "
           "runs 2026-09-22 至 10-05  ·  分支 exp/v12-drift-projection", size=12.5,
        color=RGBColor(0xA8, 0xAD, 0xBD))
    rect(s, M, Inches(5.6), Inches(0.62), Inches(0.62), ORANGE, shape=MSO_SHAPE.OVAL)
    tf = textbox(s, M + Inches(0.85), Inches(5.62), Inches(8), Inches(0.6),
                 anchor=MSO_ANCHOR.MIDDLE)
    run(tf.paragraphs[0], "橘色 = 論文的生成／擴充機制產生的東西；藍色 = 真實的 target 資料。"
                          "整份簡報都用這兩個顏色。", size=12, color=RGBColor(0xCF, 0xD3, 0xDE))

    # 1. What was reproduced ---------------------------------------------------
    s = content_slide(prs, "我們重現了什麼、沒重現什麼",
                      kicker="架構、兩個元件、Fig. 5 的畫法、以及論文自己的死類別都重現了；重現不了的是準確率的增益")
    table(s, ["論文的東西", "重現狀態", "我們的數字", "論文"], [
        ["純 ResNet 基線（Fig. 2）", "架構照圖重建，source-only 訓練", "0.5556 ± 0.0137", "0.6346"],
        ["資料擴充（Eq. 7，等向噪聲）", "照做", "−0.0216", "含在 +0.036 內"],
        ["特徵生成（Eqs. 8–16）+ L_MSE（S4）", "照做，Fig. 5 畫得出同樣的圖", "+0.0004", "含在 +0.036 內"],
        ["特徵生成，Eq. (14) 補上正負號與幅度（v10）", "照 v9 的診斷修了，仍不動", "+0.0023", "—"],
        ["對比損失 L_con（S5）", "已實作，決定不跑", "—", "+0.053"],
        ["Acetaldehyde 完全誤判（Fig. S3）", "重現了，和論文一樣", "0.000", "0.00"],
    ], M, Inches(1.75), W - 2 * M, col_fracs=(0.36, 0.32, 0.16, 0.16), size=12.5,
       row_h=Inches(0.42), bold_rows=(6,), aligns=["left", "left", "right", "right"])
    bullets(s, [
        "協定：只用 Batch 1（445 筆）訓練與選擇；Batch 2–10 在所有 checkpoint 凍結、雜湊之後才開，每次執行都有稽核",
        "統計量：target mean = 九個 batch 準確率的未加權平均，與論文相同；五個 seed、附 SD，可分辨門檻 0.009–0.022",
        "接下來每頁一個實作時撞到的問題，重點在問題 6、7、8",
    ], M, Inches(4.95), W - 2 * M, Inches(2.2), size=14)

    # 2. Problem 1: text vs figure --------------------------------------------
    s = content_slide(prs, "論文的文字與圖互相矛盾，資料支持圖", number=1,
                      kicker="能照做的都照做了；每個歧義都記錄了兩種讀法的差，最大的一個是 pooling")
    table(s, ["論文的地方", "文字說", "圖說／實作", "實測"], [
        ["Channel 數", "§5.2「1 到 128 逐步增加」→ 32/64/128/128/128", "Fig. 2 印 32/64/128/256/128，內層 512", "圖版高 +0.03 到 +0.05"],
        ["Pooling", "「doesn't apply the pooling layer」", "—", "加 GAP 掉 −0.078"],
        ["Eq. (S2)", "對 minibatch 取未加權平均", "最後一批只有 61 筆", "照 S2，並記錄兩種平均的差"],
        ["Eq. (S4)", "只有 1/B、1/n，沒有 1/C", "F.mse_loss 預設會除以 6", "照 S4，六項相加"],
        ["Eqs. (10)–(11)", "結果在 R^{B×CH}，那會讓重上色變恆等", "逐純量／channel／位置三種讀法", "三種都做，都約零"],
        ["Eq. (14)", "高斯可抽到負的尺度", "照論文不截斷", "逐位置負值比例 0.000"],
    ], M, Inches(1.75), Inches(7.9), col_fracs=(0.16, 0.34, 0.30, 0.20), size=11,
       row_h=Inches(0.62), aligns=["left", "left", "left", "left"])
    picture(s, PAPER / "fig2_cdcnn.png", Inches(8.85), Inches(1.75), Inches(3.9), Inches(2.6))
    caption(s, "論文 Fig. 2：channel 32/64/128/256/128，無 pooling", Inches(8.85), Inches(4.4), Inches(3.9))
    bullets(s, [
        ("Pooling 那句是關鍵句：加了全域平均池化就掉 0.078，比任何 CDCNN 元件的效應都大", "key"),
    ], Inches(8.85), Inches(4.9), Inches(3.9), Inches(1.5), size=13)

    # 3. Problem 2: label mapping ----------------------------------------------
    s = content_slide(prs, "資料集的標籤對不上論文的 Table 2", number=2,
                      kicker="不影響任何準確率（模型只看整數），但影響所有「某氣體怎樣」的敘述")
    picture(s, PAPER / "fig4a.png", M, Inches(1.7), Inches(6.0), Inches(2.8))
    caption(s, "論文的 PCA 總圖（batch1 面板），依氣體名稱上色", M, Inches(4.5), Inches(6.0))
    picture(s, FIG / "batch1_pca.png", Inches(6.9), Inches(1.7), Inches(5.85), Inches(2.8))
    caption(s, "我們的 Batch 1 PCA，只標整數標籤，避免圖帶著讀", Inches(6.9), Inches(4.5), Inches(5.85))
    bullets(s, [
        "資料集文件：1=Ethanol, 2=Ethylene, 3=Ammonia, 4=Acetaldehyde, 5=Acetone, 6=Toluene",
        "論文 Table 2 的 Batch 1 各氣體筆數（83/30/70/98/90/74）照欄位順序讀，要求標籤 1 有 83 筆；實際是 90。六欄只有一欄對得上",
        ("用 Batch 1 的 PCA 形狀對論文的圖，判定標籤 2 與 3 互換：採用 1=Ethanol, 2=Ammonia, 3=Ethylene, 4=Acetaldehyde, 5=Acetone, 6=Toluene", "key"),
        "前一分支沒有資料集文件，用筆數硬對，對應表是錯的；這裡的衝突只記錄、不宣稱解決",
    ], M, Inches(4.95), W - 2 * M, Inches(2.3), size=13.5, gap=5)

    # 4. Problem 3: Fig. S1 -------------------------------------------------------
    s = content_slide(prs, "Fig. S1 的曲線畫的是什麼？", number=3,
                      kicker="訓練曲線和 target 曲線都對不上；而且 source-only 下沒有停止規則")
    picture(s, PAPER / "figs1.png", M, Inches(1.7), Inches(6.3), Inches(2.5))
    caption(s, "論文 Fig. S1：(a) 準確率在 0.58／0.65／0.70 進入平台，(b) CDCNN 的 loss 停在 1.5", M, Inches(4.2), Inches(6.3))
    picture(s, FIG / "figS1_overlay.png", M, Inches(4.55), Inches(6.3), Inches(2.6))
    bullets(s, [
        ("我們的 Batch 1 訓練準確率 10 個 epoch 內就過 0.95，最後 1.000；target mean 落在 0.35–0.55", "key"),
        "若 S1 是訓練曲線，0.58 太低；若是 target 曲線，它暗示訓練時一直在看 target",
        "S1(b)：CDCNN 的 loss 比另外兩個高 1.0。多了 L_MSE 與 L_con 兩項，高是合理的，但我們量到的 L_MSE 最終只佔 loss 的 3%，那 1.0 幾乎全是對比損失",
        ("沒有 source-only 的停止規則", "head"),
        "5-fold CV、120 個 fold：held-out accuracy 與 loss 的峰值都落在 epoch 60–100，而 target 的峰值在 7／40／70，兩者反相關",
        "epoch 只能固定為 100，這是唯一不看 target 的做法",
    ], Inches(7.2), Inches(1.7), Inches(5.55), Inches(5.4), size=13, gap=6)

    # 5. Problem 4: Acetaldehyde ---------------------------------------------------
    s = content_slide(prs, "論文自己的 CDCNN 也死在 Acetaldehyde", number=4,
                      kicker="三個模型都是 0.00，都送進 Ethanol；所以差距全在其他五類")
    picture(s, PAPER / "figS3_confusion.png", M, Inches(1.7), Inches(6.2), Inches(2.95))
    caption(s, "論文 Fig. S3：CDCNN 把 100% 的 Acetaldehyde 判成 Ethanol", M, Inches(4.65), Inches(6.2))
    picture(s, FIG / "baseline-confusion_best.png", Inches(7.0), Inches(1.7), Inches(5.75), Inches(2.95), align="center")
    caption(s, "我們的基線混淆矩陣：Acetaldehyde 一樣是 0.00", Inches(7.0), Inches(4.65), Inches(5.75), align=PP_ALIGN.CENTER)
    table(s, ["類別", "論文 CDCNN", "論文 CDWC", "我們的基線"], [
        ["Ethanol", "0.97", "0.88", "0.67"], ["Ethylene", "0.95", "0.87", "0.62"],
        ["Ammonia", "0.91", "0.87", "0.86"], ["Acetaldehyde", "0.00", "0.00", "0.00"],
        ["Acetone", "0.94", "0.85", "0.51"], ["Toluene", "0.92", "0.27", "0.39"],
    ], M, Inches(5.05), Inches(6.2), col_fracs=(0.34, 0.22, 0.22, 0.22), size=11,
       row_h=Inches(0.3), bold_rows=(4,))
    bullets(s, [
        "原因是漂移，不是化學相似：Acetaldehyde 在 Batch 1 上 recall 1.000，target 上每個 batch 的質心都落到 Batch 1 的 Ethanol 位置",
        "它佔 target 14%，論文的 pooled 準確率因此只有約 0.80；0.7230 是在它為零的情況下達成的",
        ("與論文的差距在其他五類，尤其 Toluene（0.92 對 0.39）與 Acetone（0.94 對 0.51）", "key"),
    ], Inches(7.0), Inches(5.05), Inches(5.75), Inches(2.2), size=12.5, gap=5)

    # 6. Problem 5: augmentation ---------------------------------------------------
    s = content_slide(prs, "等向擴充是負的，有方向才是正的，但方向來自 target", number=5,
                      kicker="無方向的位移是負的，同樣的位移給了方向就是本專案唯一可分辨的增益")
    table(s, ["變體", "擴充", "target mean", "vs 基準", "可分辨"], [
        ["R-aug-t2", "沿漂移方向，T=2，無噪聲", "0.5770", "+0.0214", "是"],
        ["R-fig-logps", "無（基準）", "0.5556", "—", "—"],
        ["R-aug-paper", "論文 Eq. (7) 等向噪聲", "0.5340", "−0.0216", "否"],
    ], M, Inches(1.75), Inches(6.4), col_fracs=(0.2, 0.34, 0.18, 0.16, 0.12), size=11.5,
       row_h=Inches(0.38), bold_rows=(1,), aligns=["left", "left", "right", "right", "center"])
    bullets(s, [
        "方向是看過 target 才選的，T 與去噪聲也是：target-informed，只能算上界",
        "增益全是 Acetone +0.255 換 Ethylene −0.272；Ethylene 是漂移方向與共用方向最不對齊的類別（cos 0.451）",
        ("死類別仍是 0.003：Acetaldehyde 與 Ethanol 幾乎一起漂（cos 0.761），共用方向平移解不開兩者", "key"),
        "Source-only 的替代：Batch 1 自己兩個採集區塊的偏移能估到方向（cos 0.850）；v10 在 block 3 用過，輸入端還沒有",
    ], M, Inches(3.55), Inches(6.4), Inches(3.6), size=12.5, gap=6)
    picture(s, FIG / "drift_dimension.png", Inches(7.3), Inches(1.75), Inches(5.45), Inches(2.1))
    caption(s, "漂移的譜：PC1 佔 70.6%，90% 要 4 個成分；Ethylene 是例外", Inches(7.3), Inches(3.85), Inches(5.45))
    picture(s, FIG / "augmentation_pca.png", Inches(7.3), Inches(4.35), Inches(5.45), Inches(2.0))
    caption(s, "Batch 1 擴充前後的三維 PCA（v8.0）：等向噪聲只把雲撐大，到 target Acetaldehyde 重心的最近距離不變（2.36）；有方向的位移才縮到 1.46", Inches(7.3), Inches(6.3), Inches(5.45))

    # 7. Problem 6: Fig. 5 is circular ---------------------------------------------
    s = content_slide(prs, "論文 Fig. 5 的畫法在 source-only 下違規", number=6,
                      kicker="caption：「the feature generation block in one of the test domains (batch2)」")
    picture(s, PAPER / "fig5-feature-generation.png", M, Inches(1.7), Inches(6.0), Inches(3.5))
    caption(s, "論文 Fig. 5：區塊套在 batch 2 上（橘），與 batch 2 的原始特徵（藍）比", M, Inches(5.2), Inches(6.0))
    picture(s, FIG / "feature_generation.png", Inches(6.85), Inches(1.7), Inches(5.9), Inches(3.5))
    caption(s, "我們照同樣畫法畫：形狀對得上，橘色比藍色寬、往外延伸", Inches(6.85), Inches(5.2), Inches(5.9))
    bullets(s, [
        ("這張圖用了 target 資料。區塊套在 batch 2 上才有橘點；每個 batch 各自這樣畫，就是每個 batch 都獨立進了模型。訓練時區塊只看得到 Batch 1，推論時區塊根本不作用（Fig. 2）", "key"),
        ("這是循環論證。從 batch 2 生成的特徵當然靠近 batch 2，不管區塊做什麼；它證明不了區塊「涵蓋到未見過的域」", "key"),
        "正確的問法：區塊只作用在 Batch 1，生成出來的雲有沒有到後面的 batch？→ 下一頁",
    ], M, Inches(5.6), W - 2 * M, Inches(1.8), size=13, gap=4)

    # 8a. Problem 7: the 2D drift-aligned figure ------------------------------------
    s = content_slide(prs, "從 Batch 1 生成的特徵沒有覆蓋任何後續 batch", number=7,
                      kicker="X 軸 = 該（氣體, batch）的真實漂移方向 u，Y 軸 = 扣掉 u 之後殘差的第一主成分。漂移軸是建構出來的，雲沒有沿它走就藏不起來")
    picture(s, FIG / "generation_drift_aligned_2d_b2-7-10.png", M, Inches(1.75), W - 2 * M, Inches(5.0), align="center")
    caption(s, "灰 = Batch 1 原始特徵；橘 = 從 Batch 1 生成的人工特徵；藍 = target 原始特徵。直線是 Batch 1 質心與 target 質心。"
               "橘色疊在灰色上，藍色在另一邊，中間一段空隙（B2／B7／B10；九個 batch 的完整版在 reports/figures/）",
            M, Inches(6.8), W - 2 * M)

    # 8b. Problem 7: the one-dimensional numbers -----------------------------------
    s = content_slide(prs, "同一件事，沿漂移軸量成數字", number=7,
                      kicker="把每片雲投影到 u 上，source 質心在 0、target 質心在 1；51 個（氣體, batch）格")
    picture(s, FIG / "generation_drift_projection.png", M, Inches(1.7), W - 2 * M, Inches(3.3), align="center")
    table(s, ["一維量（沿 u）", "中位", "範圍"], [
        ["橘雲與藍雲的直方圖重疊", "0.00", "0.00–0.42"],
        ["橘雲寬度 ÷ 灰雲寬度", "1.01", "0.80–1.16"],
        ["區塊沿 u 的配對位移（漂移距離為單位）", "0.03", "0.007–0.166"],
    ], M, Inches(5.15), Inches(6.2), col_fracs=(0.6, 0.18, 0.22), size=12, row_h=Inches(0.36))
    bullets(s, [
        ("沿漂移軸，人工雲與 Batch 1 原始雲是同一片雲", "key"),
        "重疊 ≥ 0.10 的六格全是 target 自己散回 source 附近（B2 的 Ethanol、Ammonia、Acetone 等），不是橘雲走過去",
        "換另一個 checkpoint（R-aug-t2）重跑，每格差在小數第三位",
        "之前的 2D PCA 凸包、全維半徑、質心距離三種量法互相矛盾，各有盲點；這個一維量法才誠實",
    ], Inches(7.0), Inches(5.15), Inches(5.75), Inches(2.2), size=12, gap=4)

    # 9. Why ----------------------------------------------------------------------------
    s = content_slide(prs, "為什麼會這樣：Eq. (14) 沒有正負號，沿軸幅度太小",
                      kicker="區塊瞄的軸是對的，但它對稱地位移，而且落到單一漂移軸上的分量很小")
    cw = (W - 2 * M - Inches(0.3) * 3) / 4
    stat(s, M, Inches(1.75), cw, "3.1–4.6×", "擾動對真實漂移的 |cos|，相對隨機方向的虛無值。軸是對的", color=BLUE)
    stat(s, M + (cw + Inches(0.3)), Inches(1.75), cw, "−7.5%", "有號分量 ÷ 典型幅度，逐類平均。順著與逆著漂移機率相同")
    stat(s, M + 2 * (cw + Inches(0.3)), Inches(1.75), cw, "30×", "全空間模長「只差 2 倍」，投影到任一條漂移軸差 30 倍")
    stat(s, M + 3 * (cw + Inches(0.3)), Inches(1.75), cw, "+0.0004", "特徵生成的準確率效應，四格五 seed。效應下限的十分之一", color=CHARCOAL)
    bullets(s, [
        "7.6% 的擾動能量落在 51 維漂移張成空間，虛無值 0.3%：區塊確實偏好漂移的子空間，不是亂指",
        ("但 Eq. (14) 的高斯是對稱的。教分類器對 ±ε 鄰域不變，只會均勻撐大決策區域，不會把邊界搬向漂移後的資料", "key"),
        "網路確實學到了區塊要求的不變性：L_MSE 降 2–5 倍，λ 拉到論文上限 1.0 也只差 0.0002。但真實漂移有 94% 原封不動穿過 backbone",
        "與問題 5 是同一件事的兩面：無方向的位移是負的或零，方向就是全部效應，而 Eq. (14) 把它丟掉了",
    ], M, Inches(3.8), W - 2 * M, Inches(3.4), size=14, gap=9)

    # 9b. v10.0: the sign did not help either --------------------------------------------
    s = content_slide(prs, "把正負號補上也沒用（v10.0）", number=8,
                      kicker="方向用 Batch 1 自己兩個採集區塊的偏移，在 block 3 的 style 空間估，每個 epoch 重估，不碰 target")
    table(s, ["變體", "Eq. (14)", "target mean", "SD", "vs R-gen", "可分辨"], [
        ["R-gen-shift", "論文噪聲 + 2 個區塊偏移的定向位移", "0.5796", "0.0077", "+0.0023", "否"],
        ["R-gen", "論文原版，對稱高斯", "0.5773", "0.0069", "—", "—"],
        ["R-aug-t2", "無區塊", "0.5770", "0.0078", "−0.0003", "否"],
        ["R-gen-sign", "只修正負號，幅度照論文", "0.5767", "0.0115", "−0.0006", "否"],
    ], M, Inches(1.75), Inches(7.6), col_fracs=(0.15, 0.35, 0.14, 0.11, 0.14, 0.11), size=11,
       row_h=Inches(0.4), bold_rows=(1,), aligns=["left", "left", "right", "right", "right", "center"])
    table(s, ["類別", "cos 與真實漂移", "位移 / 漂移"], [
        ["Ethanol", "+0.769", "1.05"], ["Ammonia", "+0.558", "1.16"], ["Ethylene", "+0.305", "1.41"],
        ["Acetaldehyde", "+0.596", "0.83"], ["Acetone", "+0.693", "0.96"], ["Toluene", "+0.662", "0.73"],
        ["51 格平均", "+0.593", "—"],
    ], Inches(8.5), Inches(1.75), Inches(4.25), col_fracs=(0.4, 0.34, 0.26), size=11,
       row_h=Inches(0.3), bold_rows=(7,))
    caption(s, "事後在 style 空間量；隨機方向的虛無值 0.088（scripts/measure_signed_direction.py）", Inches(8.5), Inches(4.3), Inches(4.25))
    bullets(s, [
        ("方向是對的、幅度也對：六類 cos 全為正，位移是真實漂移的 0.73 到 1.41 倍。事前記錄的「可能反向」風險沒有發生", "key"),
        "每個 batch、每個類別的差都在 0.012 內；同樣的定向位移在輸入端（v8.1）能把 Acetone、Ethylene 各搬 0.25 以上",
        ("事前否證條件觸發。特徵生成區塊作為機制關掉了，與 Eq. (14) 有沒有正負號無關：它能碰到的子空間不是決策邊界會回應的那一個", "key"),
        "混淆：四格都帶輸入端的 R-aug-t2，shift 量到的是同方向位移之上的邊際效應",
    ], M, Inches(4.05), Inches(7.6), Inches(3.2), size=12.5, gap=6)

    # 9c. L_con: implemented, not run -------------------------------------------------------
    s = content_slide(prs, "最後一個元件 L_con（Eq. S5）：已實作，決定不跑",
                      kicker="論文三個元件裡唯一要求不變性的損失，但它要求的不變性對象已經被 v9、v10 量掉了")
    rect(s, M, Inches(1.75), Inches(6.0), Inches(1.5), TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = textbox(s, M + Inches(0.25), Inches(1.9), Inches(5.5), Inches(1.3), anchor=MSO_ANCHOR.MIDDLE)
    run(tf.paragraphs[0], "L_con = − Σᵢ (1/|P(i)|) Σₚ∈P(i) log [ exp(f(zᵢ)·f(zₚ)/τ) / Σₐ∈A(i) exp(f(zᵢ)·f(zₐ)/τ) ]",
        size=13, color=INK, font="Cambria")
    pp = tf.add_paragraph(); pp.space_before = Pt(6)
    run(pp, "Z = z_f ∪ z̄_f，生成特徵沿用原標籤；f 投影到單位球。τ 與 λ_con 論文都沒給。", size=11.5, color=MUTED)
    bullets(s, [
        ("做好的東西", "head"),
        "src/loss.py 的 supervised_contrastive；train.py 接上 Eq. (4) 第三項；四格 config（R-con、R-con-shift、R-con-t5 對 R-gen）含事前預測；單元測試與單一 minibatch 前向通過",
        ("為什麼不跑", "head"),
        ("它要求 z_f 對 z̄_f 不變。v9、v10 量過 z̄_f：從 Batch 1 生成的雲沒有離開 Batch 1（沿漂移軸重疊 0.00、寬度比 1.01）；連方向對、幅度對的 z̄_f 都只值 +0.0023。對一個沒離開 source 的擾動做不變性，不會變成對真實漂移的不變性", "key"),
        "前一分支在另一個 backbone 上量了五次：−0.005 到 +0.003",
    ], M, Inches(3.5), Inches(6.0), Inches(3.8), size=12.5, gap=5)
    cw = Inches(5.9)
    stat(s, Inches(7.0), Inches(1.75), Inches(2.85), "4.8", "L_con 的起始值 ≈ log(127)，是 L_ce 的兩倍多；乘 0.5 後主導前期梯度", color=CHARCOAL)
    stat(s, Inches(9.95), Inches(1.75), Inches(2.8), "1.5", "論文 Fig. S1(b) 裡 CDCNN 的 loss 平台，另外兩個是 0.5。與上面一致", color=CHARCOAL)
    bullets(s, [
        ("結論", "head"),
        ("論文三個元件：兩個量到零，第三個的前提已被量掉。復現到此為止", "key"),
        "config 與預測留檔；要補跑是一次 gpu-smoke 加一次 launch，約 15 分鐘",
        "否證條件也寫好了：R-con 若與 R-gen 可分辨，代表前一分支的五次零是 backbone 的問題，論文的 +0.053 在 source-only 下有立足點",
    ], Inches(7.0), Inches(3.7), Inches(5.75), Inches(3.6), size=12.5, gap=6)

    # 9d. v12: projecting the drift axis out of the input ------------------------------
    s = content_slide(prs, "我們自己的延伸：把 Batch 1 估到的漂移軸投影掉（v12）",
                      kicker="離開 CDCNN，留在 source-only：x′ = x − U Uᵀ x，U 來自 Batch 1 兩次採集的偏移，不帶擴充、不帶生成區塊")
    table(s, ["變體", "拿掉", "k", "target mean", "vs 基準", "可分辨"], [
        ["R-proj-eth", "Ethanol 的採集偏移（選它是看過 target）", "1", "0.5747", "+0.0191", "是（上界）"],
        ["R-proj-axis", "三個偏移的共同軸（頭條）", "1", "0.5684", "+0.0128", "否，門檻 0.0145"],
        ["R-proj-sub3", "三個偏移張成的空間", "3", "0.5610", "+0.0055", "否"],
        ["R-fig-logps", "不拿", "0", "0.5556", "—", "—"],
    ], M, Inches(1.75), Inches(7.4), col_fracs=(0.15, 0.37, 0.05, 0.14, 0.13, 0.16), size=10.5,
       row_h=Inches(0.38), bold_rows=(2,), aligns=["left", "left", "center", "right", "right", "left"])
    table(s, ["機制指標", "axis", "sub3", "eth"], [
        ["真實漂移幅度保留", "0.84", "0.52", "0.69"],
        ["類內半徑保留", "0.58", "0.50", "0.74"],
        ["Acetaldehyde 落在 Ethanol 上", "7/9", "8/9", "8/9"],
    ], Inches(8.3), Inches(1.75), Inches(4.45), col_fracs=(0.49, 0.17, 0.17, 0.17), size=11,
       row_h=Inches(0.38))
    picture(s, FIG / "batch1_drift_axis.png", M, Inches(3.95), Inches(7.4), Inches(2.6))
    caption(s, "Batch 1 自己的 PCA 平面：氣體沿 PC1 分開，兩次採集沿 PC2 錯開；拿掉那條軸後 Batch 1 還分得開（CV 0.9709 對 0.9680）", M, Inches(6.55), Inches(7.4))
    bullets(s, [
        ("Batch 1 估得到的軸只裝 29% 的三年漂移能量，卻裝 42% 的類內散佈：它是方向估計，不是漂移本身", "key"),
        "要拿掉夠多漂移就得拿三維，而那三維裝 76% 的類間變異。漂移與類別資訊在同一個低維子空間裡糾在一起",
        "逐類別形狀與擴充一模一樣：Acetone +0.24、Ethylene −0.28。加樣本與刪座標撞到同一個天花板，天花板是資料的",
        ("兩個事前否證條件都觸發；source-only 定案仍是 0.5556", "key"),
    ], Inches(8.3), Inches(3.4), Inches(4.45), Inches(3.9), size=11.5, gap=5)

    # 10. Summary ---------------------------------------------------------------------
    s = content_slide(prs, "總結：機制都重現了，增益沒有",
                      kicker="兩個分支、兩個 backbone，兩個元件在 source-only 下都約零")
    table(s, ["元件", "本分支（五 seed）", "前一分支（另一 backbone）", "論文宣稱"], [
        ["資料擴充（論文等向）", "−0.0216", "−0.013 到 −0.009", "ResNet→CDWC +0.036"],
        ["資料擴充（有方向，target-informed）", "+0.0214", "—", "—"],
        ["特徵生成 + L_MSE（論文原版）", "+0.0004", "+0.001 到 +0.014", "（含在上列）"],
        ["特徵生成，Eq. (14) 加正負號與幅度（v10.0）", "+0.0023", "—", "—"],
        ["對比損失 L_con", "已實作，未跑", "−0.005 到 +0.003", "CDWC→CDCNN +0.053"],
        ["（非論文）投影掉 Batch 1 估的漂移軸（v12）", "+0.0128，不可分辨", "—", "—"],
    ], M, Inches(1.75), W - 2 * M, col_fracs=(0.36, 0.18, 0.24, 0.22), size=12.5,
       row_h=Inches(0.4), bold_rows=(3, 4), aligns=["left", "right", "right", "right"])
    bullets(s, [
        ("重現了的：架構、Eq. (7) 擴充、Eqs. (8)–(16) 生成、Fig. 5 的畫法、Acetaldehyde 的死亡", "head"),
        ("重現不了的：論文的準確率增益。source-only 定案 0.5556，所有 0.577 系列都是看過 target 的上界", "head"),
        "與論文的差距在 Acetaldehyde 之外的五類，來源不在這三個元件裡",
        "Fig. 5 用 target 資料畫圖、標籤對不上 Table 2、Fig. S1 的曲線來源不明，三件事都寫進了文件，只記錄、不推測動機",
    ], M, Inches(4.75), W - 2 * M, Inches(2.5), size=13.5, gap=7)

    # 10b. Literature comparison on the same setting -------------------------------------
    s = content_slide(prs, "文獻對照：Batch 1 訓練、Batch 2–10 測試",
                      kicker="平均準確率是九個 batch 的未加權平均，與我們的 target mean 相同；差別只在各方法看到多少 target 的資訊")
    rows = [
        ["不碰 target", "SVM-rbf，資料集原論文的基線", "Vergara 2012", "38.9"],
        ["不碰 target", "FCNN，無補償（NC）", "KD 論文 2025", "38.7"],
        ["不碰 target", "OSC 正交訊號校正", "Yi 2019 表", "56.5"],
        ["不碰 target", "我們的 ResNet，R-fig-logps", "本專案", "55.6"],
        ["不碰 target", "我們的投影，R-proj-axis（不可分辨）", "本專案 v12", "56.8"],
        ["不碰 target（宣稱）", "CDCNN 論文的 ResNet / CDWC / CDCNN", "論文 Table 3", "63.5 / 67.1 / 72.3"],
        ["看過 target 選方向", "我們的 R-aug-t2 / R-proj-eth（上界）", "本專案 v8.1 / v12", "57.7 / 57.5"],
        ["無標籤 target 樣本", "SVM-gfk / SVM-comgfk 測地流核", "Vergara 2012", "63.8 / 64.0"],
        ["無標籤 target 樣本", "ML-comgfk 流形正則化", "Vergara 2012", "67.3"],
        ["無標籤 target 樣本", "DRCA 子空間對齊", "Zhang 2017", "62.2"],
        ["無標籤 target 樣本", "D-DRCA 有判別力的子空間對齊", "Yi 2019", "73.8"],
        ["無標籤 target，半批調參", "KD-DM 知識蒸餾", "KD 論文 2025", "47.9"],
        ["兩個 source batch + 無標籤", "AMDS-PFFA 注意力多源域適應（加權平均）", "2024", "83.2"],
        ["每批 20 筆有標籤", "DAELM-S(20)", "Zhang 2015", "80.8"],
        ["每批 50 筆有標籤", "DAELM-T(50)", "Zhang 2015", "91.9"],
    ]
    table(s, ["用到的 target 資訊", "方法", "來源", "平均 B2–10 (%)"], rows,
          M, Inches(1.7), Inches(8.4), col_fracs=(0.25, 0.41, 0.17, 0.17), size=10.5,
          row_h=Inches(0.33), bold_rows=(4, 5), aligns=["left", "left", "left", "right"])
    bullets(s, [
        ("不碰 target 的方法沒有一個超過 60%", "key"),
        "我們的 55.6 比原論文的 SVM 高 17 個百分點，與 OSC 持平；CDCNN 論文的 63.5 是這一列唯一的例外，而它的 Fig. 5 已被抓到用了 target 資料",
        ("跨過 60% 的做法清一色是看 target 的無標籤樣本做子空間對齊，DRCA 那一家", "key"),
        "DRCA 做的事和 v12 的投影幾乎一樣，差別是方向來自 source 與 target 的平均差，而不是 Batch 1 自己猜。這正好對上 v12 的失敗原因",
        "80% 以上要每批幾十筆有標籤的校正樣本，或兩個以上的 source batch",
        ("結論：這份資料集該用的設定是「無標籤 target 樣本可用於對齊」，那是文獻主流，也有明確對照組", "key"),
    ], Inches(9.3), Inches(1.7), Inches(3.45), Inches(5.6), size=11, gap=5)

    # 11. Next steps (dark) -----------------------------------------------------------
    s = blank(prs, dark=True)
    tf = textbox(s, M, Inches(0.6), W - 2 * M, Inches(0.9))
    run(tf.paragraphs[0], "結論與留下的東西", size=34, bold=True, color=WHITE, font=HEAD_FONT)
    cards = [
        ("1", "復現的結論",
         "三個元件在 source-only 協定下沒有一個重現得出增益：等向擴充 −0.0216、特徵生成 +0.0004、補上正負號 +0.0023，L_con 的前提已被量掉。與論文的 0.11 差距在其他五類，不在這三個元件。"),
        ("2", "留著的東西",
         "L_con 的實作、四格 config、事前預測與否證條件都在版控裡，補跑是 15 分鐘。另一個沒跑的乾淨對照：R-gen-shift 不帶輸入擴充對 R-fig-logps。"),
        ("3", "如果要往前走",
         "文獻在同一設定下不碰 target 沒有人過 60%；過 60% 的全是用 target 無標籤樣本做子空間對齊（DRCA 62、D-DRCA 74）。v12 的投影程式把方向來源換成 target 平均差就是最簡單的版本。報告時明說是 unsupervised domain adaptation。"),
    ]
    cw = (W - 2 * M - Inches(0.4) * 2) / 3
    for i, (n, head, body) in enumerate(cards):
        x = M + i * (cw + Inches(0.4))
        rect(s, x, Inches(1.9), cw, Inches(4.4), RGBColor(0x3A, 0x3D, 0x55), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        badge(s, x + Inches(0.3), Inches(2.2), n, fill=CHARCOAL if i == 0 else (ORANGE if i == 1 else BLUE))
        tf = textbox(s, x + Inches(0.3), Inches(3.05), cw - Inches(0.6), Inches(0.8))
        run(tf.paragraphs[0], head, size=18, bold=True, color=WHITE, font=HEAD_FONT)
        tf = textbox(s, x + Inches(0.3), Inches(3.9), cw - Inches(0.6), Inches(2.3))
        run(tf.paragraphs[0], body, size=12.5, color=RGBColor(0xDD, 0xE0, 0xE8))

    # 12. Appendix -----------------------------------------------------------------------
    s = content_slide(prs, "附錄：圖檔、執行目錄、設定")
    table(s, ["頁", "論文的圖（docs/paper/）", "我們的圖（reports/figures/）"], [
        ["1", "fig2_cdcnn.png", "—"],
        ["2", "fig4a.png（batch1 面板）", "batch1_pca.png、batch1_pca_zoom23.png"],
        ["3", "figs1.png", "figS1_overlay.png"],
        ["4", "figS3_confusion.png", "baseline-confusion_best.png"],
        ["5", "—", "drift_dimension.png、augmentation_pca.png"],
        ["6", "fig5-feature-generation.png", "feature_generation.png"],
        ["7", "—", "generation_drift_aligned_2d.png、generation_drift_projection.png"],
        ["8", "—", "表格；run 20260929T031927475103Z"],
        ["L_con", "—", "configs/contrastive.json、src/loss.py"],
        ["v12", "—", "batch1_drift_axis.png；run 20261005T145146298993Z"],
        ["文獻", "—", "arXiv 1505.06405、1901.02321、2409.13167、2507.17071"],
    ], M, Inches(1.6), Inches(7.8), col_fracs=(0.09, 0.31, 0.60), size=10.5, row_h=Inches(0.34),
       aligns=["center", "left", "left"])
    bullets(s, [
        ("執行目錄", "head"),
        ("v7.5 基線  runs/20260923T024922889421Z_input_norm_5seed", "sub"),
        ("v8.0  runs/20260923T033747817275Z_augmentation_full", "sub"),
        ("v8.1  runs/20260923T045843638020Z_displacement_full", "sub"),
        ("v9.0  runs/20260923T071856507630Z_baseline_ladder_full", "sub"),
        ("v10.0  runs/20260929T031927475103Z_baseline_ladder_full", "sub"),
        ("v12.0  runs/20261005T145146298993Z_baseline_ladder_full", "sub"),
        ("設定", "head"),
        ("seeds 1042 / 2024 / 3407 / 42 / 123；lr 0.0003、100 epochs、batch 64；BatchNorm、flatten head、signed-log → per-sample", "sub"),
        ("文件", "head"),
        ("baseline.md、docs/v8-augmentation.md、docs/v9-feature-generation.md、docs/drift-geometry.md、docs/label-mapping.md、docs/early-stopping.md", "sub"),
    ], Inches(8.5), Inches(1.6), Inches(4.25), Inches(5.5), size=12, gap=3)

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
