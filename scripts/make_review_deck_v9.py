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
           "runs 2026-09-22 至 09-23  ·  分支 exp/v7-redesign", size=12.5,
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
        ["對比損失 L_con（S5）", "尚未", "—", "+0.053"],
        ["Acetaldehyde 完全誤判（Fig. S3）", "重現了，和論文一樣", "0.000", "0.00"],
    ], M, Inches(1.75), W - 2 * M, col_fracs=(0.34, 0.34, 0.16, 0.16), size=13,
       row_h=Inches(0.46), bold_rows=(5,), aligns=["left", "left", "right", "right"])
    bullets(s, [
        "協定：只用 Batch 1（445 筆）訓練與選擇；Batch 2–10 在所有 checkpoint 凍結、雜湊之後才開，每次執行都有稽核",
        "統計量：target mean = 九個 batch 準確率的未加權平均，與論文相同；五個 seed、附 SD，可分辨門檻 0.009–0.022",
        "接下來每頁一個實作時撞到的問題，重點在第 6、7 頁",
    ], M, Inches(4.85), W - 2 * M, Inches(2.2), size=14)

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
        "Source-only 的替代：Batch 1 自己兩個採集區塊的偏移能估到方向（cos 0.850），尚未拿來訓練",
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

    # 10. Summary ---------------------------------------------------------------------
    s = content_slide(prs, "總結：機制都重現了，增益沒有",
                      kicker="兩個分支、兩個 backbone，兩個元件在 source-only 下都約零")
    table(s, ["元件", "本分支（五 seed）", "前一分支（另一 backbone）", "論文宣稱"], [
        ["資料擴充（論文等向）", "−0.0216", "−0.013 到 −0.009", "ResNet→CDWC +0.036"],
        ["資料擴充（有方向，target-informed）", "+0.0214", "—", "—"],
        ["特徵生成 + L_MSE", "+0.0004", "+0.001 到 +0.014", "（含在上列）"],
        ["對比損失 L_con", "未做", "−0.005 到 +0.003", "CDWC→CDCNN +0.053"],
    ], M, Inches(1.75), W - 2 * M, col_fracs=(0.34, 0.2, 0.24, 0.22), size=13,
       row_h=Inches(0.44), bold_rows=(3,), aligns=["left", "right", "right", "right"])
    bullets(s, [
        ("重現了的：架構、Eq. (7) 擴充、Eqs. (8)–(16) 生成、Fig. 5 的畫法、Acetaldehyde 的死亡", "head"),
        ("重現不了的：論文的準確率增益。source-only 定案 0.5556，所有 0.577 系列都是看過 target 的上界", "head"),
        "Fig. 5 用 target 資料畫圖、標籤對不上 Table 2、Fig. S1 的曲線來源不明，三件事都寫進了文件，只記錄、不推測動機",
    ], M, Inches(4.35), W - 2 * M, Inches(2.8), size=14, gap=9)

    # 11. Next steps (dark) -----------------------------------------------------------
    s = blank(prs, dark=True)
    tf = textbox(s, M, Inches(0.6), W - 2 * M, Inches(0.9))
    run(tf.paragraphs[0], "下一步", size=34, bold=True, color=WHITE, font=HEAD_FONT)
    cards = [
        ("1", "對比損失 L_con（S5）",
         "最後一個元件，也是三個裡唯一有不變性機制的。forward_pair 已回傳 (z_f, z̄_f)，train.py 已預留 Eq. (4) 的位置。事前預測與否證條件寫進 config 再跑。"),
        ("2", "給 Eq. (14) 加正負號（待決定）",
         "方向用 Batch 1 自己的採集區塊偏移（source-only，對真實漂移 cos 0.850）。直接針對上一頁量到的失效機制，而且不看 target。"),
        ("3", "分開報告五類與六類",
         "Acetaldehyde 是所有共用方向方法的天花板，論文也沒救回它。任何後續結果都應同時給含它與不含它的 target mean。"),
    ]
    cw = (W - 2 * M - Inches(0.4) * 2) / 3
    for i, (n, head, body) in enumerate(cards):
        x = M + i * (cw + Inches(0.4))
        rect(s, x, Inches(1.9), cw, Inches(4.4), RGBColor(0x3A, 0x3D, 0x55), shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        badge(s, x + Inches(0.3), Inches(2.2), n, fill=ORANGE if i < 2 else BLUE)
        tf = textbox(s, x + Inches(0.3), Inches(3.05), cw - Inches(0.6), Inches(0.8))
        run(tf.paragraphs[0], head, size=18, bold=True, color=WHITE, font=HEAD_FONT)
        tf = textbox(s, x + Inches(0.3), Inches(3.9), cw - Inches(0.6), Inches(2.2))
        run(tf.paragraphs[0], body, size=13, color=RGBColor(0xDD, 0xE0, 0xE8))

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
    ], M, Inches(1.6), Inches(7.6), col_fracs=(0.08, 0.4, 0.52), size=11, row_h=Inches(0.34),
       aligns=["center", "left", "left"])
    bullets(s, [
        ("執行目錄", "head"),
        ("v7.5 基線  runs/20260923T024922889421Z_input_norm_5seed", "sub"),
        ("v8.0  runs/20260923T033747817275Z_augmentation_full", "sub"),
        ("v8.1  runs/20260923T045843638020Z_displacement_full", "sub"),
        ("v9.0  runs/20260923T071856507630Z_baseline_ladder_full", "sub"),
        ("設定", "head"),
        ("seeds 1042 / 2024 / 3407 / 42 / 123；lr 0.0003、100 epochs、batch 64；BatchNorm、flatten head、signed-log → per-sample", "sub"),
        ("文件", "head"),
        ("baseline.md、docs/v8-augmentation.md、docs/v9-feature-generation.md、docs/drift-geometry.md、docs/label-mapping.md、docs/early-stopping.md", "sub"),
    ], Inches(8.5), Inches(1.6), Inches(4.25), Inches(5.5), size=12, gap=3)

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
