#!/usr/bin/env python
"""Merge the implementation-issues deck into the lab deck reports/09-22.pptx.

Slides 1-11 of 09-22.pptx (the paper introduction) are kept untouched. Its
experiment section, slides 12-16, overlapped with reports/cdcnn-implementation-
issues.pptx and carried a few numbers that match no run, so it is replaced by
the merged section below, rebuilt on the deck's own layout (內文多_1: title
placeholder, body placeholder, slide-number field) and theme fonts.

Output: reports/09-22-merged.pptx. The original is not modified.
"""
from __future__ import annotations

import copy
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "09-22.pptx"
OUT = ROOT / "reports" / "09-22-merged.pptx"
FIG = ROOT / "reports" / "figures"
PAPER = ROOT / "docs" / "paper"

KEEP = 11                     # slides 1..11 of the source deck stay as they are
LAYOUT = "內文多_1"
BODY = (Inches(0.5), Inches(1.31), Inches(9.0), Inches(5.39))   # the layout's body box
INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x59, 0x59, 0x59)
NAVY = RGBColor(0x1F, 0x49, 0x7D)      # the theme's dk2
HEAD_BG = RGBColor(0xDC, 0xE6, 0xF1)
ROW_BG = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# ------------------------------------------------------------------ helpers
def drop_slides_after(prs, keep: int) -> None:
    id_list = prs.slides._sldIdLst
    for sld in list(id_list)[keep:]:
        prs.part.drop_rel(sld.rId)
        id_list.remove(sld)


def grab_number_field(prs):
    """The slide-number placeholder of an existing slide, cloned onto new ones."""
    for shape in prs.slides[KEEP - 1].shapes:
        if shape.is_placeholder and shape.placeholder_format.idx == 12:
            return copy.deepcopy(shape._element)
    raise RuntimeError("no slide-number placeholder to clone")


class Deck:
    def __init__(self, prs):
        self.prs = prs
        self.layout = next(l for l in prs.slide_layouts if l.name == LAYOUT)
        self.number = grab_number_field(prs)

    def slide(self, title: str, *, keep_body: bool):
        s = self.prs.slides.add_slide(self.layout)
        s.shapes.title.text = title
        if len(title) > 10:
            # The layout's title box holds one line; the deck's own long
            # titles (slide 13 of the source) shrink the font the same way.
            for r in s.shapes.title.text_frame.paragraphs[0].runs:
                r.font.size = Pt(28)
        body = next(sh for sh in s.placeholders if sh.placeholder_format.idx == 1)
        if not keep_body:
            body._element.getparent().remove(body._element)
            body = None
        field = copy.deepcopy(self.number)
        s.shapes._spTree.append(field)
        # The field's cached text is the old number; PowerPoint recomputes it,
        # LibreOffice does too, but keep the cache honest for other readers.
        for t in field.iter(qn("a:t")):
            t.text = str(len(self.prs.slides))
        return s, body


def bullets(body, items, *, size=18, sub_size=16):
    """Fill the body placeholder: ('text', level) pairs, level 0 or 1."""
    tf = body.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        text, level = item if isinstance(item, tuple) else (item, 0)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size if level == 0 else sub_size)
        if level == 0 and text.endswith("："):
            r.font.bold = True
    return tf


def textbox(slide, x, y, w, h, text, *, size=12, bold=False, color=INK,
            align=PP_ALIGN.LEFT, italic=False):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.03)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(4)
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return box


def picture(slide, path, x, y, max_w, max_h, *, align="center"):
    with Image.open(path) as im:
        ar = im.size[0] / im.size[1]
    w, h = max_w, int(max_w / ar)
    if h > max_h:
        h, w = max_h, int(max_h * ar)
    if align == "center":
        x = x + (max_w - w) // 2
    return slide.shapes.add_picture(str(path), x, y, w, h)


def table(slide, headers, rows, x, y, w, *, col_fracs=None, size=11,
          row_h=Inches(0.34), bold_rows=(), aligns=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), x, y, w,
                                   row_h * (len(rows) + 1))
    t = shape.table
    tbl_pr = shape._element.graphic.graphicData.tbl.find(qn("a:tblPr"))
    if tbl_pr is not None:
        tbl_pr.set("bandRow", "0")
        tbl_pr.set("firstRow", "0")
    if col_fracs:
        for i, f in enumerate(col_fracs):
            t.columns[i].width = Emu(int(w * f))
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)

    def fill(cell, text, *, bold, bg, align, color=INK):
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg
        cell.margin_left = cell.margin_right = Inches(0.06)
        cell.margin_top = cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = {"left": PP_ALIGN.LEFT, "right": PP_ALIGN.RIGHT,
                       "center": PP_ALIGN.CENTER}[align]
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color

    for c, head in enumerate(headers):
        fill(t.cell(0, c), head, bold=True, bg=HEAD_BG, align=aligns[c], color=NAVY)
    for r_i, row in enumerate(rows, start=1):
        bg = ROW_BG if r_i in bold_rows else WHITE
        for c, text in enumerate(row):
            fill(t.cell(r_i, c), text, bold=(r_i in bold_rows), bg=bg, align=aligns[c])
    for r_i in range(len(rows) + 1):
        t.rows[r_i].height = row_h
    return shape


def caption(slide, x, y, w, text):
    textbox(slide, x, y, w, Inches(0.3), text, size=10, color=MUTED, italic=True)


# ------------------------------------------------------------------ slides
def build() -> Path:
    prs = Presentation(SRC)
    drop_slides_after(prs, KEEP)
    deck = Deck(prs)
    X, Y, W, H = BODY

    # 12. executive summary ----------------------------------------------------
    s, body = deck.slide("CDCNN 重現與探討", keep_body=True)
    bullets(body, [
        "Executive Summary：",
        ("只用 Batch 1 訓練與選擇；Batch 2–10 在所有 checkpoint 凍結、雜湊之後才開，每次執行有稽核。五個 seed、100 epochs，比較統計量是九個 batch 準確率的未加權平均（target mean），與論文相同。", 1),
        ("架構照 Fig. 2 重建，source-only 基線 0.5556（論文 ResNet 0.6346）。", 1),
        ("論文三個元件逐一加上去量：等向擴充 −0.0216、特徵生成 +0.0004、補上正負號 +0.0023；對比損失已實作，前提已被量掉，決定不跑。", 1),
        ("另做一個不在論文裡的延伸：把 Batch 1 估到的漂移軸投影掉，+0.0128，不可分辨。", 1),
        "接下來每頁一個實作時撞到的問題，重點在 Fig. 5 的畫法、生成特徵有沒有覆蓋 target、以及為什麼。",
    ], size=18, sub_size=16)

    # 13. what was reproduced ----------------------------------------------------
    s, _ = deck.slide("重現了什麼、沒重現什麼", keep_body=False)
    table(s, ["論文的東西", "重現狀態", "我們的數字", "論文"], [
        ["純 ResNet 基線（Fig. 2）", "架構照圖重建，source-only 訓練", "0.5556 ± 0.0137", "0.6346"],
        ["資料擴充（Eq. 7，等向噪聲）", "照做", "−0.0216", "含在 +0.036 內"],
        ["特徵生成（Eqs. 8–16）+ L_MSE（S4）", "照做，Fig. 5 畫得出同樣的圖", "+0.0004", "含在 +0.036 內"],
        ["特徵生成，Eq. (14) 補上正負號與幅度", "照診斷修了，仍不動", "+0.0023", "—"],
        ["對比損失 L_con（S5）", "已實作，決定不跑", "—", "+0.053"],
        ["Acetaldehyde 完全誤判（Fig. S3）", "重現了，和論文一樣", "0.000", "0.00"],
    ], X, Y, W, col_fracs=(0.36, 0.32, 0.16, 0.16), size=13, row_h=Inches(0.58),
       bold_rows=(6,), aligns=["left", "left", "right", "right"])
    textbox(s, X, Y + Inches(4.25), W, Inches(1.2), [
        "機制都重現得出來，準確率的增益重現不出來。",
        "可分辨門檻 0.009–0.022（五個 seed、2 × 標準誤）；低於 0.005 的差不算效應。",
    ], size=15)

    # 14. text vs figure -----------------------------------------------------------
    s, _ = deck.slide("問題一：文字與圖互相矛盾，資料支持圖", keep_body=False)
    table(s, ["論文的地方", "文字說", "圖說／實作", "實測"], [
        ["Channel 數", "§5.2「1 到 128 逐步增加」", "Fig. 2 印 32/64/128/256/128", "圖版高 +0.03 到 +0.05"],
        ["Pooling", "「doesn't apply the pooling layer」", "—", "加 GAP 掉 −0.078"],
        ["Eq. (S2)", "對 minibatch 取未加權平均", "最後一批只有 61 筆", "照 S2，記錄差異"],
        ["Eq. (S4)", "沒有 1/C", "F.mse_loss 預設除以 6", "照 S4，六項相加"],
        ["Eqs. (10)–(11)", "統計量維度自相矛盾", "逐純量／channel／位置三種讀法", "三種都做，都約零"],
        ["Eq. (14)", "高斯可抽到負的尺度", "照論文不截斷", "負值比例 0.000"],
    ], X, Y, W, col_fracs=(0.16, 0.32, 0.30, 0.22), size=12, row_h=Inches(0.56),
       aligns=["left", "left", "left", "left"])
    textbox(s, X, Y + Inches(4.1), W, Inches(1.2), [
        "Pooling 那句是關鍵句：加了全域平均池化就掉 0.078，比任何 CDCNN 元件的效應都大。",
        "能照做的都照做了；沒有一個歧義能解釋後面的差距。",
    ], size=15)

    # 15. label mapping --------------------------------------------------------------
    s, _ = deck.slide("問題二：標籤對不上論文的 Table 2", keep_body=False)
    picture(s, PAPER / "fig4a.png", X, Y, Inches(4.4), Inches(2.1), align="left")
    picture(s, FIG / "batch1_pca.png", X + Inches(4.6), Y, Inches(4.4), Inches(2.1), align="left")
    caption(s, X, Y + Inches(2.1), Inches(4.4), "論文的 PCA 總圖（batch1 面板），依氣體名稱上色")
    caption(s, X + Inches(4.6), Y + Inches(2.1), Inches(4.4), "我們的 Batch 1 PCA，只標整數標籤")
    textbox(s, X, Y + Inches(2.5), W, Inches(2.8), [
        "• 資料集文件：1=Ethanol, 2=Ethylene, 3=Ammonia, 4=Acetaldehyde, 5=Acetone, 6=Toluene",
        "• 論文 Table 2 的 Batch 1 筆數（83/30/70/98/90/74）照欄位讀，要求標籤 1 有 83 筆；實際是 90，六欄只有一欄對得上",
        "• 用 Batch 1 的 PCA 形狀對論文的圖，判定標籤 2 與 3 互換：採用 1=Ethanol, 2=Ammonia, 3=Ethylene, 4=Acetaldehyde, 5=Acetone, 6=Toluene",
        "• 不影響任何準確率（模型只看整數），只影響「某氣體怎樣」的敘述；衝突只記錄、不宣稱解決",
    ], size=14)

    # 16. Fig. S1 ------------------------------------------------------------------------
    s, _ = deck.slide("問題三：Fig. S1 的曲線畫的是什麼", keep_body=False)
    picture(s, PAPER / "figs1.png", X, Y, Inches(4.4), Inches(1.8), align="left")
    picture(s, FIG / "figS1_overlay.png", X + Inches(4.6), Y, Inches(4.4), Inches(1.8), align="left")
    caption(s, X, Y + Inches(1.8), Inches(4.4), "論文 Fig. S1：準確率在 0.58／0.65／0.70 進入平台；CDCNN 的 loss 停在 1.5")
    caption(s, X + Inches(4.6), Y + Inches(1.8), Inches(4.4), "我們：Batch 1 訓練準確率到 1.000，target mean 0.35–0.55")
    textbox(s, X, Y + Inches(2.25), W, Inches(3.1), [
        "• 若 S1 是訓練曲線，0.58 太低；若是 target 曲線，它暗示訓練時一直在看 target",
        "• S1(b)：CDCNN 的 loss 比另外兩個高 1.0。我們量到的 L_MSE 最終只佔 loss 的 3%，那 1.0 幾乎全是對比損失",
        "• source-only 下沒有停止規則：5-fold CV、120 個 fold，held-out 準確率與 loss 的峰值都在 epoch 60–100，target 的峰值在 7／40／70，反相關",
        "• epoch 只能固定為 100，這是唯一不看 target 的做法",
    ], size=14)

    # 17. Acetaldehyde ---------------------------------------------------------------------
    s, _ = deck.slide("問題四：論文也死在 Acetaldehyde", keep_body=False)
    picture(s, PAPER / "figS3_confusion.png", X, Y, Inches(4.4), Inches(2.1), align="left")
    picture(s, FIG / "baseline-confusion_best.png", X + Inches(4.6), Y, Inches(4.4), Inches(2.1), align="left")
    caption(s, X, Y + Inches(2.1), Inches(4.4), "論文 Fig. S3：CDCNN 把 100% 的 Acetaldehyde 判成 Ethanol")
    caption(s, X + Inches(4.6), Y + Inches(2.1), Inches(4.4), "我們的基線混淆矩陣：Acetaldehyde 一樣是 0.00")
    table(s, ["類別", "論文 CDCNN", "論文 CDWC", "我們的基線"], [
        ["Ethanol", "0.97", "0.88", "0.67"], ["Ethylene", "0.95", "0.87", "0.62"],
        ["Ammonia", "0.91", "0.87", "0.86"], ["Acetaldehyde", "0.00", "0.00", "0.00"],
        ["Acetone", "0.94", "0.85", "0.51"], ["Toluene", "0.92", "0.27", "0.39"],
    ], X, Y + Inches(2.5), Inches(4.4), col_fracs=(0.37, 0.21, 0.21, 0.21), size=11,
       row_h=Inches(0.31), bold_rows=(4,))
    textbox(s, X + Inches(4.6), Y + Inches(2.5), Inches(4.4), Inches(2.8), [
        "• 原因是漂移，不是化學相似：Acetaldehyde 在 Batch 1 上 recall 1.000，到 target 每個 batch 的質心都落到 Batch 1 的 Ethanol 位置",
        "• 它佔 target 14%，論文的 pooled 準確率因此只有約 0.80；0.7230 是在它為零的情況下達成的",
        "• 與論文的差距在其他五類，尤其 Toluene（0.92 對 0.39）與 Acetone（0.94 對 0.51）",
    ], size=13)

    # 18. augmentation --------------------------------------------------------------------
    s, _ = deck.slide("問題五：等向擴充是負的，有方向才是正的", keep_body=False)
    table(s, ["變體", "擴充", "target mean", "vs 基準", "可分辨"], [
        ["R-aug-t2", "沿漂移方向，T=2，無噪聲", "0.5770", "+0.0214", "是"],
        ["R-fig-logps", "無（基準）", "0.5556", "—", "—"],
        ["R-aug-paper", "論文 Eq. (7) 等向噪聲", "0.5340", "−0.0216", "否"],
    ], X, Y, W, col_fracs=(0.2, 0.38, 0.16, 0.14, 0.12), size=12, row_h=Inches(0.38),
       bold_rows=(1,), aligns=["left", "left", "right", "right", "center"])
    picture(s, FIG / "augmentation_pca.png", X, Y + Inches(1.65), W, Inches(1.55))
    caption(s, X, Y + Inches(3.2), W, "Batch 1 擴充前後的三維 PCA：等向噪聲只把雲撐大，到 target Acetaldehyde 重心的最近距離不變（2.36）；有方向的位移才縮到 1.46")
    textbox(s, X, Y + Inches(3.6), W, Inches(1.8), [
        "• 方向是看過 target 才選的，T 與去噪聲也是：target-informed，只能算上界",
        "• 增益全是 Acetone +0.255 換 Ethylene −0.272；Ethylene 是漂移方向與共用方向最不對齊的類別（cos 0.451）",
        "• 死類別仍是 0.003：Acetaldehyde 與 Ethanol 幾乎一起漂（cos 0.761），共用方向的平移解不開兩者",
    ], size=14)

    # 19. Fig. 5 is circular ----------------------------------------------------------------
    s, _ = deck.slide("問題六：Fig. 5 的畫法違規", keep_body=False)
    picture(s, PAPER / "fig5-feature-generation.png", X, Y, Inches(4.4), Inches(2.55), align="left")
    picture(s, FIG / "feature_generation.png", X + Inches(4.6), Y, Inches(4.4), Inches(2.55), align="left")
    caption(s, X, Y + Inches(2.55), Inches(4.4), "論文 Fig. 5：區塊套在 batch 2 上（橘），與 batch 2 的原始特徵（藍）比")
    caption(s, X + Inches(4.6), Y + Inches(2.55), Inches(4.4), "我們照同樣畫法畫：形狀對得上")
    textbox(s, X, Y + Inches(3.0), W, Inches(2.4), [
        "• 這張圖用了 target 資料：橘點是把區塊套在 batch 2 上才有的，每個 batch 各自這樣畫，就是每個 batch 都獨立進了模型。訓練時區塊只看得到 Batch 1，推論時區塊根本不作用（Fig. 2）",
        "• 這是循環論證：從 batch 2 生成的特徵當然靠近 batch 2，不管區塊做什麼；它證明不了區塊「涵蓋到未見過的域」",
        "• 正確的問法：區塊只作用在 Batch 1，生成出來的雲有沒有到後面的 batch？→ 下一頁",
    ], size=14)

    # 20. the drift-aligned figure --------------------------------------------------------
    s, _ = deck.slide("問題七：生成特徵沒有覆蓋任何後續 batch", keep_body=False)
    picture(s, FIG / "generation_drift_aligned_2d_b2-7-10.png", X, Y, W, Inches(3.5))
    textbox(s, X, Y + Inches(3.55), W, Inches(1.9), [
        "• X 軸為該氣體從 Batch 1 到 target batch 的漂移方向（質心差），Y 軸為扣除漂移方向後的第一主成分；漂移軸是建構出來的，雲沒有沿它走就藏不起來",
        "• 灰 = Batch 1 原始特徵，橘 = 從 Batch 1 生成的人工特徵，藍 = target 原始特徵。若特徵生成有效，橘色應覆蓋藍色；實際上橘色只疊在灰色上，藍色在另一邊，中間一段空隙",
        "• 在這樣的情況下，對比損失拉近 z_f 與 z̄_f 沒有意義：生成特徵本來就和 source 特徵幾乎相同",
    ], size=12)

    # 21. the one-dimensional numbers -----------------------------------------------------
    s, _ = deck.slide("同一件事，沿漂移軸量成數字", keep_body=False)
    picture(s, FIG / "generation_drift_projection.png", X, Y, W, Inches(2.5))
    table(s, ["一維量（沿 u；source 質心 0，target 質心 1）", "中位", "範圍"], [
        ["橘雲與藍雲的直方圖重疊", "0.00", "0.00–0.42"],
        ["橘雲寬度 ÷ 灰雲寬度", "1.01", "0.80–1.16"],
        ["區塊沿 u 的配對位移（漂移距離為單位）", "0.03", "0.007–0.166"],
    ], X, Y + Inches(2.65), Inches(5.6), col_fracs=(0.62, 0.16, 0.22), size=11, row_h=Inches(0.32))
    textbox(s, X + Inches(5.8), Y + Inches(2.65), Inches(3.2), Inches(2.6), [
        "• 51 個（氣體, batch）格；沿漂移軸，人工雲與 Batch 1 原始雲是同一片雲",
        "• 重疊 ≥ 0.10 的六格全是 target 自己散回 source 附近",
        "• 換另一個 checkpoint 重跑，每格差在小數第三位",
    ], size=13)

    # 22. why ------------------------------------------------------------------------------
    s, body = deck.slide("為什麼：Eq. (14) 沒有正負號", keep_body=True)
    bullets(body, [
        "軸是對的：擾動對真實漂移的 |cos| 是隨機虛無值的 3.1–4.6 倍；7.6% 的擾動能量落在 51 維漂移張成空間（虛無值 0.3%）",
        "正負號沒有：有號分量 ÷ 典型幅度，逐類平均 −7.5%。Eq. (14) 的高斯是對稱的，順著與逆著漂移機率相同",
        ("教分類器對 ±ε 鄰域不變，只會均勻撐大決策區域，不會把邊界搬向漂移後的資料", 1),
        "沿軸幅度太小：全空間模長「只差 2 倍」是誤導，投影到任一條漂移軸差 30 倍",
        "網路確實學到區塊要求的不變性：L_MSE 降 2–5 倍，λ 拉到上限 1.0 也只差 0.0002；但真實漂移有 94% 原封不動穿過 backbone",
        "與問題五是同一件事的兩面：無方向的位移是負的或零，方向就是全部效應，而 Eq. (14) 把它丟掉了。準確率效應 +0.0004",
    ], size=18, sub_size=16)

    # 23. v10 ------------------------------------------------------------------------------
    s, _ = deck.slide("問題八：補上正負號也沒用（v10）", keep_body=False)
    table(s, ["變體", "Eq. (14)", "target mean", "SD", "vs R-gen", "可分辨"], [
        ["R-gen-shift", "論文噪聲 + 2 個區塊偏移的定向位移", "0.5796", "0.0077", "+0.0023", "否"],
        ["R-gen", "論文原版，對稱高斯", "0.5773", "0.0069", "—", "—"],
        ["R-aug-t2", "無區塊", "0.5770", "0.0078", "−0.0003", "否"],
        ["R-gen-sign", "只修正負號，幅度照論文", "0.5767", "0.0115", "−0.0006", "否"],
    ], X, Y, W, col_fracs=(0.16, 0.36, 0.14, 0.1, 0.13, 0.11), size=11, row_h=Inches(0.36),
       bold_rows=(1,), aligns=["left", "left", "right", "right", "right", "center"])
    table(s, ["類別", "cos 與真實漂移", "位移 / 漂移"], [
        ["Ethanol", "+0.769", "1.05"], ["Ammonia", "+0.558", "1.16"], ["Ethylene", "+0.305", "1.41"],
        ["Acetaldehyde", "+0.596", "0.83"], ["Acetone", "+0.693", "0.96"], ["Toluene", "+0.662", "0.73"],
        ["51 格平均", "+0.593", "—"],
    ], X, Y + Inches(2.0), Inches(3.8), col_fracs=(0.42, 0.33, 0.25), size=11, row_h=Inches(0.3), bold_rows=(7,))
    textbox(s, X + Inches(4.0), Y + Inches(2.0), Inches(5.0), Inches(3.3), [
        "• 方向用 Batch 1 自己兩個採集區塊的偏移，在 block 3 的 style 空間估，每個 epoch 重估，不碰 target",
        "• 方向是對的、幅度也對：六類 cos 全為正（虛無值 0.088），位移是真實漂移的 0.73 到 1.41 倍",
        "• 每個 batch、每個類別的差都在 0.012 內；同樣的定向位移在輸入端（v8.1）能把 Acetone、Ethylene 各搬 0.25 以上",
        "• 事前否證條件觸發。特徵生成區塊作為機制關掉了，與 Eq. (14) 有沒有正負號無關：它能碰到的子空間不是決策邊界會回應的那一個",
    ], size=13)

    # 24. L_con ------------------------------------------------------------------------------
    s, body = deck.slide("L_con（Eq. S5）：已實作，決定不跑", keep_body=True)
    bullets(body, [
        "S5 是標準的 Supervised Contrastive Loss，作用在 z_f ∪ z̄_f，生成特徵沿用原標籤，f 投影到單位球；τ 與 λ_con 論文都沒給",
        "做好的東西：損失函數、Eq. (4) 第三項、四格 config（R-con、R-con-shift、R-con-t5 對 R-gen）含事前預測；單元測試與單一 minibatch 前向通過",
        "為什麼不跑：它要求 z_f 對 z̄_f 不變，而 z̄_f 沒有離開 Batch 1（沿漂移軸重疊 0.00、寬度比 1.01），連方向對、幅度對的 z̄_f 都只值 +0.0023。對一個沒離開 source 的擾動做不變性，不會變成對真實漂移的不變性",
        ("前一分支在另一個 backbone 上量了五次：−0.005 到 +0.003", 1),
        "L_con 起始值 ≈ log(127) ≈ 4.8，是 L_ce 的兩倍多，乘 0.5 後主導前期梯度；論文 Fig. S1(b) 的 CDCNN loss 平台 1.5 與此一致",
        "config 與預測留檔，補跑是 15 分鐘。否證條件：R-con 若與 R-gen 可分辨，代表前一分支的五次零是 backbone 的問題",
    ], size=17, sub_size=15)

    # 25. v12 ------------------------------------------------------------------------------
    s, _ = deck.slide("延伸：投影掉漂移軸（v12）", keep_body=False)
    table(s, ["變體", "拿掉", "k", "target mean", "vs 基準", "可分辨"], [
        ["R-proj-eth", "Ethanol 的採集偏移（看過 target）", "1", "0.5747", "+0.0191", "是（上界）"],
        ["R-proj-axis", "三個偏移的共同軸（頭條）", "1", "0.5684", "+0.0128", "否，門檻 0.0145"],
        ["R-proj-sub3", "三個偏移張成的空間", "3", "0.5610", "+0.0055", "否"],
        ["R-fig-logps", "不拿", "0", "0.5556", "—", "—"],
    ], X, Y, W, col_fracs=(0.16, 0.36, 0.06, 0.14, 0.12, 0.16), size=11, row_h=Inches(0.36),
       bold_rows=(2,), aligns=["left", "left", "center", "right", "right", "left"])
    picture(s, FIG / "batch1_drift_axis.png", X, Y + Inches(1.95), W, Inches(1.65))
    caption(s, X, Y + Inches(3.6), W, "Batch 1 自己的 PCA 平面：氣體沿 PC1 分開，兩次採集沿 PC2 錯開；拿掉那條軸後 Batch 1 還分得開（CV 0.9709 對 0.9680）")
    textbox(s, X, Y + Inches(3.95), W, Inches(1.5), [
        "• x′ = x − U Uᵀ x，U 來自 Batch 1 兩次採集的偏移，不帶擴充、不帶生成區塊，對照是定案基準",
        "• 那條軸只裝 29% 的三年漂移能量，卻裝 42% 的類內散佈：它是方向估計，不是漂移本身。拿三維就拿掉 76% 類間變異，漂移與類別資訊糾在同一個低維子空間裡",
        "• 逐類別形狀與擴充一模一樣（Acetone +0.24、Ethylene −0.28）；Acetaldehyde 仍落在 Ethanol。兩個否證條件都觸發，source-only 定案仍是 0.5556",
    ], size=13)

    # 26. results table --------------------------------------------------------------------
    s, _ = deck.slide("實驗結果", keep_body=False)
    table(s, ["元件", "本分支（五 seed）", "前一分支（另一 backbone）", "論文宣稱"], [
        ["資料擴充（論文等向）", "−0.0216", "−0.013 到 −0.009", "ResNet→CDWC +0.036"],
        ["資料擴充（有方向，target-informed）", "+0.0214", "—", "—"],
        ["特徵生成 + L_MSE（論文原版）", "+0.0004", "+0.001 到 +0.014", "（含在上列）"],
        ["特徵生成，Eq. (14) 加正負號與幅度（v10）", "+0.0023", "—", "—"],
        ["對比損失 L_con", "已實作，未跑", "−0.005 到 +0.003", "CDWC→CDCNN +0.053"],
        ["（非論文）投影掉 Batch 1 估的漂移軸（v12）", "+0.0128，不可分辨", "—", "—"],
    ], X, Y, W, col_fracs=(0.35, 0.21, 0.24, 0.2), size=13, row_h=Inches(0.52),
       bold_rows=(3, 4), aligns=["left", "right", "right", "right"])
    textbox(s, X, Y + Inches(3.8), W, Inches(1.5), [
        "重現了的：架構、Eq. (7) 擴充、Eqs. (8)–(16) 生成、Fig. 5 的畫法、Acetaldehyde 的死亡",
        "重現不了的：論文的準確率增益。source-only 定案 0.5556，所有 0.577 系列都是看過 target 的上界",
        "• 與論文的差距在 Acetaldehyde 之外的五類，來源不在這三個元件裡",
    ], size=14)

    # 27. literature -----------------------------------------------------------------------
    s, _ = deck.slide("文獻對照：Batch 1 訓練，Batch 2–10 測試", keep_body=False)
    table(s, ["用到的 target 資訊", "方法", "來源", "平均 B2–10 (%)"], [
        ["不碰 target", "SVM-rbf，資料集原論文的基線", "Vergara 2012", "38.9"],
        ["不碰 target", "OSC 正交訊號校正", "Yi 2019 表", "56.5"],
        ["不碰 target", "我們的 ResNet，R-fig-logps", "本專案", "55.6"],
        ["不碰 target", "我們的投影，R-proj-axis（不可分辨）", "本專案 v12", "56.8"],
        ["不碰 target（宣稱）", "CDCNN 論文的 ResNet / CDWC / CDCNN", "論文 Table 3", "63.5 / 67.1 / 72.3"],
        ["看過 target 選方向", "我們的 R-aug-t2 / R-proj-eth（上界）", "本專案", "57.7 / 57.5"],
        ["無標籤 target 樣本", "SVM-comgfk / ML-comgfk", "Vergara 2012", "64.0 / 67.3"],
        ["無標籤 target 樣本", "DRCA / D-DRCA 子空間對齊", "Zhang 2017 / Yi 2019", "62.2 / 73.8"],
        ["無標籤 target，半批調參", "KD-DM 知識蒸餾（FCNN 無補償 38.7）", "2025", "47.9"],
        ["兩個 source + 無標籤", "AMDS-PFFA 多源域適應（加權平均）", "2024", "83.2"],
        ["每批 20 / 50 筆有標籤", "DAELM-S(20) / DAELM-T(50)", "Zhang 2015", "80.8 / 91.9"],
    ], X, Y, W, col_fracs=(0.24, 0.42, 0.18, 0.16), size=11, row_h=Inches(0.35),
       bold_rows=(3, 4), aligns=["left", "left", "left", "right"])
    textbox(s, X, Y + Inches(4.3), W, Inches(1.1), [
        "• 不碰 target 的方法沒有一個超過 60%；我們的 55.6 比原論文的 SVM 高 17 個百分點，CDCNN 論文的 63.5 是唯一例外",
        "• 跨過 60% 的做法清一色是看 target 的無標籤樣本做子空間對齊（DRCA 那一家）；80% 以上要有標籤的校正樣本或多個 source batch",
    ], size=12)

    # 28. conclusion -------------------------------------------------------------------------
    s, body = deck.slide("結論", keep_body=True)
    bullets(body, [
        "消融實驗的結果：在嚴格 source-only 下，資料擴充與特徵生成都不能有效提高準確率；對比損失要求的不變性對象（生成特徵）沒有離開 source，前提已被量掉，故未執行。論文三個元件沒有一個重現得出增益",
        "與論文的 0.11 差距在 Acetaldehyde 之外的五類，不在這三個元件；Fig. 5 用 target 資料畫圖、標籤對不上 Table 2、Fig. S1 的曲線來源不明，三件事只記錄、不推測動機",
        "不依靠 target domain leakage 的模型是現實中比較需要的種類，但文獻在同一設定下不碰 target 沒有人過 60%；這份資料集該用的設定是「無標籤 target 樣本可用於對齊」，報告時明說是 unsupervised domain adaptation",
        "UCI Gas drift dataset 中的 Acetaldehyde 在 target 上必然誤判成 Ethanol，論文也救不回；任何後續結果都應分「五類」與「六類」報告",
    ], size=18, sub_size=16)

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
