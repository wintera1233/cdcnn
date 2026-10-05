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
def kicker(slide, text):
    """One plain sentence under the title: the slide's message for a reader
    who has not read the paper."""
    textbox(slide, Inches(1.45), Inches(1.22), Inches(8.05), Inches(0.35), text,
            size=12, color=NAVY, bold=True)


TOP = Inches(1.62)   # content starts below the kicker


def build() -> Path:
    prs = Presentation(SRC)
    drop_slides_after(prs, KEEP)
    deck = Deck(prs)
    X, _, W, _ = BODY
    Y = TOP

    # 12. what we built --------------------------------------------------------------
    s, _ = deck.slide("實驗設計：照 Fig. 2 重建架構", keep_body=False)
    kicker(s, "先把論文的網路照圖做出來，只用第一批資料訓練，再看它在後面九批的表現。")
    picture(s, PAPER / "fig2_cdcnn.png", X, Y, Inches(4.6), Inches(2.7), align="left")
    caption(s, X, Y + Inches(2.7), Inches(4.6), "論文 Fig. 2：五個一維 ResNet 區塊，攤平後接全連接層；資料擴充與特徵生成只在訓練時作用")
    textbox(s, X + Inches(4.8), Y, Inches(4.2), Inches(4.0), [
        "架構：五個 ResNet 區塊（通道 32/64/128/256/128）、不做 pooling、攤平後 FC128 → BatchNorm → FC6，完全照圖",
        "資料：UCI 漂移資料集，128 維特徵、六種氣體、十批，橫跨三年",
        "規則（source-only）：訓練、調參、選模型只用 Batch 1 的 445 筆；Batch 2–10 要等所有模型存檔、雜湊之後才打開，每次執行都有稽核",
        "指標：九批各自的準確率再取平均（target mean），與論文 Table 3 相同；五個隨機種子，附標準差",
    ], size=13)
    textbox(s, X, Y + Inches(3.15), Inches(4.6), Inches(1.6), [
        "三個元件一個一個加上去量：",
        "① 資料擴充　② 特徵生成 + L_MSE　③ 對比損失 L_con",
        "每加一個，問兩件事：準確率動了多少？機制有沒有真的發生？",
    ], size=13)

    # 13. the baseline ---------------------------------------------------------------
    s, _ = deck.slide("基線結果：0.5556，論文的 ResNet 0.6346", keep_body=False)
    kicker(s, "差 0.08。先看錯在哪：有一種氣體整批判錯，而論文自己的模型也一樣。")
    picture(s, FIG / "baseline-confusion_best.png", X, Y, Inches(4.5), Inches(2.9), align="left")
    caption(s, X, Y + Inches(2.9), Inches(4.5), "我們的混淆矩陣（Batch 2–10 合併）：Acetaldehyde 那一列全部判成 Ethanol")
    picture(s, PAPER / "figS3_confusion.png", X + Inches(4.7), Y, Inches(4.3), Inches(2.1), align="left")
    caption(s, X + Inches(4.7), Y + Inches(2.1), Inches(4.3), "論文 Fig. S3：CDCNN 也把 100% 的 Acetaldehyde 判成 Ethanol")
    textbox(s, X + Inches(4.7), Y + Inches(2.5), Inches(4.3), Inches(2.5), [
        "• Acetaldehyde 在 Batch 1 上分得很好（recall 1.000），到了後面的批次整個雲團移到 Ethanol 原本的位置",
        "• 這不是模型做錯，是感測器老化造成的位移；論文的模型也救不回它",
        "• 所以和論文的差距在其他五種氣體，尤其 Toluene（論文 0.92，我們 0.39）與 Acetone（0.94 對 0.51）",
    ], size=13)
    textbox(s, X, Y + Inches(3.3), Inches(4.5), Inches(1.5), [
        "訓練集 Batch 1 上的準確率是 1.000，所以問題不在學不會，在學到的東西搬不到後面的批次。",
    ], size=13)

    # 14. text vs figure ----------------------------------------------------------------
    s, _ = deck.slide("問題一：論文文字和圖對不上", keep_body=False)
    kicker(s, "論文有幾處文字與圖互相矛盾，我們每一處都做了兩種版本，資料支持圖。")
    table(s, ["哪裡", "文字說", "圖或公式說", "我們實測"], [
        ["網路寬度", "通道數從 1 逐步增加到 128", "Fig. 2 印的是 32/64/128/256/128", "圖版高 0.03 到 0.05"],
        ["Pooling", "「這個網路不用 pooling layer」", "—", "加了就掉 0.078，是全研究最大的單一效應"],
        ["四條公式的細節", "平均怎麼取、要不要除以類別數、統計量的維度、高斯能不能抽到負值", "各有兩種讀法", "都做了，差異都約零"],
    ], X, Y, W, col_fracs=(0.14, 0.32, 0.30, 0.24), size=12, row_h=Inches(0.7),
       aligns=["left", "left", "left", "left"])
    textbox(s, X, Y + Inches(3.1), W, Inches(1.5), [
        "結論：能照做的都照做了，沒有一個歧義能解釋 0.08 的差距。",
        "唯一真正重要的是「不用 pooling」那句話：它比論文的三個元件加起來都重要。",
    ], size=14)

    # 15. labels ----------------------------------------------------------------------
    s, _ = deck.slide("問題二：標籤對不上論文的 Table 2", keep_body=False)
    kicker(s, "資料集的氣體編號和論文表格對不上，我們用 Batch 1 的形狀對回去。不影響任何準確率。")
    picture(s, PAPER / "fig4a.png", X, Y, Inches(4.4), Inches(2.1), align="left")
    picture(s, FIG / "batch1_pca.png", X + Inches(4.6), Y, Inches(4.4), Inches(2.1), align="left")
    caption(s, X, Y + Inches(2.1), Inches(4.4), "論文的 PCA 圖（Batch 1 面板），依氣體名稱上色")
    caption(s, X + Inches(4.6), Y + Inches(2.1), Inches(4.4), "我們的 Batch 1 PCA，只標整數編號")
    textbox(s, X, Y + Inches(2.55), W, Inches(2.5), [
        "• 資料集文件說 2 是 Ethylene、3 是 Ammonia；論文 Table 2 的筆數照欄位讀卻只有一欄對得上",
        "• 把 Batch 1 的 PCA 形狀拿去對論文的圖，判定 2 與 3 互換；之後所有氣體名字都用這個對應",
        "• 模型只看整數編號，所以準確率完全不受影響；影響的只是「哪一種氣體怎麼樣」的敘述",
    ], size=14)

    # 16. Fig. S1 -----------------------------------------------------------------------
    s, _ = deck.slide("問題三：論文的訓練曲線對不上", keep_body=False)
    kicker(s, "論文 Fig. S1 的曲線既不像訓練曲線也不像測試曲線；而且在這個規則下沒有辦法選停止的時間點。")
    picture(s, PAPER / "figs1.png", X, Y, Inches(4.4), Inches(1.8), align="left")
    picture(s, FIG / "figS1_overlay.png", X + Inches(4.6), Y, Inches(4.4), Inches(1.8), align="left")
    caption(s, X, Y + Inches(1.8), Inches(4.4), "論文：準確率在 0.58／0.65／0.70 進入平台")
    caption(s, X + Inches(4.6), Y + Inches(1.8), Inches(4.4), "我們：訓練集到 1.000，測試批次 0.35–0.55")
    textbox(s, X, Y + Inches(2.25), W, Inches(3.0), [
        "• 如果 S1 畫的是訓練集，0.58 太低；如果畫的是測試批次，那代表訓練時一直在看測試資料",
        "• 只看 Batch 1 選不出該在第幾個 epoch 停：交叉驗證的最佳點落在 epoch 60–100，測試批次的最佳點在 7、40、70，兩者反向",
        "• 所以我們固定訓練 100 個 epoch，這是唯一不偷看測試資料的做法",
    ], size=14)

    # 17. augmentation ------------------------------------------------------------------
    s, _ = deck.slide("元件一：資料擴充，照論文做反而變差", keep_body=False)
    kicker(s, "論文的擴充是加隨機噪聲，加了反而變差；噪聲要有方向才會變好，但方向得看測試資料才知道。")
    table(s, ["做法", "target mean", "與基線差", "可分辨"], [
        ["不擴充（基線）", "0.5556", "—", "—"],
        ["論文的擴充：各方向隨機的噪聲", "0.5340", "−0.0216", "否"],
        ["沿感測器老化方向推一段距離，不加噪聲", "0.5770", "+0.0214", "是"],
    ], X, Y, W, col_fracs=(0.5, 0.18, 0.16, 0.16), size=13, row_h=Inches(0.42),
       bold_rows=(3,), aligns=["left", "right", "right", "center"])
    picture(s, FIG / "augmentation_pca.png", X, Y + Inches(1.8), W, Inches(1.5))
    caption(s, X, Y + Inches(3.3), W, "Batch 1 擴充前後：隨機噪聲只把雲撐大，離目標的距離不變；有方向的位移才靠過去")
    textbox(s, X, Y + Inches(3.7), W, Inches(1.6), [
        "• 「有方向」的版本是本研究唯一可分辨的增益，但它的方向是看過測試資料才選的，只能當上界",
        "• 增益全來自 Acetone 變好 0.26，代價是 Ethylene 變差 0.27；整批判錯的 Acetaldehyde 一樣沒救回來",
    ], size=14)

    # 18. Fig. 5 ---------------------------------------------------------------------------
    s, _ = deck.slide("元件二：特徵生成，Fig. 5 的畫法有問題", keep_body=False)
    kicker(s, "這張圖用了測試批次的資料，而且是循環論證：從 batch 2 生成的特徵當然靠近 batch 2。")
    picture(s, PAPER / "fig5-feature-generation.png", X, Y, Inches(4.4), Inches(2.55), align="left")
    picture(s, FIG / "feature_generation.png", X + Inches(4.6), Y, Inches(2.55 * 1.81), Inches(2.55), align="left")
    caption(s, X, Y + Inches(2.55), Inches(4.4), "論文 Fig. 5：把生成區塊套在 batch 2 上（橘），和 batch 2 自己（藍）比")
    caption(s, X + Inches(4.6), Y + Inches(2.55), Inches(4.4), "我們照同樣畫法畫，形狀對得上")
    textbox(s, X, Y + Inches(3.0), W, Inches(2.3), [
        "• 圖的 caption 寫「in one of the test domains (batch2)」：橘點是把區塊套在測試批次上才有的。訓練時區塊只看得到 Batch 1，預測時區塊根本不作用",
        "• 從 batch 2 生成的特徵當然落在 batch 2 旁邊，不管區塊做什麼。這張圖證明不了區塊能「涵蓋沒見過的批次」",
        "• 正確的問法：區塊只作用在 Batch 1，生成出來的特徵有沒有走到後面的批次？下一頁",
    ], size=14)

    # 19. coverage ------------------------------------------------------------------------
    s, _ = deck.slide("生成的特徵沒有走到任何後續批次", keep_body=False)
    kicker(s, "橘色（生成的）疊在灰色（Batch 1）上，藍色（後面的批次）在另一邊，中間一段空隙。")
    picture(s, FIG / "generation_drift_aligned_2d_b2-7-10.png", X, Y, W, Inches(3.05))
    textbox(s, X, Y + Inches(3.1), W, Inches(2.0), [
        "• 每一格是一種氣體乘一個批次。橫軸是這種氣體從 Batch 1 搬到那個批次的方向，所以「有沒有走到」一眼就看得出來",
        "• 灰 = Batch 1 原始特徵；橘 = 區塊從 Batch 1 生成的特徵，也就是訓練時網路看到的東西；藍 = 那個批次的真實特徵",
        "• 量成數字：橘和藍的重疊中位數 0.00；橘的寬度是灰的 1.01 倍；區塊沿這個方向只推了距離的 3%",
        "• 結果：加了特徵生成，準確率 +0.0004，等於零",
    ], size=13)

    # 20. why ---------------------------------------------------------------------------------
    s, body = deck.slide("為什麼特徵生成沒有用", keep_body=True)
    kicker(s, "區塊推的方向是對的，但往前推和往後推的機率一樣，而且推的距離太短。")
    bullets(body, [
        "方向對：區塊的擾動確實偏向感測器老化的方向，偏好程度是隨機方向的 3 到 4.6 倍",
        "正負號沒有：論文公式用對稱的高斯取樣，往老化方向推和往反方向推一樣多，平均起來是零",
        ("教分類器「對一個對稱的小鄰域不變」，只會把決策區域均勻撐大，不會把邊界搬到資料真正搬去的地方", 1),
        "距離太短：投影到老化方向上，區塊推的距離只有真實位移的 3%",
        "網路確實學到了區塊要求的東西（L_MSE 下降 2–5 倍），但真實的位移有 94% 原封不動穿過網路",
        "和元件一是同一件事：沒有方向的位移是負的或零，給了方向才是正的，而論文的公式把方向丟掉了",
    ], size=16, sub_size=14)

    # 21. v10 ----------------------------------------------------------------------------------
    s, _ = deck.slide("補上方向和距離，特徵生成還是沒用", keep_body=False)
    kicker(s, "把公式缺的方向和距離補上，方向用 Batch 1 自己就能量到的。準確率 +0.0023，不可分辨。")
    table(s, ["版本", "做了什麼", "target mean", "與原版差"], [
        ["論文原版", "對稱高斯", "0.5773", "—"],
        ["只補正負號", "往老化方向的一側取樣，距離照論文", "0.5767", "−0.0006"],
        ["補正負號加距離", "往老化方向推兩個「Batch 1 內部的位移」", "0.5796", "+0.0023"],
    ], X, Y, W, col_fracs=(0.22, 0.46, 0.16, 0.16), size=13, row_h=Inches(0.42),
       bold_rows=(3,), aligns=["left", "left", "right", "right"])
    textbox(s, X, Y + Inches(1.9), W, Inches(3.4), [
        "• 方向從哪裡來：Batch 1 裡有三種氣體量了兩次，中間隔了一段時間；兩次的差就是老化的方向，不需要看測試資料",
        "• 事後檢查：這個方向和真實老化方向的餘弦是 +0.59（六種氣體全為正），推的距離是真實位移的 0.7 到 1.4 倍。方向對、距離也對",
        "• 即使這樣，每一批、每一種氣體的差都在 0.012 以內。同樣的位移在輸入端（元件一）能把 Acetone 和 Ethylene 各搬 0.25",
        "• 結論：特徵生成這個區塊作用的空間，不是分類器決策邊界會回應的那個空間。這個元件關掉了",
    ], size=14)

    # 22. L_con --------------------------------------------------------------------------------
    s, body = deck.slide("元件三：對比損失，已實作、決定不跑", keep_body=True)
    kicker(s, "它要把原始特徵和生成特徵拉近；但生成特徵本來就在原始特徵旁邊，拉近它換不到對老化的抵抗力。")
    bullets(body, [
        "論文的對比損失（S5）是標準的 Supervised Contrastive Loss：同一種氣體的原始特徵和生成特徵在單位球上互相拉近",
        "程式寫好了：損失函數、三個版本的設定、事前預測與否證條件、單元測試都通過。補跑只要 15 分鐘",
        "不跑的理由：它要求「原始特徵對生成特徵不變」，而第 19 頁已經量到生成特徵沒有離開 Batch 1（重疊 0.00、寬度比 1.01）。對一個沒走開的東西做不變性，不會變成對真正搬走的資料的不變性",
        ("前一個分支在另一個網路上量了五次：−0.005 到 +0.003", 1),
        "附帶一個觀察：對比損失的起始值約 4.8，是交叉熵的兩倍多；論文 Fig. S1 裡 CDCNN 的 loss 停在 1.5、其他兩個停在 0.5，和這個一致",
    ], size=16, sub_size=14)

    # 23. v12 ------------------------------------------------------------------------------------
    s, _ = deck.slide("延伸：把老化方向從輸入裡扣掉", keep_body=False)
    kicker(s, "老化主要沿一條方向，Batch 1 自己量得到，就不讓模型看那條方向。結果 +0.0128，不可分辨。")
    textbox(s, X, Y, Inches(4.4), Inches(0.35), "用兩個數字的例子說明。假設只有 x 會隨感測器老化變大：", size=13)
    table(s, ["", "Batch 1", "幾個月後", "只留 y"], [
        ["Ethanol", "(5, 2)", "(9, 2)", "2 → 2"],
        ["Acetaldehyde", "(1, 3)", "(5, 3)", "3 → 3"],
    ], X, Y + Inches(0.4), Inches(4.4), col_fracs=(0.34, 0.22, 0.22, 0.22), size=12, row_h=Inches(0.34),
       aligns=["left", "center", "center", "center"])
    textbox(s, X, Y + Inches(1.5), Inches(4.4), Inches(2.0), [
        "幾個月後的 Acetaldehyde 落在 (5, 3)，離 Batch 1 的 Ethanol (5, 2) 最近，被判成 Ethanol。這就是整批判錯的來源。",
        "把 x 扔掉只看 y，兩種氣體不管過多久都在原地。真實資料有 128 個數字，老化方向是一條斜線，做法一樣：把每筆資料在那條線上的分量減掉。",
    ], size=12)
    table(s, ["扣掉什麼", "target mean", "與基線差", "可分辨"], [
        ["不扣（基線）", "0.5556", "—", "—"],
        ["Batch 1 三種氣體兩次量測的共同方向（頭條）", "0.5684", "+0.0128", "否"],
        ["三個方向張成的空間", "0.5610", "+0.0055", "否"],
        ["只用 Ethanol 的方向（看過測試資料才知道選它）", "0.5747", "+0.0191", "是，上界"],
    ], X + Inches(4.6), Y, Inches(4.4), col_fracs=(0.5, 0.18, 0.16, 0.16), size=11, row_h=Inches(0.42),
       bold_rows=(2,), aligns=["left", "right", "right", "center"])
    textbox(s, X + Inches(4.6), Y + Inches(2.25), Inches(4.4), Inches(2.9), [
        "• 為什麼不夠：Batch 1 兩次量測之間的方向只裝了三年老化的 29%，裝的主要是 Batch 1 自己的濃度差異。它是方向的估計，不是老化本身",
        "• 要扣掉夠多老化就得扣三個方向，但那三個方向也裝了 76% 的氣體差異。老化和氣體資訊擠在同一個小空間裡，扣不乾淨",
        "• Acetaldehyde 在每個版本下仍落在 Ethanol 上。這個延伸也關掉了",
    ], size=12)

    # 24. results table --------------------------------------------------------------------------
    s, _ = deck.slide("實驗結果總表", keep_body=False)
    kicker(s, "論文三個元件在只用 Batch 1 的規則下沒有一個重現得出增益；我們自己的延伸也一樣。")
    table(s, ["元件", "我們量到的", "前一個分支", "論文宣稱"], [
        ["資料擴充，照論文", "−0.0216", "−0.013 到 −0.009", "+0.036（含特徵生成）"],
        ["資料擴充，有方向（看過測試資料）", "+0.0214", "—", "—"],
        ["特徵生成 + L_MSE，照論文", "+0.0004", "+0.001 到 +0.014", "（含在上列）"],
        ["特徵生成，補上方向與距離", "+0.0023", "—", "—"],
        ["對比損失 L_con", "已實作，未跑", "−0.005 到 +0.003", "+0.053"],
        ["延伸：扣掉老化方向", "+0.0128，不可分辨", "—", "—"],
    ], X, Y, W, col_fracs=(0.37, 0.21, 0.22, 0.2), size=13, row_h=Inches(0.5),
       bold_rows=(3, 4), aligns=["left", "right", "right", "right"])
    textbox(s, X, Y + Inches(3.7), W, Inches(1.5), [
        "基線 0.5556 是只用 Batch 1 的定案數字；所有 0.57 系列都是看過測試資料才選得出來的上界。",
        "可分辨的門檻是五個種子的兩倍標準誤，約 0.01 到 0.02；低於 0.005 的差不算效應。",
    ], size=14)

    # 25. literature ------------------------------------------------------------------------------
    s, _ = deck.slide("文獻對照：這份資料集別人怎麼做", keep_body=False)
    kicker(s, "同一個設定下，不看測試資料的方法沒有一個超過 60%；超過的全都用了測試批次的無標籤資料。")
    table(s, ["用到多少測試批次的資訊", "代表方法", "平均準確率 (%)"], [
        ["完全不看", "原論文的 SVM 38.9；正交訊號校正 56.5；我們的 ResNet 55.6", "39–57"],
        ["完全不看（宣稱）", "CDCNN 論文的 ResNet / CDWC / CDCNN", "63.5 / 67.1 / 72.3"],
        ["看測試批次的無標籤資料做對齊", "DRCA 62.2、D-DRCA 73.8、流形方法 64–67", "62–74"],
        ["兩個訓練批次 + 無標籤資料", "AMDS-PFFA（2024）", "83.2"],
        ["每個測試批次給 20 到 50 筆有標籤的校正樣本", "DAELM（2015）", "81–92"],
    ], X, Y, W, col_fracs=(0.32, 0.5, 0.18), size=12, row_h=Inches(0.5),
       bold_rows=(1,), aligns=["left", "left", "right"])
    textbox(s, X, Y + Inches(3.3), W, Inches(2.0), [
        "• 我們的 55.6 比原論文的 SVM 高 17 個百分點；CDCNN 論文的 63.5 是「不看測試資料」這一列唯一的例外，而它的 Fig. 5 已被抓到用了測試資料",
        "• 跨過 60% 的做法都是看測試批次的無標籤樣本做對齊。那和我們的延伸幾乎一樣，差別只在方向是從測試批次量的，不是從 Batch 1 猜的",
        "• 來源：Zhang 2015（DAELM）、Zhang 2017（DRCA）、Yi 2019（D-DRCA）、AMDS-PFFA 2024",
    ], size=13)

    # 26. conclusion ------------------------------------------------------------------------------
    s, body = deck.slide("結論", keep_body=True)
    bullets(body, [
        "照論文重建的網路在只用 Batch 1 的規則下得到 0.5556；論文的三個元件一個一個加上去，資料擴充變差、特徵生成等於零、對比損失的前提已被量掉。論文的增益重現不出來",
        "和論文的差距不在這三個元件，在 Acetaldehyde 以外的五種氣體；而 Acetaldehyde 整批判錯這件事，論文的模型也一樣",
        "論文的 Fig. 5 用了測試批次的資料、標籤對不上 Table 2、Fig. S1 的曲線來源不明。三件事只記錄，不推測動機",
        "不看測試資料的模型是現實中比較需要的，但這份資料集上沒有人在這個規則下超過 60%。要再往前，文獻的主流是讓每個測試批次的無標籤資料參與對齊，報告時要明說那是另一個設定",
    ], size=17, sub_size=15)

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
