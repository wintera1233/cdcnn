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
        else:
            # Start the body below the message line the slides add under the title.
            # A placeholder inherits its box from the layout; once one edge is
            # set the whole box must be written, or the shape gets zero size.
            x0, y0, w0, h0 = BODY
            body.left, body.width = x0, w0
            body.top, body.height = TOP, h0 - (TOP - y0)
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
        if level == 0 and text.endswith("：") and len(text) <= 14:
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
        "架構：五個 ResNet 區塊、不做 pooling、攤平後接全連接層，完全照 Fig. 2",
        "資料：UCI 漂移資料集，128 維、六種氣體、十批、三年",
        "規則：訓練與所有選擇只用 Batch 1；Batch 2–10 等模型凍結後才打開",
        "指標：九批準確率的平均（target mean），五個隨機種子",
    ], size=14)
    textbox(s, X, Y + Inches(3.15), Inches(4.6), Inches(1.6), [
        "三個元件一個一個加上去量：",
        "① 資料擴充　② 特徵生成　③ 對比損失",
    ], size=14)

    # 13. the baseline ---------------------------------------------------------------
    s, _ = deck.slide("基線結果：0.5556，論文的 ResNet 0.6346", keep_body=False)
    kicker(s, "差 0.08。先看錯在哪：有一種氣體整批判錯，而論文自己的模型也一樣。")
    picture(s, FIG / "baseline-confusion_best.png", X, Y, Inches(4.5), Inches(2.9), align="left")
    caption(s, X, Y + Inches(2.9), Inches(4.5), "我們的混淆矩陣（Batch 2–10 合併）：Acetaldehyde 那一列全部判成 Ethanol")
    picture(s, PAPER / "figS3_confusion.png", X + Inches(4.7), Y, Inches(4.3), Inches(2.1), align="left")
    caption(s, X + Inches(4.7), Y + Inches(2.1), Inches(4.3), "論文 Fig. S3：CDCNN 也把 100% 的 Acetaldehyde 判成 Ethanol")
    textbox(s, X + Inches(4.7), Y + Inches(2.5), Inches(4.3), Inches(2.5), [
        "• Acetaldehyde 在 Batch 1 分得很好，到後面的批次整個雲團移到 Ethanol 原本的位置",
        "• 論文的模型也一樣；所以和論文的差距在其他五種氣體",
    ], size=14)
    textbox(s, X, Y + Inches(3.3), Inches(4.5), Inches(1.5), [
        "Batch 1 上的準確率是 1.000：問題不在學不會，在搬不過去。",
    ], size=14)

    # 16. Fig. S1 -----------------------------------------------------------------------
    s, _ = deck.slide("問題：論文的訓練曲線對不上", keep_body=False)
    kicker(s, "論文 Fig. S1 的曲線既不像訓練曲線也不像測試曲線；而且在這個規則下沒有辦法選停止的時間點。")
    picture(s, PAPER / "figs1.png", X, Y, Inches(4.4), Inches(1.8), align="left")
    picture(s, FIG / "figS1_overlay.png", X + Inches(4.6), Y, Inches(4.4), Inches(1.8), align="left")
    caption(s, X, Y + Inches(1.8), Inches(4.4), "論文 Fig. S1：(a) 三個模型的準確率在 0.58／0.65／0.70 進入平台，(b) CDCNN 的 loss 停在 1.5")
    caption(s, X + Inches(4.6), Y + Inches(1.8), Inches(4.4), "我們：左為 Batch 1 訓練準確率，右為測試批次的平均；虛線是論文的三個平台")
    textbox(s, X, Y + Inches(2.3), W, Inches(3.1), [
        "• 論文沒說縱軸是訓練集還是測試批次；兩種讀法都對不上我們的曲線",
        "• 只看 Batch 1 選不出停止點：交叉驗證的最佳點和測試批次的最佳點走反方向",
        "• 所以固定訓練 100 個 epoch",
    ], size=14)

    # 17. augmentation ------------------------------------------------------------------
    s, _ = deck.slide("元件一：資料擴充，照論文做反而變差", keep_body=False)
    kicker(s, "論文的擴充是加隨機噪聲，加了反而變差；噪聲要有方向才會變好，但方向得看測試資料才知道。")
    table(s, ["做法", "target mean", "與基線差"], [
        ["不擴充（基線）", "0.5556", "—"],
        ["論文的擴充：各方向隨機的噪聲（Eq. 5–7）", "0.5340", "−0.0216"],
        ["沿感測器老化方向推一段距離，不加噪聲", "0.5770", "+0.0214"],
    ], X, Y, Inches(5.3), col_fracs=(0.6, 0.2, 0.2), size=12, row_h=Inches(0.42),
       bold_rows=(3,), aligns=["left", "right", "right"])
    textbox(s, X + Inches(5.5), Y, Inches(3.5), Inches(1.8), [
        "• 「有方向」的版本要看測試資料才知道方向，只能當上界",
        "• 增益來自 Acetone，代價是 Ethylene；Acetaldehyde 沒救回來",
    ], size=13)
    picture(s, FIG / "augmentation_pca_3panel.png", X, Y + Inches(1.85), W, Inches(3.4))
    caption(s, X, Y + Inches(5.15), W, "菱形：各格 Acetaldehyde 雲的重心；紅叉：測試批次的 Acetaldehyde 重心；d：兩者在 128 維空間的距離（Batch 1 的前兩個主成分）")

    # 18. Fig. 5 ---------------------------------------------------------------------------
    s, _ = deck.slide("元件二：特徵生成，Fig. 5 怎麼讀", keep_body=False)
    kicker(s, "我們畫得出和論文一樣的圖；但它是在測試批次上畫的，看不出區塊對沒見過的批次有沒有幫助。")
    picture(s, PAPER / "fig5-feature-generation.png", X, Y, Inches(4.45), Inches(3.3), align="left")
    picture(s, FIG / "feature_generation.png", X + Inches(4.55), Y, Inches(4.45), Inches(3.3), align="left")
    caption(s, X, Y + Inches(3.05), Inches(4.45), "論文 Fig. 5：生成區塊套在 batch 2 上（橘），與 batch 2 自己的特徵（藍）比")
    caption(s, X + Inches(4.55), Y + Inches(3.05), Inches(4.45), "我們照同樣的畫法畫，形狀對得上：橘色比藍色寬、往外延伸")
    textbox(s, X, Y + Inches(3.45), W, Inches(1.9), [
        "• 這張圖是把生成區塊套在測試批次上，再和它自己比",
        "• 我們改從 Batch 1 生成，看它有沒有走到後面的批次（下一頁）",
    ], size=14)

    # 19. coverage ------------------------------------------------------------------------
    s, _ = deck.slide("生成的特徵沒有走到任何後續批次", keep_body=False)
    kicker(s, "橘色（生成的）疊在灰色（Batch 1）上，藍色（後面的批次）在另一邊，中間一段空隙。")
    picture(s, FIG / "generation_drift_aligned_2d_b2-7-10.png", X, Y, W, Inches(3.05))
    textbox(s, X, Y + Inches(3.1), W, Inches(2.0), [
        "• 橫軸是這種氣體從 Batch 1 搬到那個批次的方向；灰 = Batch 1，橘 = 從 Batch 1 生成，藍 = 那個批次的真實特徵",
        "• 橘色只疊在灰色上；加了特徵生成，準確率 +0.0004",
    ], size=14)

    # 22. L_con --------------------------------------------------------------------------------
    s, body = deck.slide("元件三：對比損失，照論文做反而變差", keep_body=True)
    kicker(s, "它把原始特徵和生成特徵拉近；但生成特徵本來就在 Batch 1 裡，拉近它換來的是對老化更敏感。")
    bullets(body, [
        "論文的對比損失（S5）是標準的 Supervised Contrastive Loss，同一種氣體的原始特徵和生成特徵在單位球上互相拉近：",
        ("L_con = − Σᵢ (1/|P(i)|) Σ_{p∈P(i)} log [ exp(f(zᵢ)·f(zₚ)/τ) / Σ_{a∈A(i)} exp(f(zᵢ)·f(zₐ)/τ) ]", 1),
        "結果：論文版（τ=0.07）−0.0138，可分辨地更差；τ=0.5 時損失幾乎不作用，−0.0014",
        "機制：它把 Batch 1 同類的點拉緊，類內半徑縮小，但真實的老化一點都沒縮；特徵上的相對漂移反而放大一半",
        ("逐類別：Ethylene +0.19、Toluene −0.25、Acetone −0.10，把擴充換來的交易倒回去一半", 1),
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
        "幾個月後的 Acetaldehyde 落在 (5, 3)，離 Batch 1 的 Ethanol (5, 2) 最近，被判成 Ethanol。",
        "把 x 扔掉只看 y，兩種氣體不管過多久都在原地。真實資料是 128 維，做法一樣。",
    ], size=13)
    table(s, ["扣掉什麼", "target mean", "與基線差", "可分辨"], [
        ["不扣（基線）", "0.5556", "—", "—"],
        ["Batch 1 三種氣體兩次量測的共同方向（頭條）", "0.5684", "+0.0128", "否"],
        ["三個方向張成的空間", "0.5610", "+0.0055", "否"],
        ["只用 Ethanol 的方向（看過測試資料才知道選它）", "0.5747", "+0.0191", "是，上界"],
    ], X + Inches(4.6), Y, Inches(4.4), col_fracs=(0.5, 0.18, 0.16, 0.16), size=11, row_h=Inches(0.42),
       bold_rows=(2,), aligns=["left", "right", "right", "center"])
    textbox(s, X + Inches(4.6), Y + Inches(2.25), Inches(4.4), Inches(2.9), [
        "• Batch 1 自己量到的方向只裝了三年老化的 29%，裝的主要是濃度差異",
        "• 扣掉夠多老化就會連氣體差異一起扣掉",
        "• Acetaldehyde 仍落在 Ethanol 上",
    ], size=13)

    # 24. results table --------------------------------------------------------------------------
    s, _ = deck.slide("實驗結果總表", keep_body=False)
    kicker(s, "論文三個元件在只用 Batch 1 的規則下沒有一個重現得出增益；我們自己的延伸也一樣。")
    table(s, ["元件", "我們量到的", "前一個分支", "論文宣稱"], [
        ["資料擴充，照論文", "−0.0216", "−0.013 到 −0.009", "+0.036（含特徵生成）"],
        ["資料擴充，有方向（看過測試資料）", "+0.0214", "—", "—"],
        ["特徵生成 + L_MSE，照論文", "+0.0004", "+0.001 到 +0.014", "（含在上列）"],
        ["對比損失 L_con，照論文（τ=0.07）", "−0.0138", "−0.005 到 +0.003", "+0.053"],
        ["延伸：扣掉老化方向", "+0.0128，不可分辨", "—", "—"],
    ], X, Y, W, col_fracs=(0.37, 0.21, 0.22, 0.2), size=13, row_h=Inches(0.5),
       bold_rows=(3,), aligns=["left", "right", "right", "right"])
    textbox(s, X, Y + Inches(3.2), W, Inches(1.5), [
        "基線 0.5556 是只用 Batch 1 的定案數字；0.57 系列都是看過測試資料才選得出來的上界。",
    ], size=14)

    # 25. literature -----------------------------------------------------------------------------
    s, _ = deck.slide("文獻對照：這份資料集別人怎麼做", keep_body=False)
    kicker(s, "同一設定（Batch 1 訓練、Batch 2–10 測試）下各方法的架構，以及用了多少、用什麼方式的測試批次資料。")
    table(s, ["文獻", "主要架構", "source 怎麼用", "target 怎麼用", "平均 (%)"], [
        ["Vergara 2012（資料集原論文）", "SVM，RBF 核", "Batch 1 全部有標籤", "不用", "38.9"],
        ["Vergara 2012", "SVM，測地流核（GFK）", "Batch 1 有標籤", "每個測試批次的無標籤樣本，用來建核", "64.0"],
        ["Vergara 2012", "流形正則化（ML-comgfk）", "Batch 1 有標籤", "無標籤樣本，半監督", "67.3"],
        ["Yi 2019 表內的 OSC", "正交訊號校正 + 分類器", "Batch 1 有標籤", "不用", "56.5"],
        ["Zhang 2015（DAELM）", "極限學習機 + 域適應", "Batch 1 有標籤", "每個測試批次 20–50 筆有標籤校正樣本", "80.8–91.9"],
        ["Zhang 2017（DRCA）", "線性子空間投影 + 分類器", "Batch 1 有標籤", "無標籤樣本，對齊兩域平均", "62.2"],
        ["Yi 2019（D-DRCA）", "同上，加入 source 標籤的判別項", "Batch 1 有標籤", "無標籤樣本", "73.8"],
        ["AMDS-PFFA 2024", "CNN + 注意力，多源域適應", "Batch 1 與 2 有標籤", "無標籤樣本，對齊特徵分布", "83.2（加權）"],
        ["KD 2025", "全連接網路 + 知識蒸餾", "Batch 1 有標籤", "每批一半無標籤，做偽標籤與調參", "47.9"],
        ["CDCNN 2024（本篇）", "一維 ResNet + 擴充 + 特徵生成 + 對比損失", "Batch 1 有標籤", "宣稱不用；Fig. 5 用了 batch 2", "63.5 / 72.3"],
        ["我們", "一維 ResNet，照 Fig. 2", "Batch 1 有標籤", "不用", "55.6"],
    ], X, Y, W, col_fracs=(0.2, 0.26, 0.15, 0.28, 0.11), size=9.5, row_h=Inches(0.38),
       bold_rows=(11,), aligns=["left", "left", "left", "left", "right"])
    textbox(s, X, Y + Inches(4.65), W, Inches(0.7), [
        "不用測試批次資料的只有三列，都在 60% 以下。",
    ], size=13)

    # 26. conclusion ------------------------------------------------------------------------------
    s, body = deck.slide("結論", keep_body=True)
    bullets(body, [
        "只用 Batch 1 的規則下，照論文重建的網路得到 0.5556；三個元件一個一個加上去，資料擴充變差、特徵生成等於零、對比損失也變差",
        "和論文的差距不在這三個元件，在 Acetaldehyde 以外的五種氣體",
        "不看測試資料的模型是現實中比較需要的，但這份資料集上沒有人在這個規則下超過 60%；要再往前，文獻的主流是讓測試批次的無標籤資料參與對齊",
    ], size=18, sub_size=16)

    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
