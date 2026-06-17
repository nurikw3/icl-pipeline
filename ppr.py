from manim import *
import numpy as np

# ─── Pixel palette ───────────────────────────────────────────────────────────
BG      = "#000000"
WHITE   = "#FFFFFF"
LGRAY   = "#AAAAAA"
DGRAY   = "#444444"
MID     = "#666666"

config.background_color = BG
config.pixel_width  = 960
config.pixel_height = 540
config.frame_rate   = 12   # low fps ⇒ chunkier pixel feel

# ─── Helper: pixel-block rectangle ───────────────────────────────────────────
def pxbox(w, h, color=WHITE, fill=False):
    rect = Rectangle(width=w, height=h,
                     color=color,
                     fill_color=color if fill else BG,
                     fill_opacity=1.0 if fill else 0.0,
                     stroke_width=3)
    return rect

# ─── Helper: chunky label ────────────────────────────────────────────────────
def px_label(text, scale=0.35, color=WHITE):
    return Text(text, font="Courier New",
                color=color, weight=BOLD).scale(scale)

# ─── Helper: node (square with label) ────────────────────────────────────────
def make_node(label, w=1.15, h=0.42, node_color=WHITE, label_color=BG,
              fill=True, label_scale=0.28):
    box = pxbox(w, h, color=node_color, fill=fill)
    lbl = Text(label, font="Courier New",
               color=label_color if fill else node_color,
               weight=BOLD).scale(label_scale)
    lbl.move_to(box.get_center())
    return VGroup(box, lbl)

# ─── Helper: pixel arrow (dashed/solid) ──────────────────────────────────────
def px_arrow(start, end, color=WHITE, stroke=2, dashed=False):
    if dashed:
        arr = DashedLine(start, end, color=color,
                         stroke_width=stroke, dash_length=0.08)
    else:
        arr = Arrow(start, end, color=color,
                    stroke_width=stroke,
                    buff=0.05,
                    tip_length=0.12,
                    max_stroke_width_to_length_ratio=999)
    return arr

# ─── Helper: glitch flicker (fast opacity toggle) ────────────────────────────
def glitch(mob, scene):
    scene.play(mob.animate.set_opacity(0.3), run_time=0.05)
    scene.play(mob.animate.set_opacity(1.0), run_time=0.05)

# ═══════════════════════════════════════════════════════════════════════════════
class PPRPixel(Scene):
    def construct(self):
        self._frame1()
        self._frame2()
        self._frame3()
        self._frame4()
        self._frame5()
        self._frame6()
        self._frame7()
        self._frame8()

    # ── scanline overlay (static decoration) ─────────────────────────────────
    def _scanlines(self):
        lines = VGroup()
        for y in np.arange(-3.5, 3.5, 0.18):
            l = Line(LEFT * 7, RIGHT * 7,
                     stroke_width=0.6,
                     color=WHITE).shift(UP * y)
            l.set_opacity(0.04)
            lines.add(l)
        return lines

    # ── title banner ─────────────────────────────────────────────────────────
    def _banner(self, txt, sub=""):
        title = Text(txt, font="Courier New",
                     color=WHITE, weight=BOLD).scale(0.52)
        title.to_edge(UP, buff=0.18)
        group = VGroup(title)
        if sub:
            s = Text(sub, font="Courier New",
                     color=LGRAY, weight=BOLD).scale(0.28)
            s.next_to(title, DOWN, buff=0.08)
            group.add(s)
        box = pxbox(title.get_width() + 0.4,
                    group.get_height() + 0.2, WHITE, fill=False)
        box.move_to(group.get_center())
        return VGroup(box, group)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 1 – Query node
    # ══════════════════════════════════════════════════════════════════════════
    def _frame1(self):
        scan = self._scanlines()
        self.add(scan)

        banner = self._banner("PERSONALIZED PAGERANK",
                              "In-Context Learning · Example Retrieval")
        self.play(FadeIn(banner), run_time=0.5)

        query_node = make_node("şorpa göşt wa tuzdur",
                               w=3.0, h=0.55,
                               node_color=WHITE, label_color=BG,
                               fill=True, label_scale=0.30)
        query_node.move_to(ORIGIN)

        q_label = Text("QUERY", font="Courier New",
                       color=LGRAY, weight=BOLD).scale(0.26)
        q_label.next_to(query_node, UP, buff=0.12)

        # pixel blink-in
        for _ in range(3):
            tmp = query_node.copy().set_opacity(0.0)
            self.add(tmp)
            self.wait(0.06)
            self.remove(tmp)
            self.wait(0.06)

        self.play(Write(query_node), run_time=0.5)
        self.play(FadeIn(q_label), run_time=0.3)
        glitch(query_node, self)
        self.wait(1.2)
        self.play(FadeOut(banner), FadeOut(q_label), run_time=0.3)
        self._keep("q_node", query_node)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 2 – Feature nodes
    # ══════════════════════════════════════════════════════════════════════════
    def _frame2(self):
        q_node = self._get("q_node")
        self.play(q_node.animate.move_to(LEFT * 3.8), run_time=0.4)

        banner = self._banner("FRAME 2 · FEATURE EXPANSION")
        self.play(FadeIn(banner), run_time=0.3)

        token_feats = [
            "token:şorpa", "token:göşt",
            "token:wa",    "token:tuzdur",
        ]
        small_feats = [
            "suf3:dur",   "suf4:zdur",
            "pre3:tuz",   "bigram:wa·tuzdur",
            "type:phrase","length:3-4",
        ]

        # token nodes (large, filled)
        t_nodes = VGroup()
        ty = [0.95, 0.32, -0.32, -0.95]
        for i, lbl in enumerate(token_feats):
            n = make_node(lbl, w=1.55, h=0.42,
                          node_color=WHITE, label_color=BG,
                          fill=True, label_scale=0.24)
            n.move_to(RIGHT * 0.3 + UP * ty[i])
            t_nodes.add(n)

        # small feature nodes (outline only)
        s_nodes = VGroup()
        sy = [1.5, 0.9, 0.3, -0.3, -0.9, -1.5]
        for i, lbl in enumerate(small_feats):
            n = make_node(lbl, w=1.45, h=0.36,
                          node_color=LGRAY, label_color=LGRAY,
                          fill=False, label_scale=0.22)
            n.move_to(RIGHT * 1.9 + UP * sy[i])
            s_nodes.add(n)

        # arrows from query
        t_arrows = VGroup()
        for n in t_nodes:
            a = px_arrow(q_node.get_right(),
                         n.get_left(), stroke=2)
            t_arrows.add(a)

        s_arrows = VGroup()
        for n in s_nodes:
            a = px_arrow(q_node.get_right(),
                         n.get_left(), stroke=1, dashed=True)
            s_arrows.add(a)

        self.play(LaggedStart(*[GrowArrow(a) for a in t_arrows],
                              lag_ratio=0.15), run_time=0.7)
        self.play(LaggedStart(*[FadeIn(n, shift=RIGHT*0.1)
                                for n in t_nodes],
                              lag_ratio=0.15), run_time=0.8)
        self.play(LaggedStart(*[Create(a) for a in s_arrows],
                              lag_ratio=0.1), run_time=0.5)
        self.play(LaggedStart(*[FadeIn(n) for n in s_nodes],
                              lag_ratio=0.1), run_time=0.5)
        self.wait(0.8)

        self.play(FadeOut(banner), run_time=0.2)
        self._keep("feat_t", t_nodes)
        self._keep("feat_s", s_nodes)
        self._keep("t_arr",  t_arrows)
        self._keep("s_arr",  s_arrows)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 3 – Probability mass flows to train examples
    # ══════════════════════════════════════════════════════════════════════════
    def _frame3(self):
        banner = self._banner("FRAME 3 · MASS FLOWS TO TRAIN EXAMPLES")
        self.play(FadeIn(banner), run_time=0.3)

        train_labels = ["S1","S2","S3","S4"]
        train_y      = [1.1, 0.37, -0.37, -1.1]
        ex_nodes = VGroup()
        for i, lbl in enumerate(train_labels):
            n = make_node(lbl, w=0.65, h=0.38,
                          node_color=MID, label_color=LGRAY,
                          fill=False, label_scale=0.30)
            n.move_to(RIGHT * 3.7 + UP * train_y[i])
            ex_nodes.add(n)

        self.play(LaggedStart(*[FadeIn(n) for n in ex_nodes],
                              lag_ratio=0.15), run_time=0.5)

        # thick arrows from token features → S1, S2 (strong match)
        feat_t = self._get("feat_t")
        feat_s = self._get("feat_s")

        strong_arrows = VGroup()
        for src_i in [0, 3]:   # token:şorpa, token:tuzdur → S1
            a = px_arrow(feat_t[src_i].get_right(),
                         ex_nodes[0].get_left(), stroke=3)
            strong_arrows.add(a)
        for src_i in [0, 3]:   # → S2
            a = px_arrow(feat_t[src_i].get_right(),
                         ex_nodes[1].get_left(), stroke=3)
            strong_arrows.add(a)
        # thin arrows to S3, S4
        weak_arrows = VGroup()
        for src_i in [2]:       # token:wa → S3
            a = px_arrow(feat_t[src_i].get_right(),
                         ex_nodes[2].get_left(), stroke=1)
            weak_arrows.add(a)
        for src_i in [2]:       # token:wa → S4
            a = px_arrow(feat_t[src_i].get_right(),
                         ex_nodes[3].get_left(), stroke=1)
            weak_arrows.add(a)

        self.play(LaggedStart(*[GrowArrow(a) for a in strong_arrows],
                              lag_ratio=0.1), run_time=0.6)
        self.play(LaggedStart(*[GrowArrow(a) for a in weak_arrows],
                              lag_ratio=0.1), run_time=0.4)

        # "flow dots" along arrows
        for arrow in strong_arrows:
            dot = Dot(radius=0.06, color=WHITE)
            dot.move_to(arrow.get_start())
            self.play(MoveAlongPath(dot, arrow), run_time=0.25)
            self.remove(dot)

        prob_lbl = Text("P(s|q) ∝ edge weight", font="Courier New",
                        color=LGRAY, weight=BOLD).scale(0.26)
        prob_lbl.to_corner(DR, buff=0.2)
        self.play(FadeIn(prob_lbl), run_time=0.3)
        self.wait(0.5)

        self.play(FadeOut(banner), FadeOut(prob_lbl), run_time=0.2)
        self._keep("ex_nodes",  ex_nodes)
        self._keep("s_arrows2", strong_arrows)
        self._keep("w_arrows2", weak_arrows)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 4 – Highlight real train examples
    # ══════════════════════════════════════════════════════════════════════════
    def _frame4(self):
        banner = self._banner("FRAME 4 · TRAIN EXAMPLES")
        self.play(FadeIn(banner), run_time=0.3)

        ex_nodes = self._get("ex_nodes")

        texts = [
            "tuz aççıqdur → salt is bitter",
            "tuz şı̇̄rı̇̄n emes → salt is not sweet",
            "wa ǧarı̇̄bdin biri budur → stranger tales",
            "Samarqand wa Ḫocand bolǧay → and Khujand",
        ]
        colors = [WHITE, WHITE, LGRAY, MID]

        cards = VGroup()
        for i, (txt, col) in enumerate(zip(texts, colors)):
            card = pxbox(3.8, 0.42, color=col, fill=(i < 2))
            card.move_to(RIGHT * 3.7 + UP * [1.1, 0.37, -0.37, -1.1][i])
            lbl = Text(txt, font="Courier New",
                       color=BG if (i < 2) else col,
                       weight=BOLD).scale(0.195)
            lbl.move_to(card.get_center())
            cards.add(VGroup(card, lbl))

        # remove old S1..S4 labels
        self.play(FadeOut(ex_nodes), run_time=0.2)
        self.play(LaggedStart(*[FadeIn(c) for c in cards],
                              lag_ratio=0.2), run_time=0.8)

        # glow pulse on S1, S2
        for _ in range(2):
            self.play(cards[0].animate.set_opacity(0.5),
                      cards[1].animate.set_opacity(0.5), run_time=0.1)
            self.play(cards[0].animate.set_opacity(1.0),
                      cards[1].animate.set_opacity(1.0), run_time=0.1)

        note = Text("S1, S2: strong tuz/dur match!",
                    font="Courier New", color=WHITE,
                    weight=BOLD).scale(0.26)
        note.to_corner(DR, buff=0.2)
        self.play(FadeIn(note), run_time=0.3)
        self.wait(0.8)
        self.play(FadeOut(banner), FadeOut(note), run_time=0.2)
        self._keep("cards", cards)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 5 – Mass flows back + formula
    # ══════════════════════════════════════════════════════════════════════════
    def _frame5(self):
        banner = self._banner("FRAME 5 · MASS RETURNS TO FEATURES")
        self.play(FadeIn(banner), run_time=0.3)

        cards   = self._get("cards")
        feat_t  = self._get("feat_t")

        back_arrows = VGroup()
        for c_i in [0, 1]:
            for f_i in [0, 3]:
                a = px_arrow(cards[c_i][0].get_left(),
                             feat_t[f_i].get_right(),
                             stroke=2, dashed=True)
                back_arrows.add(a)

        self.play(LaggedStart(*[Create(a) for a in back_arrows],
                              lag_ratio=0.1), run_time=0.6)

        # flow dots back
        for arrow in back_arrows[:2]:
            dot = Dot(radius=0.06, color=LGRAY)
            dot.move_to(arrow.get_start())
            self.play(MoveAlongPath(dot, arrow), run_time=0.2)
            self.remove(dot)

        formula = Text("P(f|s) = w(s,f) / Σ_f w(s,f)",
                       font="Courier New", color=WHITE,
                       weight=BOLD).scale(0.30)
        formula.to_corner(DR, buff=0.22)
        box_f = pxbox(formula.get_width() + 0.2,
                      formula.get_height() + 0.15, WHITE)
        box_f.move_to(formula.get_center())
        self.play(Create(box_f), Write(formula), run_time=0.5)
        self.wait(0.9)
        self.play(FadeOut(banner), FadeOut(box_f),
                  FadeOut(formula), FadeOut(back_arrows), run_time=0.2)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 6 – Restart (35% mass back to query)
    # ══════════════════════════════════════════════════════════════════════════
    def _frame6(self):
        banner = self._banner("FRAME 6 · RESTART  α = 0.35")
        self.play(FadeIn(banner), run_time=0.3)

        q_node = self._get("q_node")
        feat_t = self._get("feat_t")

        alpha_lbl = Text("restart  α = 0.35",
                         font="Courier New", color=WHITE,
                         weight=BOLD).scale(0.36)
        alpha_lbl.to_corner(DR, buff=0.22)
        box_a = pxbox(alpha_lbl.get_width() + 0.2,
                      alpha_lbl.get_height() + 0.15, WHITE)
        box_a.move_to(alpha_lbl.get_center())
        self.play(Create(box_a), Write(alpha_lbl), run_time=0.4)

        # pulse from each feature back to query
        pulses = VGroup()
        for n in feat_t:
            a = px_arrow(n.get_left(), q_node.get_right(),
                         stroke=2, dashed=True)
            pulses.add(a)

        self.play(LaggedStart(*[Create(a) for a in pulses],
                              lag_ratio=0.1), run_time=0.5)

        for _ in range(3):
            self.play(q_node.animate.set_opacity(0.3), run_time=0.07)
            self.play(q_node.animate.set_opacity(1.0), run_time=0.07)

        self.wait(0.7)
        self.play(FadeOut(banner), FadeOut(box_a),
                  FadeOut(alpha_lbl), FadeOut(pulses), run_time=0.2)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 7 – 8-step walk, nodes fade/brighten
    # ══════════════════════════════════════════════════════════════════════════
    def _frame7(self):
        banner = self._banner("FRAME 7 · RANDOM WALK  ×8 STEPS")
        self.play(FadeIn(banner), run_time=0.3)

        cards  = self._get("cards")
        feat_s = self._get("feat_s")

        step_lbl = Text("step 1 / 8", font="Courier New",
                        color=WHITE, weight=BOLD).scale(0.32)
        step_lbl.to_corner(UL, buff=0.2)
        self.play(FadeIn(step_lbl), run_time=0.2)

        # relevance weights per step (0-1) for each card 0..3
        relevance = [
            [0.8, 0.7, 0.5, 0.4],
            [0.9, 0.8, 0.4, 0.3],
            [0.95, 0.85, 0.35, 0.25],
            [1.0, 0.9, 0.3, 0.2],
            [1.0, 0.95, 0.28, 0.18],
            [1.0, 0.95, 0.25, 0.15],
            [1.0, 0.98, 0.22, 0.12],
            [1.0, 1.0, 0.20, 0.10],
        ]

        for step in range(8):
            new_lbl = Text(f"step {step+1} / 8",
                           font="Courier New", color=WHITE,
                           weight=BOLD).scale(0.32)
            new_lbl.to_corner(UL, buff=0.2)
            self.remove(step_lbl)
            step_lbl = new_lbl
            self.add(step_lbl)

            r = relevance[step]
            anims = []
            for ci, op in enumerate(r):
                anims.append(cards[ci].animate.set_opacity(op))
            # fade irrelevant small features
            for fi, fn in enumerate(feat_s):
                fade = max(0.1, 1.0 - step * 0.1 - fi * 0.05)
                anims.append(fn.animate.set_opacity(fade))

            self.play(*anims, run_time=0.22)
            self.wait(0.05)

        self.wait(0.5)
        self.play(FadeOut(banner), FadeOut(step_lbl), run_time=0.2)

    # ══════════════════════════════════════════════════════════════════════════
    # FRAME 8 – Final ranking
    # ══════════════════════════════════════════════════════════════════════════
    def _frame8(self):
        # Clear everything
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.4)

        scan = self._scanlines()
        self.add(scan)

        banner = self._banner("FINAL RANKING",
                              "top retrieved examples")
        self.play(FadeIn(banner), run_time=0.3)

        ranking = [
            ("1.", "tuz aççıqdur",         "salt is bitter"),
            ("2.", "tuz şı̇̄rı̇̄n emes",       "salt is not sweet"),
            ("3.", "wa ǧarı̇̄bdin biri budur","This is one of the stranger tales"),
        ]

        cards = VGroup()
        for i, (num, src, tgt) in enumerate(ranking):
            card = pxbox(8.0, 0.54, WHITE, fill=(i == 0))
            card.move_to(UP * (0.6 - i * 0.72))
            num_t = Text(num, font="Courier New",
                         color=BG if i==0 else WHITE,
                         weight=BOLD).scale(0.32)
            num_t.move_to(card.get_left() + RIGHT * 0.35)
            src_t = Text(src, font="Courier New",
                         color=BG if i==0 else WHITE,
                         weight=BOLD).scale(0.28)
            src_t.move_to(card.get_center() + LEFT * 1.0)
            arr_t = Text("→", font="Courier New",
                         color=BG if i==0 else LGRAY,
                         weight=BOLD).scale(0.28)
            arr_t.move_to(card.get_center())
            tgt_t = Text(tgt, font="Courier New",
                         color=BG if i==0 else LGRAY,
                         weight=BOLD).scale(0.28)
            tgt_t.move_to(card.get_center() + RIGHT * 1.6)
            cards.add(VGroup(card, num_t, src_t, arr_t, tgt_t))

        self.play(LaggedStart(*[FadeIn(c, shift=DOWN*0.15)
                                for c in cards],
                              lag_ratio=0.3), run_time=1.0)

        # glitch highlight #1
        for _ in range(2):
            self.play(cards[0].animate.set_opacity(0.4), run_time=0.06)
            self.play(cards[0].animate.set_opacity(1.0), run_time=0.06)

        caption = Text(
            "graph_ppr does not translate.\nIt selects examples for the LLM prompt.",
            font="Courier New", color=LGRAY,
            weight=BOLD, line_spacing=1.1
        ).scale(0.275)
        caption.to_edge(DOWN, buff=0.2)
        box_c = pxbox(caption.get_width() + 0.3,
                      caption.get_height() + 0.15, LGRAY)
        box_c.move_to(caption.get_center())
        self.play(Create(box_c), Write(caption), run_time=0.7)

        self.wait(2.0)

    # ─── tiny state store ───────────────────────────────────────────────────
    def _keep(self, key, mob):
        if not hasattr(self, "_store"):
            self._store = {}
        self._store[key] = mob

    def _get(self, key):
        return self._store[key]