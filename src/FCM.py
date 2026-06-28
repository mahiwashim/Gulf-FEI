import math
import os
import textwrap
import logging
from typing import List, Dict, Tuple, Optional

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from FCM.extractor import CausalExtractor
from FCM.fcm_graph import FCMGraphBuilder
from FCM.schemas import Edge, FCMMap
from FCM.categorizer import get_summary_theme

MAX_CONCEPTS = 12
EDGE_THRESHOLD = 0.2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _wrap_label(text: str, width: int = 14) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def _arc_midpoint(p1: tuple, p2: tuple, rad: float) -> tuple:
    mx = (p1[0] + p2[0]) / 2
    my = (p1[1] + p2[1]) / 2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length < 1e-9:
        return mx, my
    px, py = -dy / length, dx / length
    return mx + px * rad * length * 0.5, my + py * rad * length * 0.5


# ── Main class ────────────────────────────────────────────────────────────────

class FCMGenerator:
    def __init__(
        self,
        max_concepts: int = MAX_CONCEPTS,
        edge_threshold: float = EDGE_THRESHOLD,
        max_edges: Optional[int] = None,
        directed: bool = True,
    ):
        self.max_concepts  = max_concepts
        self.edge_threshold = edge_threshold
        self.max_edges     = max_edges
        self.directed      = directed

        self._extractor     = CausalExtractor()
        self._graph_builder = FCMGraphBuilder()

    # ------------------------------------------------------------------
    def _extract_edges(self, text: str, num_results: int = 0) -> Tuple[List[str], List[Edge]]:
        chunks = [{"text": text, "score": 1.0, "doc_id": "context"}]
        concepts, edges = self._extractor.extract(chunks)
        if num_results:
            edges = edges[:num_results]
        return concepts, edges

    def _build_graph(self, query: str, edges: List[Edge]) -> Tuple[nx.DiGraph, "FCMMap"]:
        # Derive the concept set from the edges so the adjacency matrix is
        # populated (passing [] would leave the matrix empty).
        edge_concepts: List[str] = []
        seen: set = set()
        for e in edges:
            for node in (e.source, e.target):
                if node and node not in seen:
                    seen.add(node)
                    edge_concepts.append(node)

        fcm_map: FCMMap = self._graph_builder.build(query, edge_concepts, edges)
        G = nx.DiGraph() if self.directed else nx.Graph()
        # Carry richer attrs (edge_type, hops, confidence) so the plot can
        # style transitive edges differently, and the UI can show evidence.
        edge_attrs: Dict[Tuple[str, str], Edge] = {(e.source, e.target): e for e in edges}
        for edge in fcm_map.edges:
            src_edge = edge_attrs.get((edge.source, edge.target), edge)
            G.add_edge(
                edge.source, edge.target,
                weight     = edge.weight,
                polarity   = edge.polarity,
                edge_type  = getattr(src_edge, 'edge_type', 'pattern'),
                hops       = getattr(src_edge, 'hops', 1),
                confidence = getattr(src_edge, 'confidence', 0.7),
                evidence   = getattr(src_edge, 'evidence', ''),
            )
        return G, fcm_map

    # ------------------------------------------------------------------
    def build_fcm(self, text: str, query: str, num_results: int = 6) -> Tuple[nx.DiGraph, Dict]:
        _, all_edges = self._extract_edges(text, 0)

        # ── Quality gate: edge_threshold actually filters edges ─────────────
        # Tier 0: pattern edges (always ranked first — direct evidence).
        # Tier 1: co-occurrence and transitive compete on quality. This lets
        # high-confidence inferred 2-hop edges surface in dense graphs
        # instead of being pushed out by lower-quality co-occurrence.
        def _tier(e) -> int:
            return 0 if getattr(e, 'edge_type', 'pattern') == 'pattern' else 1

        ranked = sorted(
            all_edges,
            key=lambda e: (_tier(e), -(e.confidence * abs(e.weight))),
        )
        # Drop edges below the magnitude floor (always keep at least the
        # 5 strongest so the graph is never empty)
        floored = [e for e in ranked if abs(e.weight) >= self.edge_threshold]
        if len(floored) < min(5, len(ranked)):
            floored = ranked[: max(5, len(floored))]

        lim_edges = floored[:num_results] if num_results else floored

        G, fcm_map = self._build_graph(query, lim_edges)

        num_n = G.number_of_nodes()
        # Build a lookup for the rich edge metadata (Edge objects)
        rich_lookup: Dict[Tuple[str, str], Edge] = {(e.source, e.target): e for e in lim_edges}
        edges_meta = []
        for e in fcm_map.edges:
            src = rich_lookup.get((e.source, e.target), e)
            edges_meta.append({
                "source":     e.source,
                "target":     e.target,
                "weight":     round(float(e.weight), 3),
                "polarity":   e.polarity,
                "confidence": round(float(getattr(src, 'confidence', 0.7)), 3),
                "edge_type":  getattr(src, 'edge_type', 'pattern'),
                "hops":       int(getattr(src, 'hops', 1)),
                "evidence":   getattr(src, 'evidence', ''),
                "evidence_doc_id": getattr(src, 'evidence_doc_id', None),
            })
        type_tally = {}
        for em in edges_meta:
            type_tally[em["edge_type"]] = type_tally.get(em["edge_type"], 0) + 1
        details = {
            "nodes": fcm_map.concepts,
            "edges": [(e.source, e.target, e.weight) for e in fcm_map.edges],
            "edges_meta": edges_meta,
            "adjacency_matrix": fcm_map.adjacency_matrix,
            "stats": {
                "num_nodes":      num_n,
                "num_edges":      G.number_of_edges(),
                "average_degree": sum(dict(G.degree()).values()) / num_n if num_n else 0,
                "density":        nx.density(G),
                "edge_types":     type_tally,
            },
        }
        logger.info("Built FCM with %d nodes and %d edges (%s).",
                    num_n, G.number_of_edges(), type_tally)
        return G, details

    # ------------------------------------------------------------------
    def build_fcm_from_matrix(
        self,
        concepts: List[str],
        matrix: List[List[float]],
        query: str = "Uploaded Matrix",
    ) -> Tuple[nx.DiGraph, Dict]:
        """Build FCM graph directly from an adjacency matrix (e.g. uploaded CSV)."""
        G = nx.DiGraph() if self.directed else nx.Graph()
        edges_list: List[Tuple] = []

        for i, src in enumerate(concepts):
            for j, tgt in enumerate(concepts):
                if i == j:
                    continue
                try:
                    w = float(matrix[i][j])
                except (IndexError, TypeError, ValueError):
                    w = 0.0
                if w != 0.0:
                    G.add_edge(src, tgt, weight=w,
                               polarity="positive" if w > 0 else "negative")
                    edges_list.append((src, tgt, w))

        num_n = G.number_of_nodes()
        edges_meta = [
            {
                "source": s, "target": t, "weight": round(float(w), 3),
                "polarity": "positive" if w >= 0 else "negative",
                "confidence": 0.95, "edge_type": "matrix", "hops": 1,
                "evidence": "loaded from uploaded adjacency matrix",
                "evidence_doc_id": None,
            }
            for s, t, w in edges_list
        ]
        details = {
            "nodes": concepts,
            "edges": edges_list,
            "edges_meta": edges_meta,
            "adjacency_matrix": matrix,
            "stats": {
                "num_nodes":      num_n,
                "num_edges":      G.number_of_edges(),
                "average_degree": sum(dict(G.degree()).values()) / num_n if num_n else 0,
                "density":        nx.density(G),
                "edge_types":     {"matrix": len(edges_meta)},
            },
        }
        logger.info("Built FCM from matrix: %d nodes, %d edges.", num_n, len(edges_list))
        return G, details

    def format_edges_as_results(self, relations: List[Dict], limit: Optional[int] = None) -> List[Dict]:
        results = []
        for r in (relations[:limit] if limit else relations):
            try:
                results.append({
                    "result":   r.get("id", len(results) + 1),
                    "cause":    r.get("cause", "unknown"),
                    "effect":   r.get("effect", "unknown"),
                    "polarity": r.get("polarity", "+"),
                    "weight":   round(float(r.get("weight", 0.5)), 2),
                })
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping invalid relation: %s — %s", r, exc)
        return results

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_fcm(
        self,
        G: nx.DiGraph,
        filename: str = "fcm_graph",
        query: str = "",
    ) -> Optional[str]:
        """
        Publication-quality FCM — all labels guaranteed visible.

        Root causes of invisible text that are fixed here:
          1. No manual ax.set_xlim/ylim — labels were clipped outside the bounds
          2. Correct node-radius → data-unit conversion for label offsets
          3. Minimum font size 13 pt; weight badges 11 pt; stats panel 10 pt
          4. bbox_inches="tight" captures all text artists outside the axis
        """
        if not G.nodes:
            return None

        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch
        import matplotlib.patheffects as pe

        static_dir = os.path.join(os.getcwd(), "static")
        os.makedirs(static_dir, exist_ok=True)
        png_path = os.path.join(static_dir, f"{filename}.png")

        nodes = list(G.nodes())
        n     = len(nodes)

        # ── 1. FCM node metrics ───────────────────────────────────────
        in_str: Dict[str, float] = {
            nd: sum(abs(float(d.get("weight", 0))) for _, _, d in G.in_edges(nd, data=True))
            for nd in nodes
        }
        out_str: Dict[str, float] = {
            nd: sum(abs(float(d.get("weight", 0))) for _, _, d in G.out_edges(nd, data=True))
            for nd in nodes
        }
        total_str: Dict[str, float] = {nd: in_str[nd] + out_str[nd] for nd in nodes}

        # ── 2. FCM node role classification ──────────────────────────
        def _role(nd: str) -> str:
            o, i = out_str[nd], in_str[nd]
            if o == 0 and i == 0:
                return "Isolated"
            if i < 1e-9 or (o > 0 and o / (i + 1e-9) > 2.5):
                return "Transmitter"
            if o < 1e-9 or (i > 0 and i / (o + 1e-9) > 2.5):
                return "Receiver"
            return "Ordinary"

        roles: Dict[str, str] = {nd: _role(nd) for nd in nodes}

        ROLE_RING = {
            "Transmitter": "#E67E22",
            "Receiver":    "#2471A3",
            "Ordinary":    "#1E8449",
            "Isolated":    "#808B96",
        }
        ROLE_LW = {"Transmitter": 5.5, "Receiver": 5.0, "Ordinary": 4.5, "Isolated": 2.5}

        # ── 3. Centrality (importance) ────────────────────────────────
        try:
            btw = nx.betweenness_centrality(G, normalized=True)
        except Exception:
            btw = {nd: 0.5 for nd in nodes}
        deg = nx.degree_centrality(G)
        importance: Dict[str, float] = {
            nd: 0.55 * btw.get(nd, 0) + 0.45 * deg.get(nd, 0) for nd in nodes
        }
        mx_imp = max(importance.values()) if any(v > 0 for v in importance.values()) else 1.0

        def _rel(nd: str) -> float:
            return importance[nd] / max(mx_imp, 1e-9)

        # Node scatter size — wider dynamic range so important hubs visibly
        # dominate the map (more dramatic = more eye-catching).
        def _size(nd: str) -> float:
            return 900 + 2700 * _rel(nd)

        # Arrow shrink in display pts (keeps arrowheads off the circle surface)
        def _shrink(nd: str) -> float:
            return math.sqrt(_size(nd)) / 2.0 + 4

        # ── 4. Community / module detection (greedy modularity) ─────
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            UG    = G.to_undirected()
            comms = list(greedy_modularity_communities(UG)) if UG.number_of_edges() > 0 \
                    else [frozenset(nodes)]
        except Exception:
            comms = [frozenset(nodes)]
        if not comms:
            comms = [frozenset(nodes)]

        node_comm: Dict[str, int] = {nd: i for i, c in enumerate(comms) for nd in c}

        # Vivid, brand-forward module palette — high contrast on the light
        # canvas so distinct perception clusters pop at a glance. Brand colors
        # lead (teal / navy / data-current), then a curated vibrant spread.
        PALETTE = [
            "#056F73",  # Council Teal
            "#013056",  # Gulf Authority
            "#27A798",  # Data Current
            "#E8743B",  # coral
            "#7B4FB5",  # violet
            "#D63B6A",  # magenta
            "#2D9CDB",  # sky blue
            "#E2A33B",  # amber
            "#15A07F",  # emerald
            "#5B6CE0",  # indigo
            "#C0392B",  # brick
            "#0E8C99",  # deep teal
        ]

        def _ccolor(nd: str) -> str:
            return PALETTE[node_comm.get(nd, 0) % len(PALETTE)]

        # ── 5. Layout + mandatory spacing enforcement ─────────────────
        # Kamada-Kawai gives structured topology but packs dense graphs
        # too tightly.  We rescale the raw positions so every adjacent
        # pair of nodes is visually separated, then enforce a hard
        # minimum-distance floor to eliminate residual overlaps.
        try:
            pos = nx.kamada_kawai_layout(G, weight=None)
        except Exception:
            try:
                pos = nx.spring_layout(
                    G, k=4.0 / math.sqrt(max(n, 1)), iterations=600, seed=42
                )
            except Exception:
                pos = nx.circular_layout(G)

        # Rescale: spread positions so the layout spans ±target_half
        # target_half grows with node count to keep per-node area constant
        target_half = max(2.9, math.sqrt(n) * 0.80)
        xs = [v[0] for v in pos.values()]
        ys = [v[1] for v in pos.values()]
        x_mid = (max(xs) + min(xs)) / 2
        y_mid = (max(ys) + min(ys)) / 2
        x_span = max(max(xs) - min(xs), 1e-9)
        y_span = max(max(ys) - min(ys), 1e-9)
        pos = {
            nd: (
                (x - x_mid) / x_span * target_half * 2,
                (y - y_mid) / y_span * target_half * 2,
            )
            for nd, (x, y) in pos.items()
        }

        # Hard minimum-distance floor: push overlapping nodes apart
        min_sep = 1.15   # data units — wider gap gives each relation room to breathe
        _moved  = True
        _iters  = 0
        while _moved and _iters < 45:
            _moved = False
            _iters += 1
            node_list = list(pos.keys())
            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    na, nb = node_list[i], node_list[j]
                    ax_, ay_ = pos[na]
                    bx_, by_ = pos[nb]
                    dx_, dy_ = bx_ - ax_, by_ - ay_
                    dist_   = math.sqrt(dx_ ** 2 + dy_ ** 2) + 1e-12
                    if dist_ < min_sep:
                        push    = (min_sep - dist_) / 2.0
                        ux_, uy_ = dx_ / dist_, dy_ / dist_
                        pos[na] = (ax_ - ux_ * push, ay_ - uy_ * push)
                        pos[nb] = (bx_ + ux_ * push, by_ + uy_ * push)
                        _moved  = True

        # Graph centroid for radial label direction
        cx_all = sum(p[0] for p in pos.values()) / n
        cy_all = sum(p[1] for p in pos.values()) / n

        # ── 6. Figure — DO NOT set explicit xlim/ylim ─────────────────
        # Reason: explicit limits clip external labels placed outside
        # the data range.  bbox_inches="tight" captures all artists.
        FIG_W, FIG_H = 28, 20
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        CANVAS = "#E7EEF0"            # deeper Tidal Mist (no white anywhere)
        ax.set_facecolor(CANVAS)     # panels/labels sit on top in lighter #F4F7F6
        fig.patch.set_facecolor(CANVAS)
        ax.axis("off")

        # ── 7. Edges ─────────────────────────────────────────────────
        C_POS = "#1FA85C"   # vivid green — reinforcing (+)
        C_NEG = "#E0413A"   # vivid red   — opposing (−)

        all_edges   = list(G.edges(data=True))
        pos_edges   = [(u, v, d) for u, v, d in all_edges if float(d.get("weight", 0)) >= 0]
        neg_edges   = [(u, v, d) for u, v, d in all_edges if float(d.get("weight", 0)) < 0]
        drawn_pairs: set = set()

        logger.info("FCM plot: %d positive, %d negative edges",
                    len(pos_edges), len(neg_edges))

        def _draw_edge(u: str, v: str, d: dict,
                       base_color: str, base_rad: float, z: int) -> None:
            w   = float(d.get("weight", 0))
            lw  = 2.0 + abs(w) * 4.2
            rad = -base_rad if (v, u) in drawn_pairs else base_rad
            drawn_pairs.add((u, v))

            etype = d.get("edge_type", "pattern")
            conf  = float(d.get("confidence", 0.7))
            # Visual differentiation: transitive (inferred) edges get a dashed
            # stroke and faded alpha so reviewers can tell them apart from
            # primary, evidence-backed (pattern/cooccurrence/matrix) edges.
            if etype == "transitive":
                style = (0, (5, 3))
                edge_alpha = 0.55
                lw = max(1.0, lw * 0.75)
            else:
                style = "solid"
                edge_alpha = 0.65 + 0.30 * min(1.0, conf)

            arrow = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=30,
                color=base_color,
                linewidth=lw,
                alpha=edge_alpha,
                linestyle=style,
                shrinkA=_shrink(u),
                shrinkB=_shrink(v),
                zorder=z,
            )
            # White halo behind each edge so it stays crisp and vivid where it
            # crosses nodes or other edges (the key to an eye-catching network).
            arrow.set_path_effects([
                pe.withStroke(linewidth=lw + 3.4, foreground="white", alpha=0.6),
            ])
            ax.add_patch(arrow)

            # Weight badge — white bold text on a solid color pill with a thin
            # white outline so it reads cleanly over any background.
            lx, ly = _arc_midpoint(pos[u], pos[v], rad)
            badge  = f"+{abs(w):.2f}" if w >= 0 else f"-{abs(w):.2f}"
            ax.text(
                lx, ly, badge,
                fontsize=12,
                color="white",
                ha="center", va="center", fontweight="bold",
                bbox=dict(
                    facecolor=C_POS if w >= 0 else C_NEG,
                    alpha=0.98, edgecolor="white", linewidth=1.3,
                    boxstyle="round,pad=0.3",
                ),
                zorder=z + 2,
            )

        for u, v, d in pos_edges:
            _draw_edge(u, v, d, C_POS,  0.22, 2)
        for u, v, d in neg_edges:
            _draw_edge(u, v, d, C_NEG, -0.22, 3)

        # ── 8. Nodes + external labels ────────────────────────────────
        # After rescaling, positions span ±target_half ≈ 1.8–4.
        # Figure: 28 in, plot width ≈ 28*0.64 = 17.9 in (rect leaves 14% left, 22% right).
        # Data range = 2 * target_half.
        # pts/data-unit = (17.9 / (2*target_half)) * 72
        data_range = target_half * 2
        SCALE = (19.0 / data_range) * 72      # pts per data unit

        for nd in nodes:
            x, y  = pos[nd]
            color = _ccolor(nd)
            sz    = _size(nd)
            rel   = _rel(nd)
            role  = roles[nd]

            # Soft colored glow halo — gives each node an aura in its module
            # color and lifts it off the canvas (eye-catching depth).
            ax.scatter(x, y, s=sz * 2.4, c=color, alpha=0.13,
                       linewidths=0, zorder=2)
            # Role ring (outer)
            ax.scatter(x, y, s=sz * 1.80,
                       c="none", edgecolors=ROLE_RING[role],
                       linewidths=ROLE_LW[role], zorder=4)
            # Main circle with a drop shadow for a glossy, lifted look
            main = ax.scatter(x, y, s=sz, c=color, alpha=0.96,
                              edgecolors="white", linewidths=3.0, zorder=5)
            main.set_path_effects([
                pe.SimplePatchShadow(offset=(2.5, -2.5), alpha=0.22),
                pe.Normal(),
            ])

            # Radial direction: outward from graph centroid
            dx, dy = x - cx_all, y - cy_all
            dist   = math.sqrt(dx ** 2 + dy ** 2) + 1e-9
            ux, uy = dx / dist, dy / dist

            # Visual node radius in data units, plus clearance gap
            r_data  = math.sqrt(sz / math.pi) / SCALE
            lbl_off = r_data + 0.32          # larger gap → keyword clearly separated
            tx, ty  = x + ux * lbl_off, y + uy * lbl_off

            # Bigger cause/effect keyword labels so they are easy to read
            fsize = max(15, int(15 + 6 * rel))

            # Wrap at 15 chars so no label exceeds 3 lines
            label = _wrap_label(nd, width=15)

            # Connector: starts at node surface, ends at label center
            ax.plot(
                [x + ux * r_data * 1.05, tx],
                [y + uy * r_data * 1.05, ty],
                color=color, lw=1.8, alpha=0.65, zorder=4,
                solid_capstyle="round",
            )

            ax.text(
                tx, ty, label,
                ha="center", va="center",
                fontsize=fsize, fontweight="bold",
                color="#0B1F30",
                zorder=7,
                multialignment="center",
                linespacing=1.35,
                bbox=dict(
                    facecolor="#F4F7F6", alpha=0.98,
                    edgecolor=color, linewidth=2.6,
                    boxstyle="round,pad=0.46",
                ),
            )

        # ── 9. Legend ─────────────────────────────────────────────────
        legend_handles: list = []

        for i, comm in enumerate(comms):
            color  = PALETTE[i % len(PALETTE)]
            rep_nd = max(comm, key=lambda nd: importance.get(nd, 0))
            rep    = (rep_nd[:26] + "...") if len(rep_nd) > 26 else rep_nd
            legend_handles.append(
                mpatches.Patch(facecolor=color, edgecolor="white",
                               linewidth=1.5,
                               label=f"Module {i + 1}  -  {rep}")
            )

        legend_handles.append(
            mpatches.Patch(facecolor="none", edgecolor="none", label="")
        )
        n_transitive = sum(
            1 for _, _, d in all_edges
            if d.get("edge_type") == "transitive"
        )
        legend_handles += [
            Line2D([0], [0], color=C_POS, linewidth=4.5,
                   label=f"Positive causal link   n = {len(pos_edges)}"),
            Line2D([0], [0], color=C_NEG, linewidth=4.5,
                   label=f"Negative causal link   n = {len(neg_edges)}"),
        ]
        if n_transitive:
            legend_handles.append(
                Line2D([0], [0], color="#5a7099", linewidth=2.5,
                       linestyle=(0, (5, 3)),
                       label=f"Inferred (2-hop)      n = {n_transitive}")
            )
        legend_handles.append(
            mpatches.Patch(facecolor="none", edgecolor="none", label="")
        )
        for role, rcolor in ROLE_RING.items():
            rc = sum(1 for v in roles.values() if v == role)
            if rc > 0:
                legend_handles.append(
                    Line2D([0], [0], marker="o", color="none",
                           markerfacecolor="none",
                           markeredgecolor=rcolor,
                           markeredgewidth=4.5, markersize=15,
                           label=f"{role}   n = {rc}")
                )

        # Legend anchored to the figure's TOP-RIGHT corner (in the reserved
        # right margin) so it never overlaps node labels in the plot area.
        leg = fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.988, 0.94),
            framealpha=0.97,
            facecolor="#F4F7F6",
            edgecolor="#cfdadd",
            labelcolor="#0d1b2a",
            fontsize=11,
            title="Modules  |  Edge Types  |  Node Roles",
            title_fontsize=12.5,
            borderpad=1.1,
            labelspacing=0.78,
        )
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_color("#013056")
        leg.set_zorder(20)   # draw above every node label (z=7) and edge badge (z=5)

        # ── 10. Statistics panel ──────────────────────────────────────
        density    = nx.density(G)
        avg_deg    = sum(dict(G.degree()).values()) / n if n > 0 else 0.0
        top5       = sorted(total_str.items(), key=lambda x: -x[1])[:5]
        role_counts = {r: sum(1 for v in roles.values() if v == r) for r in ROLE_RING}

        top5_lines = "\n".join(
            f"  {idx + 1}. {nd[:20]:<20}  {v:.3f}"
            for idx, (nd, v) in enumerate(top5)
        )
        sep = "-" * 30
        stats_block = (
            f"  Graph Statistics\n"
            f"  {sep}\n"
            f"  Concepts (nodes) : {n}\n"
            f"  Causal edges     : {G.number_of_edges()}\n"
            f"  Positive (+)     : {len(pos_edges)}\n"
            f"  Negative (-)     : {len(neg_edges)}\n"
            f"  Graph density    : {density:.4f}\n"
            f"  Avg degree       : {avg_deg:.2f}\n"
            f"  Modules          : {len(comms)}\n"
            f"\n"
            f"  Node Roles\n"
            f"  {sep}\n"
            f"  Transmitters     : {role_counts.get('Transmitter', 0)}\n"
            f"  Receivers        : {role_counts.get('Receiver', 0)}\n"
            f"  Ordinary         : {role_counts.get('Ordinary', 0)}\n"
            f"  Isolated         : {role_counts.get('Isolated', 0)}\n"
            f"\n"
            f"  Top Concepts (Total Strength)\n"
            f"  {sep}\n"
            f"{top5_lines}"
        )

        fig.text(
            0.012, 0.94,
            stats_block,
            fontsize=10,            # ← 10 pt — readable on web and print
            color="#0d1b2a",
            fontfamily="monospace",
            va="top", ha="left",
            bbox=dict(facecolor="#F4F7F6", alpha=0.96,
                      edgecolor="#cfdadd", linewidth=1.0,
                      boxstyle="round,pad=0.6"),
            transform=fig.transFigure,
        )

        # ── 11. Title ─────────────────────────────────────────────────
        q_line = (
            f'\nQuery: "{query[:85]}{"..." if len(query) > 85 else ""}"'
            if query else ""
        )
        ax.set_title(
            f"Gulf FEI  |  Public Perception of Gulf of Mexico Fisheries\n"
            f"Fuzzy Cognitive Map  -  Perceived Causal Network from Community Discourse"
            f"{q_line}",
            fontsize=17, fontweight="bold", color="#013056",
            pad=26, linespacing=1.85,
        )

        # ── 12. Methodological footnote ───────────────────────────────
        fig.text(
            0.5, 0.005,
            "Built from public discourse (blogs · forums · YouTube · podcasts)  "
            "|  Node size proportional to betweenness centrality  "
            "|  Border ring = FCM role: orange=Transmitter  blue=Receiver  "
            "green=Ordinary  gray=Isolated  "
            "|  Fill = perception theme (greedy modularity)  "
            "|  Arrow width proportional to |weight|  "
            "|  Weight scale [-1.0, +1.0]  "
            "|  Layout: Kamada-Kawai energy minimization",
            ha="center", va="bottom",
            fontsize=9.5, color="#056F73", style="italic",
        )

        # rect reserves margins: left for the legend, right for the stats panel
        plt.tight_layout(rect=[0.17, 0.03, 0.83, 0.93])
        plt.savefig(
            png_path, dpi=200,
            bbox_inches="tight",    # captures legend + labels outside axis bounds
            facecolor="#E7EEF0", edgecolor="none",
        )
        plt.close(fig)

        logger.info("FCM saved -> %s  (%d nodes, %d modules, %d+/%d-)",
                    png_path, n, len(comms), len(pos_edges), len(neg_edges))
        return f"/static/{filename}.png"

    # ------------------------------------------------------------------
    # Summary / Thematic Aggregation (mirrors R summary_mat_final pipeline)
    # ------------------------------------------------------------------

    def aggregate_summary_graph(
        self,
        G: nx.DiGraph,
        weight_threshold: float = 0.5,
    ) -> nx.DiGraph:
        """Collapse a full FCM graph into thematic summary nodes.

        Each original node is mapped to its SummaryTheme via FCM.categorizer.
        Edge weights between same-theme pairs are summed (matching the R
        rowsum/colsum consolidation). Self-loops and edges whose absolute
        aggregated weight is <= weight_threshold are dropped.
        """
        if G is None or G.number_of_nodes() == 0:
            return nx.DiGraph()

        sum_weights: Dict[Tuple[str, str], float] = {}
        for u, v, data in G.edges(data=True):
            s = get_summary_theme(u)
            t = get_summary_theme(v)
            if not s or not t or s == t:
                continue
            w = float(data.get("weight", 0.0))
            sum_weights[(s, t)] = sum_weights.get((s, t), 0.0) + w

        H = nx.DiGraph()
        for (s, t), w in sum_weights.items():
            if abs(w) <= weight_threshold:
                continue
            H.add_edge(s, t, weight=w,
                       polarity="positive" if w >= 0 else "negative")
        return H

    def plot_summary_fcm(
        self,
        G: nx.DiGraph,
        filename: str = "fcm_summary",
        query: str = "",
        weight_threshold: float = 0.5,
    ) -> Optional[str]:
        """Build the summary-theme graph and render it with the same styling."""
        H = self.aggregate_summary_graph(G, weight_threshold=weight_threshold)
        if H.number_of_nodes() == 0:
            return None
        return self.plot_fcm(H, filename=filename, query=query)
