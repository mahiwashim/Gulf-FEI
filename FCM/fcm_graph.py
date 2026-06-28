from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import networkx as nx

from .schemas import Edge, FCMMap


class FCMGraphBuilder:
    def build(self, query: str, concepts: List[str], edges: List[Edge]) -> FCMMap:
        ordered = sorted(dict.fromkeys(c for c in concepts if c))
        index = {c: i for i, c in enumerate(ordered)}
        matrix = [[0.0 for _ in ordered] for _ in ordered]
        for edge in edges:
            if edge.source in index and edge.target in index:
                matrix[index[edge.source]][index[edge.target]] = float(edge.weight)
        return FCMMap(query=query, concepts=ordered, edges=edges, adjacency_matrix=matrix)

    def export_network(self, fcm_map: FCMMap, outdir: str | Path) -> Dict[str, str]:
        outdir = Path(outdir).resolve()
        outdir.mkdir(parents=True, exist_ok=True)
        json_path = outdir / 'fcm_map.json'
        gexf_path = outdir / 'fcm_map.gexf'
        csv_path = outdir / 'adjacency_matrix.csv'
        png_path = outdir / 'fcm_map.png'
        edge_csv_path = outdir / 'edge_list.csv'
        node_csv_path = outdir / 'node_metrics.csv'

        with json_path.open('w', encoding='utf-8') as fh:
            fh.write(fcm_map.model_dump_json(indent=2))

        with csv_path.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow([''] + fcm_map.concepts)
            for concept, row in zip(fcm_map.concepts, fcm_map.adjacency_matrix):
                writer.writerow([concept] + row)

        graph = nx.DiGraph()
        for concept in fcm_map.concepts:
            graph.add_node(concept)
        for edge in fcm_map.edges:
            graph.add_edge(
                edge.source,
                edge.target,
                weight=edge.weight,
                abs_weight=abs(edge.weight),
                sign='positive' if edge.weight >= 0 else 'negative',
                confidence=edge.confidence,
                polarity=edge.polarity,
            )

        if graph.number_of_nodes() > 0:
            in_strength = {
                n: sum(abs(data.get('weight', 0.0)) for _, _, data in graph.in_edges(n, data=True))
                for n in graph.nodes
            }
            out_strength = {
                n: sum(abs(data.get('weight', 0.0)) for _, _, data in graph.out_edges(n, data=True))
                for n in graph.nodes
            }
            total_strength = {n: in_strength[n] + out_strength[n] for n in graph.nodes}
            nx.set_node_attributes(graph, in_strength, 'in_strength')
            nx.set_node_attributes(graph, out_strength, 'out_strength')
            nx.set_node_attributes(graph, total_strength, 'total_strength')

        nx.write_gexf(graph, gexf_path)

        with edge_csv_path.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['source', 'target', 'weight', 'abs_weight', 'confidence', 'sign'])
            for source, target, data in graph.edges(data=True):
                writer.writerow(
                    [
                        source,
                        target,
                        round(float(data.get('weight', 0.0)), 4),
                        round(float(abs(data.get('weight', 0.0))), 4),
                        round(float(data.get('confidence', 0.0)), 4),
                        data.get('sign', ''),
                    ]
                )

        with node_csv_path.open('w', encoding='utf-8', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['concept', 'in_degree', 'out_degree', 'in_strength', 'out_strength', 'total_strength'])
            for node in graph.nodes:
                writer.writerow(
                    [
                        node,
                        graph.in_degree(node),
                        graph.out_degree(node),
                        round(float(graph.nodes[node].get('in_strength', 0.0)), 4),
                        round(float(graph.nodes[node].get('out_strength', 0.0)), 4),
                        round(float(graph.nodes[node].get('total_strength', 0.0)), 4),
                    ]
                )

        try:
            import math
            import textwrap
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D
            from matplotlib.patches import FancyArrowPatch

            if graph.number_of_nodes() > 0:
                nodes_g = list(graph.nodes())
                ng = len(nodes_g)

                # Node strength metrics
                node_strength = [float(graph.nodes[nd].get('total_strength', 0.0)) for nd in nodes_g]
                min_s = min(node_strength) if node_strength else 0.0
                max_s = max(node_strength) if node_strength else 1.0

                def _ns(v: float) -> float:
                    if max_s == min_s:
                        return 2000.0
                    return 1800.0 + ((v - min_s) / (max_s - min_s)) * 3800.0

                sizes_g = [_ns(s) for s in node_strength]

                # Kamada-Kawai layout
                try:
                    pos_g = nx.kamada_kawai_layout(graph, weight=None)
                except Exception:
                    pos_g = nx.spring_layout(graph, seed=42,
                                             k=2.8 / max(1, ng ** 0.5),
                                             iterations=500)

                fig_g, ax_g = plt.subplots(figsize=(18, 13))
                ax_g.set_facecolor('#F7F9FC')
                fig_g.patch.set_facecolor('#F7F9FC')
                ax_g.axis('off')
                ax_g.set_xlim(-1.45, 1.45)
                ax_g.set_ylim(-1.45, 1.45)

                cx_g = sum(p[0] for p in pos_g.values()) / ng
                cy_g = sum(p[1] for p in pos_g.values()) / ng

                C_POS_G = '#1A6B3C'
                C_NEG_G = '#B03A2E'

                positive_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                                  if float(d.get('weight', 0.0)) >= 0]
                negative_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                                  if float(d.get('weight', 0.0)) < 0]

                drawn_g: set = set()

                def _shrink_g(nd: str) -> float:
                    idx = nodes_g.index(nd) if nd in nodes_g else 0
                    return math.sqrt(sizes_g[idx]) / 2.0 + 5

                def _arc_mp(p1, p2, rad):
                    mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
                    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
                    length = (dx**2 + dy**2)**0.5
                    if length < 1e-9:
                        return mx, my
                    px, py = -dy/length, dx/length
                    return mx + px*rad*length*0.5, my + py*rad*length*0.5

                def _draw_g(u, v, d, col, rad, z):
                    w  = float(d.get('weight', 0.0))
                    lw = 1.4 + abs(w) * 3.0
                    r  = -rad if (v, u) in drawn_g else rad
                    drawn_g.add((u, v))
                    ax_g.add_patch(FancyArrowPatch(
                        posA=pos_g[u], posB=pos_g[v],
                        connectionstyle=f'arc3,rad={r}',
                        arrowstyle='-|>', mutation_scale=24,
                        color=col, linewidth=lw, alpha=0.90,
                        shrinkA=_shrink_g(u), shrinkB=_shrink_g(v), zorder=z,
                    ))
                    lx, ly = _arc_mp(pos_g[u], pos_g[v], r)
                    badge = f'+{abs(w):.2f}' if w >= 0 else f'\u2212{abs(w):.2f}'
                    ax_g.text(lx, ly, badge, fontsize=9, color='white',
                              ha='center', va='center', fontweight='bold',
                              bbox=dict(facecolor=C_POS_G if w >= 0 else C_NEG_G,
                                        alpha=0.95, edgecolor='none',
                                        boxstyle='round,pad=0.24'),
                              zorder=z + 2)

                for u, v, d in positive_edges:
                    _draw_g(u, v, d, C_POS_G,  0.20, 2)
                for u, v, d in negative_edges:
                    _draw_g(u, v, d, C_NEG_G, -0.20, 3)

                # Draw nodes + external labels
                for idx, nd in enumerate(nodes_g):
                    x_g, y_g = pos_g[nd]
                    sz_g = sizes_g[idx]
                    ax_g.scatter(x_g, y_g, s=sz_g * 1.65,
                                 c='none', edgecolors='#2C3E50',
                                 linewidths=3.5, zorder=4)
                    ax_g.scatter(x_g, y_g, s=sz_g, c='#2C6E9E',
                                 alpha=0.88, edgecolors='white',
                                 linewidths=2.0, zorder=5)
                    # External label
                    dx_g = x_g - cx_g
                    dy_g = y_g - cy_g
                    dist_g = math.sqrt(dx_g**2 + dy_g**2) + 1e-9
                    ux_g, uy_g = dx_g/dist_g, dy_g/dist_g
                    off_g = math.sqrt(sz_g)/130.0 + 0.13
                    tx_g  = x_g + ux_g * off_g
                    ty_g  = y_g + uy_g * off_g
                    label_g = '\n'.join(textwrap.wrap(nd, width=14))
                    rel_g   = (node_strength[idx] - min_s) / (max_s - min_s + 1e-9)
                    fsize_g = max(9, int(9 + 4 * rel_g))
                    ax_g.plot([x_g, tx_g], [y_g, ty_g],
                              color='#2C6E9E', lw=0.9, alpha=0.5, zorder=4)
                    ax_g.text(tx_g, ty_g, label_g,
                              ha='center', va='center',
                              fontsize=fsize_g, fontweight='bold',
                              color='#0d1b2a', zorder=7,
                              multialignment='center', linespacing=1.3,
                              bbox=dict(facecolor='white', alpha=0.92,
                                        edgecolor='#2C6E9E', linewidth=1.4,
                                        boxstyle='round,pad=0.30'))

                legend_g = [
                    Line2D([0], [0], color=C_POS_G, linewidth=3.5,
                           label=f'Positive causal link  (n={len(positive_edges)})'),
                    Line2D([0], [0], color=C_NEG_G, linewidth=3.5,
                           label=f'Negative causal link  (n={len(negative_edges)})'),
                ]
                leg_g = ax_g.legend(handles=legend_g, loc='upper right',
                                    framealpha=0.96, facecolor='white',
                                    edgecolor='#cccccc', fontsize=10.5,
                                    title='Edge Types', title_fontsize=11.5)
                leg_g.get_title().set_fontweight('bold')
                leg_g.get_title().set_color('#013056')

                ax_g.set_title(
                    'Gulf of Mexico FEI  \u00b7  Fuzzy Cognitive Map\n'
                    'Signed Causal Ecosystem Network',
                    fontsize=14, fontweight='bold', color='#013056',
                    pad=18, linespacing=1.7,
                )
                plt.figtext(
                    0.5, 0.008,
                    f'Nodes={graph.number_of_nodes()}  |  Edges={graph.number_of_edges()}  |  '
                    'Node size = total causal strength  |  '
                    'Green = positive  |  Red = negative  |  Layout: Kamada-Kawai',
                    ha='center', fontsize=9, color='#5a7099', style='italic',
                )
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                plt.savefig(png_path, dpi=240, bbox_inches='tight', facecolor='#F7F9FC')
                plt.close()
        except Exception:
            pass

        return {
            'json': str(json_path),
            'gexf': str(gexf_path),
            'csv': str(csv_path),
            'png': str(png_path) if png_path.exists() else '',
            'edge_csv': str(edge_csv_path),
            'node_csv': str(node_csv_path),
        }
