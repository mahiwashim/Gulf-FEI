"""Generate FCM_Walkthrough.ipynb — a self-contained walkthrough notebook for
the Gulf FEI Fuzzy Cognitive Map pipeline. Re-run this script any time the
underlying FCM modules change to refresh the notebook.

    python -m FCM.docs._build_notebook
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parents[2]
FCM_DIR = REPO_ROOT / "FCM"
OUT_PATH = Path(__file__).resolve().parent / "FCM_Walkthrough.ipynb"


def _read(rel: str) -> str:
    """Return the source of a project file relative to the repo root."""
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src)


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list = []

    # ── 0. Title ──────────────────────────────────────────────────────────
    cells.append(md(
        "# Gulf FEI — Fuzzy Cognitive Map (FCM) Walkthrough\n"
        "\n"
        "**Audience:** Gulf FEI research team\n"
        "**Goal:** end-to-end reference for how the FCM subsystem turns "
        "research text into a signed causal map, and how to extend or audit it.\n"
        "\n"
        "This notebook is generated from the live source tree by "
        "`FCM/docs/_build_notebook.py`. Every code cell labelled *Source* "
        "below is a verbatim copy of the production module — what you read "
        "here is what runs in the web app.\n"
        "\n"
        "## Contents\n"
        "1. What is a Fuzzy Cognitive Map?\n"
        "2. System architecture\n"
        "3. Data contract — `schemas.py`\n"
        "4. Causal extraction — `extractor.py`\n"
        "5. Graph assembly — `fcm_graph.py`\n"
        "6. High-level orchestrator — `src/FCM.py`\n"
        "7. Thematic categorisation — `categorizer.py`\n"
        "8. Scenario simulation — `simulator.py`\n"
        "9. End-to-end demo (runnable)\n"
        "10. Quality controls and validation\n"
    ))

    # ── 1. What is an FCM ─────────────────────────────────────────────────
    cells.append(md(
        "## 1. What is a Fuzzy Cognitive Map?\n"
        "\n"
        "A Fuzzy Cognitive Map (FCM, Kosko 1986) is a **signed, weighted, "
        "directed graph** whose nodes are domain *concepts* and whose edges "
        "encode *causal influence*:\n"
        "\n"
        "- **Sign** (+/-) — does the source reinforce or oppose the target?\n"
        "- **Weight** in `[-1, +1]` — how strong is the influence?\n"
        "- **Direction** — which way does the causal arrow point?\n"
        "\n"
        "Once a map is built, you can run *what-if* scenarios by clamping a "
        "subset of concepts (the **drivers**) and propagating activation "
        "through the matrix until the system reaches equilibrium. This makes "
        "FCMs a lightweight bridge between qualitative expert knowledge and "
        "quantitative simulation — exactly what NOAA's Ecosystem-Based "
        "Fisheries Management (EBFM) framework calls for in the Gulf of Mexico."
    ))

    # ── 2. Architecture ──────────────────────────────────────────────────
    cells.append(md(
        "## 2. System architecture\n"
        "\n"
        "```\n"
        " User question / uploaded file\n"
        "        |\n"
        "        v\n"
        "  +--------------------+\n"
        "  | RAG retrieval +    |   (src/rag.py, src/vector_loader.py)\n"
        "  | LLM answer         |\n"
        "  +--------------------+\n"
        "        | text bundle (Q + A + retrieved context)\n"
        "        v\n"
        "  +--------------------+\n"
        "  | CausalExtractor    |   (FCM/extractor.py)\n"
        "  |  - regex patterns  |\n"
        "  |  - co-occurrence   |\n"
        "  |  - transitive 2hop |\n"
        "  +--------------------+\n"
        "        | concepts + signed edges\n"
        "        v\n"
        "  +--------------------+\n"
        "  | FCMGraphBuilder    |   (FCM/fcm_graph.py)\n"
        "  |  - adjacency mat.  |\n"
        "  |  - exports         |\n"
        "  +--------------------+\n"
        "        | FCMMap\n"
        "        v\n"
        "  +--------------------+\n"
        "  | FCMGenerator       |   (src/FCM.py)\n"
        "  |  - networkx graph  |\n"
        "  |  - publication plot|\n"
        "  |  - thematic summary|\n"
        "  +--------------------+\n"
        "        | adjacency + plots\n"
        "        v\n"
        "  +--------------------+\n"
        "  | FCMSimulator       |   (FCM/simulator.py)\n"
        "  |  - Kosko propagate |\n"
        "  |  - sigmoid/tanh    |\n"
        "  +--------------------+\n"
        "```\n"
    ))

    # ── 3. Data contract ────────────────────────────────────────────────
    cells.append(md(
        "## 3. Data contract — `FCM/schemas.py`\n"
        "\n"
        "Pydantic models that fix the shape of every object passed between "
        "stages. The two that matter most are `Edge` (a single causal claim "
        "with its evidence) and `FCMMap` (the assembled graph).\n"
    ))
    cells.append(md("> **Source — `FCM/schemas.py`**"))
    cells.append(code(_read("FCM/schemas.py")))

    # ── 4. Extractor ─────────────────────────────────────────────────────
    cells.append(md(
        "## 4. Causal extraction — `FCM/extractor.py`\n"
        "\n"
        "This is the heart of the system. Three complementary strategies "
        "feed the graph:\n"
        "\n"
        "1. **Pattern matching** — a registry of compiled regexes "
        "(`CAUSAL_PATTERNS`) detects explicit causal verbs "
        "(*increases, reduces, leads to, due to, …*) and assigns a base "
        "weight and polarity to each match.\n"
        "2. **Co-occurrence fallback** — when explicit patterns are sparse, "
        "any two domain keywords appearing in the same 3-sentence rolling "
        "window are connected. The window's dominant sentiment "
        "(`_NEG_MARKERS` vs `_POS_MARKERS`) sets the sign so co-occurrence "
        "edges are not blindly positive.\n"
        "3. **Transitive closure** — a damped 2-hop inference fires "
        "`A → B → C ⇒ A → C` so reviewers can see second-order pathways.\n"
        "\n"
        "Concept names pass through two strict filters:\n"
        "\n"
        "- `normalize_concept` — strips articles, useless modifiers from "
        "either end, dedupes consecutive duplicates, title-cases while "
        "preserving acronyms.\n"
        "- `_is_clean_concept` — rejects clause fragments, interrogatives "
        "anywhere in the span, all-modifier phrases, vague single-word "
        "nouns, and >50%-stopword spans.\n"
    ))
    cells.append(md("> **Source — `FCM/extractor.py`**"))
    cells.append(code(_read("FCM/extractor.py")))

    # ── 5. Graph builder ────────────────────────────────────────────────
    cells.append(md(
        "## 5. Graph assembly — `FCM/fcm_graph.py`\n"
        "\n"
        "Takes the cleaned concept list and edges and packages them into the "
        "canonical `FCMMap` (concepts in alphabetical order; adjacency matrix "
        "indexed by that order). The `export_network` helper writes JSON, "
        "GEXF, two CSVs (adjacency + edge list) and a publication-quality "
        "PNG to disk — useful when sharing artefacts with collaborators.\n"
    ))
    cells.append(md("> **Source — `FCM/fcm_graph.py`**"))
    cells.append(code(_read("FCM/fcm_graph.py")))

    # ── 6. High level orchestrator ──────────────────────────────────────
    cells.append(md(
        "## 6. High-level orchestrator — `src/FCM.py`\n"
        "\n"
        "Wraps `CausalExtractor + FCMGraphBuilder` behind two convenience "
        "methods used by the FastAPI app:\n"
        "\n"
        "- `build_fcm(text, query, num_results)` — the standard path used "
        "by `/query` and `/upload-files` on plain text.\n"
        "- `build_fcm_from_matrix(concepts, matrix)` — bypass extraction when "
        "the user uploads an adjacency CSV / XLSX.\n"
        "\n"
        "It also owns the publication-quality renderer (`plot_fcm`) with "
        "Kamada–Kawai layout, betweenness-centrality node sizing, greedy "
        "modularity colouring, role rings (transmitter / receiver / ordinary "
        "/ isolated) and an embedded statistics panel.\n"
    ))
    cells.append(md("> **Source — `src/FCM.py`**"))
    cells.append(code(_read("src/FCM.py")))

    # ── 7. Categoriser ──────────────────────────────────────────────────
    cells.append(md(
        "## 7. Thematic categorisation — `FCM/categorizer.py`\n"
        "\n"
        "Implements the FEP (Fishery Ecosystem Plan) colour scheme and the "
        "`SummaryTheme` rules used to collapse fine-grained nodes into "
        "policy-relevant super-nodes (e.g. *Storms and Flooding*, "
        "*Water Quality and Pollution*). Mirrors the original R "
        "`case_when` semantics so the Python output stays consistent with "
        "the upstream R pipeline.\n"
    ))
    cells.append(md("> **Source — `FCM/categorizer.py`**"))
    cells.append(code(_read("FCM/categorizer.py")))

    # ── 8. Simulator ────────────────────────────────────────────────────
    cells.append(md(
        "## 8. Scenario simulation — `FCM/simulator.py`\n"
        "\n"
        "Implements Kosko's propagation rule:\n"
        "\n"
        "$$ a(t+1) = f\\big(a(t) + A^{\\!\\top} a(t)\\big) $$\n"
        "\n"
        "where $A$ is the signed adjacency matrix, $a(t)$ is the activation "
        "vector at iteration $t$, and $f(\\cdot)$ is a squashing function "
        "(sigmoid, tanh, or identity). Driver concepts are clamped to their "
        "initial value at every step so the resulting equilibrium answers the "
        "question *“what does the system look like if X stays elevated?”*.\n"
    ))
    cells.append(md("> **Source — `FCM/simulator.py`**"))
    cells.append(code(_read("FCM/simulator.py")))

    # ── 9. Demo ─────────────────────────────────────────────────────────
    cells.append(md(
        "## 9. End-to-end demo (runnable)\n"
        "\n"
        "All cells above are *source*. The cells below actually exercise "
        "the pipeline on a small synthetic text so reviewers can confirm "
        "what comes out of each stage. Run them top-to-bottom from the repo "
        "root with the project venv active.\n"
    ))
    cells.append(code(
        "import sys, pathlib\n"
        "# Make sure the repo root is on sys.path when running this notebook in place\n"
        "ROOT = pathlib.Path.cwd()\n"
        "while ROOT != ROOT.parent and not (ROOT / 'FCM').is_dir():\n"
        "    ROOT = ROOT.parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "print('Repo root:', ROOT)\n"
    ))
    cells.append(code(
        "from FCM.extractor import CausalExtractor, normalize_concept, _is_clean_concept\n"
        "from FCM.fcm_graph import FCMGraphBuilder\n"
        "from FCM.simulator import FCMSimulator\n"
        "from FCM.categorizer import get_category, get_summary_theme, get_color\n"
    ))
    cells.append(md(
        "### 9.1  Concept-name filters in action\n"
        "Verify that the keyword filter rejects clause fragments, "
        "interrogatives, and useless modifiers while keeping legitimate "
        "domain phrases.\n"
    ))
    cells.append(code(
        "samples = [\n"
        "    'the very high fish stock',   # → Fish Stock\n"
        "    'fish stock high',            # → Fish Stock (trailing modifier stripped)\n"
        "    'How fish increase',          # → Fish Increase (leading interrogative stripped)\n"
        "    'fish how grow',              # rejected (interrogative anywhere)\n"
        "    'it is overfishing',          # → Overfishing\n"
        "    'very high',                  # rejected (all modifiers)\n"
        "    'fish fish stock',            # dedupe → Fish Stock\n"
        "    'CPUE',                       # acronym preserved\n"
        "    'water quality',              # legitimate\n"
        "    'thing',                      # rejected (vague single)\n"
        "    'climate change',             # legitimate\n"
        "]\n"
        "print(f\"{'INPUT':<32} {'NORMALIZED':<28} CLEAN?\")\n"
        "print('-' * 72)\n"
        "for s in samples:\n"
        "    n = normalize_concept(s)\n"
        "    ok = _is_clean_concept(n) if n else False\n"
        "    print(f'{s!r:<32} {n!r:<28} {ok}')\n"
    ))
    cells.append(md(
        "### 9.2  Run the extractor on a small Gulf-FEI vignette\n"
    ))
    cells.append(code(
        "vignette = (\n"
        "    'Overfishing reduces fish biomass. '\n"
        "    'Habitat loss threatens marine biodiversity. '\n"
        "    'Climate change increases ocean temperature. '\n"
        "    'Rising ocean temperature drives coral bleaching. '\n"
        "    'Coral bleaching damages reef habitat. '\n"
        "    'Strong fishery management restores fish biomass.'\n"
        ")\n"
        "extractor = CausalExtractor()\n"
        "concepts, edges = extractor.extract([{'text': vignette, 'score': 1.0, 'doc_id': 'vignette'}])\n"
        "\n"
        "print(f'Concepts ({len(concepts)}):')\n"
        "for c in concepts:\n"
        "    print(f'  - {c}  [{get_category(c)}]')\n"
        "print()\n"
        "print(f'Edges ({len(edges)}):')\n"
        "for e in edges:\n"
        "    sign = \"+\" if e.weight >= 0 else \"-\"\n"
        "    print(f'  {e.source:<28} -[{sign}{abs(e.weight):.2f}]-> {e.target:<28} ({e.edge_type})')\n"
    ))
    cells.append(md(
        "### 9.3  Build the `FCMMap` and inspect its adjacency matrix\n"
    ))
    cells.append(code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "fcm_map = FCMGraphBuilder().build('demo query', concepts, edges)\n"
        "matrix_df = pd.DataFrame(\n"
        "    fcm_map.adjacency_matrix,\n"
        "    index=fcm_map.concepts,\n"
        "    columns=fcm_map.concepts,\n"
        ").round(2)\n"
        "matrix_df\n"
    ))
    cells.append(md(
        "### 9.4  Visualise the graph\n"
        "We use the same publication renderer the web app calls.\n"
    ))
    cells.append(code(
        "from src.FCM import FCMGenerator\n"
        "from IPython.display import Image\n"
        "gen = FCMGenerator(max_concepts=15, edge_threshold=0.15, directed=True)\n"
        "G, details = gen.build_fcm(vignette, query='Demo: Gulf FEI vignette', num_results=20)\n"
        "png_url = gen.plot_fcm(G, filename='walkthrough_demo', query='Demo: Gulf FEI vignette')\n"
        "print('Plot saved at (app-relative):', png_url)\n"
        "Image(filename=str(ROOT / png_url.lstrip('/')))\n"
    ))
    cells.append(md(
        "### 9.5  Run a what-if scenario\n"
        "Clamp `Overfishing` high and `Fishery Management` high — see which "
        "concepts the system pushes up or down at equilibrium.\n"
    ))
    cells.append(code(
        "sim = FCMSimulator(fcm_map.concepts, fcm_map.adjacency_matrix)\n"
        "drivers = {}\n"
        "for c in fcm_map.concepts:\n"
        "    if 'Overfishing' in c:        drivers[c] = +1.0\n"
        "    if 'Management' in c:         drivers[c] = +1.0\n"
        "if not drivers:\n"
        "    drivers[fcm_map.concepts[0]] = +1.0   # fallback so the cell still runs\n"
        "result = sim.simulate(drivers, steps=40, squash='tanh', clamp_drivers=True)\n"
        "print('Converged:', result['converged'], '| iterations:', result['iterations'])\n"
        "print()\n"
        "print('Top influenced concepts:')\n"
        "for row in result['top_effects'][:8]:\n"
        "    sign = '+' if row['influence'] >= 0 else '−'\n"
        "    print(f\"  {row['concept']:<28}  final={row['final']:+.3f}  Δ={sign}{abs(row['influence']):.3f}\")\n"
    ))

    # ── 10. Quality controls ────────────────────────────────────────────
    cells.append(md(
        "## 10. Quality controls and validation\n"
        "\n"
        "Several guards keep the keyword set and graph honest:\n"
        "\n"
        "| Guard | Where | What it does |\n"
        "|---|---|---|\n"
        "| Query meaningfulness | `main._is_meaningful_query` | Rejects empty / pure-stopword / interrogative-only queries before the RAG + FCM pipeline runs. |\n"
        "| Concept normalisation | `extractor.normalize_concept` | Strips articles, leading/trailing useless modifiers, dedupes repeated tokens, preserves acronyms. |\n"
        "| Concept cleanliness | `extractor._is_clean_concept` | Rejects clause fragments, interrogatives anywhere in the span, all-modifier phrases, sub-2-letter tokens, vague singles, and >50% stopword spans. |\n"
        "| Polarity warnings | `main._build_response` | Flags maps that contain only positive (or only negative) edges so reviewers can sanity-check the source text. |\n"
        "| Bidirectional dedup | `extractor._dedupe_bidirectional` | Drops same-sign reverse pairs (A→B and B→A both positive) keeping the stronger direction; preserves opposing-sign pairs as legitimate feedback loops. |\n"
        "| Confidence-weighted merge | `extractor.fuzzy_merge_concepts` | Collapses near-duplicate concepts (plurals, fuzzy matches) into the longest canonical name and merges their edges by confidence-weighted mean. |\n"
        "| 2:1 polarity interleave | `extractor._interleave_polarity` | Re-ranks final edges so positive and negative findings stay visible in the top-N. |\n"
        "\n"
        "### Reproducing this notebook\n"
        "Re-run `python -m FCM.docs._build_notebook` from the repo root after "
        "any change to the FCM modules. The script reads each module from "
        "disk and rebuilds the cells, so the notebook never drifts from the "
        "production code.\n"
    ))

    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (Gulf FEI venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    }
    return nb


def main() -> None:
    nb = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT_PATH)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
