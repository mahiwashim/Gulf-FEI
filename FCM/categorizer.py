from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Thematic FEP colour scheme (exact hex values from client R script)
FEP_COLORS: Dict[str, str] = {
    "Habitat":             "#228B22",
    "Fishery":             "#1E90FF",
    "Biotic":              "#32CD32",
    "Reef Fish":           "#FFD700",
    "Structures":          "#808080",
    "Environmental":       "#FF8C00",
    "Social and Economic": "#9370DB",
    "Management":          "#E6194B",
    "Science and Data":    "#8B4513",
    "Land-Based Drivers":  "#A52A2A",
    "Other":               "#D3D3D3",
}

# Ordered category rules — first match wins (mirrors R case_when semantics)
_CATEGORY_RULES: List[Tuple[str, str]] = [
    ("Habitat",             r"habitat|estuary|marsh|grass|nursery|restoration|dredging|diversion"),
    ("Fishery",             r"fisher|fishery|effort|harvest|catch|discard|mortality|ifq|quota|import|bycatch|menhaden|shrimp|oyster"),
    ("Reef Fish",           r"snapper|amberjack|trigger|reef\s*fish|spawning"),
    ("Biotic",              r"biotic|biodiversity|shark|dolphin|mammal|turtle|sturgeon|manatee|birds|trophic|bait|crab"),
    ("Structures",          r"reef|rig|platform|structure|windfarm|concrete|infrastructure"),
    ("Environmental",       r"climate|temp|salinity|oxygen|hypoxia|acidification|weather|storm|hurricane|sea-level|tide"),
    ("Social and Economic", r"social|economic|well-being|price|cost|demand|tourism|angler|fleet|access|perception|jobs|fuel|aging|fleets"),
    ("Management",          r"management|council|noaa|regulation|season|limit|enforcement|efp|mmpa|authority|accountability"),
    ("Science and Data",    r"knowledge|science|research|monitoring|uncertainty|engagement|uptake|assessment"),
    ("Land-Based Drivers",  r"agriculture|flooding|runoff|nutrient|sewage|pollution|spillway|river|watershed"),
]

# Summary-theme rules: collapse related concepts into a thematic super-node
_SUMMARY_RULES: List[Tuple[str, str]] = [
    ("Storms and Flooding",          r"storm|flood|hurricane|surge|rainfall|weather changes"),
    ("Dredging and Development",     r"dredging|dredge|development|construction|infrastructure|channelization|wakes|navigation"),
    ("Oil and Gas Industry",         r"oil|gas|rig|platform|lng|p&a|decommissioned|exploration|seismic"),
    ("Water Quality and Pollution",  r"water quality|pollution|nutrient|runoff|sewage|spillway|hypoxia|debris|dumping|trash|pesticides"),
    ("Fishery Management and Rules", r"management|regulation|quota|limit|enforcement|accountability|permit|reporting|compliance|hcr"),
    ("Discard and Release Mortality",r"discard|mortality|descending|barotrauma|release|culling|highgrading|post-release"),
    ("Seafood Markets and Trade",    r"price|demand|import|trade|ex-vessel|market|deficit"),
    ("Recreational Fishing Access",  r"access|angler|satisfaction|license"),
    ("Coastal and Marine Habitat",   r"habitat|nursery|estuar|marsh|grass|oyster|benthic|coral|reef fish habitat"),
    ("Marine Mammals and Predators", r"mammal|dolphin|manatee|shark|depredation|turtle|bird|pelican"),
    ("Reef Fish Species",            r"snapper|amberjack|trigger|reef fish|redfish|red drum|mackerel|grouper"),
    ("Bait and Forage",              r"bait|menhaden|poggy|anchovy|hard tail|forage|crabs|shrimp"),
    ("Climate and Physical Trends",  r"climate|sea-level|temp|salinity|current|trends|winter|subsidence|acidification|warming"),
    ("Social and Economic",          r"social|economic|well-being|community|jobs|cost|perception|h2b|aging out"),
]

_CATEGORY_COMPILED = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in _CATEGORY_RULES]
_SUMMARY_COMPILED  = [(label, re.compile(pat, re.IGNORECASE)) for label, pat in _SUMMARY_RULES]


def get_category(name: str) -> str:
    if not name:
        return "Other"
    for label, rx in _CATEGORY_COMPILED:
        if rx.search(name):
            return label
    return "Other"


def get_summary_theme(name: str) -> str:
    """Return the summary theme for a node, or the node name itself if no rule matches."""
    if not name:
        return ""
    for label, rx in _SUMMARY_COMPILED:
        if rx.search(name):
            return label
    return name


def get_color(name: str) -> str:
    return FEP_COLORS[get_category(name)]


def build_node_lookup(nodes: List[str]) -> List[Dict[str, str]]:
    """Produce one lookup entry per node (mirrors the R `node_lookup` data-frame)."""
    return [
        {
            "OriginalNode": n,
            "Category":     get_category(n),
            "SummaryNode":  get_summary_theme(n),
            "HexColor":     get_color(n),
        }
        for n in nodes
    ]
