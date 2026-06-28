from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Tuple

from .schemas import Edge

# Domain acronyms that must never be title-cased or lemmatized (e.g. "Cpue" or "Cpu")
_ACRONYMS = {
    'CPUE', 'MSY', 'EBM', 'EBFM', 'IUU', 'TAC', 'NOAA', 'NMFS', 'GOM',
    'FEP', 'MPA', 'GDP', 'SST', 'DO', 'ENSO', 'DOC', 'EPA', 'USCG',
    'ITQ', 'ACL', 'ABC', 'OFL', 'MRIP', 'SEAMAP', 'HAB', 'HABs',
}

# ---------------------------------------------------------------------------
# Pattern templates
#
#   _FIRST  – non-greedy: matches the concept BEFORE the causal verb.
#             Non-greedy ensures it does not consume the verb itself.
#
#   _LAST   – greedy up to 4 words, stops before common conjunctions/
#             relative pronouns so the effect phrase doesn't bleed into the
#             next clause ("fish stock abundance" not "fish stock abundance
#             and threatens marine biodiversity").
#
# Both templates accept {g} as the regex group name; all other braces that
# would confuse str.format() are doubled.
# ---------------------------------------------------------------------------

_STOP_CONJ = (
    # Conjunctions & relative pronouns
    r'and|or|but|which|that|because|since|while|however|'
    r'although|if|when|where|as|so|yet|nor|'
    # Prepositions that end a noun phrase
    r'of|in|on|at|by|for|with|from|into|onto|upon|via|'
    # Causal markers (stop before "due to", "because of", etc.)
    r'due|owing|through|throughout|'
    # Spatial / scope prepositions — stop the greedy capture before words
    # like "Marine Biodiversity Across Reef" leak in
    r'across|among|amongst|between|within|without|against|around|'
    r'amid|amidst|beyond|behind|beside|before|after|during'
)

# Single word: letters, digits, hyphens, slashes (NO spaces — keeps word boundaries clean)
_W = r'[A-Za-z][A-Za-z0-9\-/]*'

# Cause / driver: 1-4 words, non-greedy (stops as early as possible before the verb)
_FIRST = (
    r'(?P<{g}>'
    + _W +
    r'(?:\s+' + _W + r'){{0,3}}?'
    r')'
)

# Effect / outcome: 1-4 words, greedy, stops before conjunctions / punctuation
_LAST = (
    r'(?P<{g}>'
    + _W +
    r'(?:\s+(?!\b(?:' + _STOP_CONJ + r')\b)' + _W + r'){{0,3}}'
    r')'
)

# Convenience aliases (active patterns: A verb B)
_A = _FIRST
_B = _LAST

# If a concept's FIRST word is one of these, it is a captured predicate, not a noun phrase.
# Words that are never the head of a proper noun phrase in this domain.
_VERB_FIRST_WORDS = {
    # Base forms — every verb that appears in CAUSAL_PATTERNS must be here so a
    # single-word capture of the verb gets rejected as a clause fragment.
    'increases', 'increase', 'reduces', 'reduce', 'causes', 'cause',
    'decreases', 'decrease', 'raises', 'raise', 'lowers', 'lower',
    'boosts', 'boost', 'improves', 'improve', 'strengthens', 'strengthen',
    'weakens', 'weaken', 'depletes', 'deplete', 'degrades', 'degrade',
    'drives', 'drive', 'leads', 'lead', 'affects', 'affect',
    'threatens', 'threaten', 'promotes', 'promote', 'damages', 'damage',
    'prevents', 'prevent', 'limits', 'limit', 'enhances', 'enhance',
    'inhibits', 'inhibit', 'produces', 'produce', 'creates', 'create',
    'generates', 'generate', 'contributes', 'contribute',
    'triggers', 'trigger', 'impacts', 'impact',
    'harms', 'harm', 'destroys', 'destroy', 'suppresses', 'suppress',
    'diminishes', 'diminish', 'impairs', 'impair', 'undermines', 'undermine',
    'hurts', 'hurt', 'minimizes', 'minimize', 'maximizes', 'maximize',
    'disrupts', 'disrupt', 'disturbs', 'disturb', 'erodes', 'erode',
    'worsens', 'worsen', 'aggravates', 'aggravate', 'exacerbates', 'exacerbate',
    'endangers', 'endanger', 'jeopardizes', 'jeopardize',
    'kills', 'kill', 'eliminates', 'eliminate',
    'stops', 'stop', 'halts', 'halt', 'curbs', 'curb',
    'controls', 'control', 'constrains', 'constrain', 'restricts', 'restrict',
    'blocks', 'block', 'alleviates', 'alleviate',
    'facilitates', 'facilitate', 'stimulates', 'stimulate',
    'expands', 'expand', 'accelerates', 'accelerate',
    'elevates', 'elevate', 'amplifies', 'amplify', 'supports', 'support',
    'influences', 'influence', 'alters', 'alter', 'changes', 'change',
    'shapes', 'shape', 'modifies', 'modify',
    'determines', 'determine', 'governs', 'govern',
    'induces', 'induce',
    # Past tense / gerund forms that sneak through as nouns
    'caused', 'causing', 'increased', 'increasing', 'decreased', 'decreasing',
    'reduced', 'reducing', 'driven', 'driving', 'leading', 'affecting',
    'threatened', 'threatening', 'promoted', 'promoting', 'damaged', 'damaging',
    'prevented', 'preventing', 'limited', 'limiting', 'enhanced', 'enhancing',
    'inhibited', 'inhibiting', 'produced', 'producing', 'created', 'creating',
    'generated', 'generating', 'contributed', 'contributing',
    'triggered', 'triggering', 'impacted', 'impacting',
    'mitigates', 'mitigate', 'mitigated', 'mitigating',
    'results', 'result', 'resulted', 'resulting',
    'raised', 'raising', 'lowered', 'lowering',
    'boosted', 'boosting', 'improved', 'improving',
    'strengthened', 'strengthening', 'weakened', 'weakening',
    'depleted', 'depleting', 'degraded', 'degrading',
    'harmed', 'harming', 'destroyed', 'destroying',
    'suppressed', 'suppressing', 'diminished', 'diminishing',
    'impaired', 'impairing', 'undermined', 'undermining',
    'hurted', 'hurting', 'minimized', 'minimizing',
    'maximized', 'maximizing',
    'disrupted', 'disrupting', 'disturbed', 'disturbing',
    'eroded', 'eroding', 'worsened', 'worsening',
    'aggravated', 'aggravating', 'exacerbated', 'exacerbating',
    'endangered', 'endangering', 'jeopardized', 'jeopardizing',
    'killed', 'killing', 'eliminated', 'eliminating',
    'stopped', 'stopping', 'halted', 'halting', 'curbed', 'curbing',
    'controlled', 'controlling', 'constrained', 'constraining',
    'restricted', 'restricting', 'blocked', 'blocking',
    'alleviated', 'alleviating',
    'facilitated', 'facilitating', 'stimulated', 'stimulating',
    'expanded', 'expanding', 'accelerated', 'accelerating',
    'elevated', 'elevating', 'amplified', 'amplifying',
    'supported', 'supporting', 'influenced', 'influencing',
    'altered', 'altering', 'changed', 'changing',
    'shaped', 'shaping', 'modified', 'modifying',
    'determined', 'determining', 'governed', 'governing',
    'induced', 'inducing',
    # Intent / purpose verbs — clause markers, never noun-phrase content
    # ("aims to enhance Y" → cause was being captured as "Aims")
    'aims', 'aim', 'aimed', 'aiming',
    'seeks', 'seek', 'sought', 'seeking',
    'intends', 'intend', 'intended', 'intending',
    'attempts', 'attempt', 'attempted', 'attempting',
    'tries', 'try', 'tried', 'trying',
    'proposes', 'propose', 'proposed', 'proposing',
    'wishes', 'wish', 'wished', 'wishing',
    'hopes', 'hope', 'hoped', 'hoping',
    'wants', 'want', 'wanted', 'wanting',
    'expects', 'expect', 'expected', 'expecting',
    'needs', 'need', 'needed', 'needing',
    'requires', 'require', 'required', 'requiring',
    # Generic copular / existential verb forms (sometimes survive normalization)
    'becomes', 'become', 'became', 'becoming',
    'remains', 'remain', 'remained', 'remaining',
    'stays', 'stay', 'stayed', 'staying',
    # Generic action / report verbs that produce phrases like
    # "Implemented Various Measures" — meta-language, not domain content.
    'implements', 'implement', 'implemented', 'implementing',
    'applies', 'apply', 'applied', 'applying',
    'uses', 'use', 'used', 'using',
    'employs', 'employ', 'employed', 'employing',
    'adopts', 'adopt', 'adopted', 'adopting',
    'establishes', 'establish', 'established', 'establishing',
    'develops', 'develop', 'developed', 'developing',
    'designs', 'design', 'designed', 'designing',
    'considers', 'consider', 'considered', 'considering',
    'identifies', 'identify', 'identified', 'identifying',
    'observes', 'observe', 'observed', 'observing',
    'reports', 'report', 'reported', 'reporting',
    'suggests', 'suggest', 'suggested', 'suggesting',
    'recommends', 'recommend', 'recommended', 'recommending',
    'evaluates', 'evaluate', 'evaluated', 'evaluating',
    'assesses', 'assess', 'assessed', 'assessing',
    'reviews', 'review', 'reviewed', 'reviewing',
    'analyzes', 'analyze', 'analyzed', 'analyzing',
    'analyses', 'analyse', 'analysed', 'analysing',
    'examines', 'examine', 'examined', 'examining',
    'investigates', 'investigate', 'investigated', 'investigating',
    'studies', 'study', 'studied', 'studying',
    'addresses', 'address', 'addressed', 'addressing',
    'discusses', 'discuss', 'discussed', 'discussing',
    'highlights', 'highlight', 'highlighted', 'highlighting',
    'emphasizes', 'emphasize', 'emphasized', 'emphasizing',
    'argues', 'argue', 'argued', 'arguing',
    'concludes', 'conclude', 'concluded', 'concluding',
    'finds', 'find', 'found', 'finding',
    'shows', 'show', 'showed', 'shown', 'showing',
    'demonstrates', 'demonstrate', 'demonstrated', 'demonstrating',
    'indicates', 'indicate', 'indicated', 'indicating',
    'notes', 'note', 'noted', 'noting',
    'describes', 'describe', 'described', 'describing',
    'explains', 'explain', 'explained', 'explaining',
    'states', 'state', 'stated', 'stating',
    'claims', 'claim', 'claimed', 'claiming',
    'mentions', 'mention', 'mentioned', 'mentioning',
    'presents', 'present', 'presented', 'presenting',
    'introduces', 'introduce', 'introduced', 'introducing',
    'provides', 'provide', 'provided', 'providing',
    'offers', 'offer', 'offered', 'offering',
    'gives', 'give', 'gave', 'giving', 'given',
    'takes', 'take', 'took', 'taking', 'taken',
    'makes', 'make', 'made', 'making',
    'gets', 'get', 'got', 'getting',
    'sets', 'set', 'setting',
    'puts', 'put', 'putting',
    'lets', 'let', 'letting',
    'allows', 'allow', 'allowed', 'allowing',
    'enables', 'enable', 'enabled', 'enabling',
    'helps', 'help', 'helped', 'helping',
    'works', 'work', 'worked', 'working',
    'runs', 'run', 'ran', 'running',
    'goes', 'go', 'went', 'gone', 'going',
    'comes', 'come', 'came', 'coming',
    'sees', 'see', 'saw', 'seen', 'seeing',
    'knows', 'know', 'knew', 'known', 'knowing',
    'thinks', 'think', 'thought', 'thinking',
    'looks', 'look', 'looked', 'looking',
    'means', 'meant', 'meaning',
    # Question/interrogative starts — never valid FCM concepts
    'how', 'what', 'why', 'when', 'where', 'does', 'did', 'do',
    'is', 'are', 'was', 'were', 'can', 'could', 'should', 'would',
    # Pronouns / determiners — never a concept head
    'it', 'its', 'this', 'that', 'these', 'those', 'they', 'their',
    'he', 'she', 'we', 'you', 'i', 'such', 'both', 'each', 'all',
    # Prepositions / conjunctions that regex can grab as "first word"
    'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'into', 'onto',
    'via', 'due', 'as', 'or', 'and', 'but', 'not', 'also', 'more', 'most',
}

# Words that, when found ANYWHERE in the concept, signal a predicate fragment
_CLAUSE_VERBS = {
    'associated', 'linked', 'correlated', 'determines', 'mitigates', 'alleviates',
    'damages', 'threatens', 'reduces', 'promotes', 'prevents', 'inhibits',
}

# Trailing words that make a concept fragment — strip these from the end
_TRAILING_STRIP = re.compile(
    r'\s+\b(?:of|in|on|at|by|for|with|from|into|onto|to|the|a|an|'
    r'and|or|but|which|that|this|these|those|its|their|'
    # Time / sequence prepositions ("Data Availability After")
    r'after|before|during|while|since|until|till|once|'
    # Causal / comparison markers ("Participation Due")
    r'due|because|as|than|then|though|although|whereas|'
    # Spatial / scope prepositions
    r'through|throughout|via|against|around|among|amongst|between|'
    r'within|without|across|along|behind|below|above|beneath|beside|'
    r'near|off|out|up|down|over|under|inside|outside|toward|towards|'
    # Lone modifiers / adverbs at the tail
    r'so|yet|also|too|even|still|just|only|more|most|less|least)\b\.?$',
    re.I,
)

# Single-word concepts that are too vague to carry FCM meaning
_VAGUE_SINGLES = {
    'level', 'levels', 'rate', 'rates', 'amount', 'amounts', 'number', 'numbers',
    'factor', 'factors', 'effect', 'effects', 'impact', 'impacts', 'change', 'changes',
    'increase', 'decrease', 'result', 'results', 'role', 'part', 'aspect', 'issue',
    'area', 'areas', 'type', 'types', 'form', 'forms', 'way', 'ways', 'case', 'cases',
    'process', 'processes', 'system', 'systems', 'term', 'terms', 'use', 'used',
    'value', 'values', 'range', 'basis', 'data', 'point', 'points', 'need', 'needs',
    'example', 'examples', 'situation', 'situations', 'condition', 'conditions',
    'period', 'time', 'year', 'years', 'month', 'months', 'day', 'days',
    # Generic placeholder nouns that carry no domain content
    'thing', 'things', 'stuff', 'matter', 'matters', 'kind', 'kinds',
    'sort', 'sorts', 'lot', 'lots', 'set', 'sets', 'group', 'groups',
    'side', 'sides', 'end', 'ends', 'start', 'starts', 'beginning',
    'place', 'places', 'person', 'people', 'someone', 'anyone', 'everyone',
    'item', 'items', 'object', 'objects', 'section', 'sections',
    'chapter', 'chapters', 'page', 'pages', 'version', 'versions',
    'name', 'names', 'word', 'words', 'idea', 'ideas', 'concept', 'concepts',
    'topic', 'topics', 'subject', 'subjects', 'detail', 'details',
    'information', 'info', 'content', 'overview', 'summary', 'background',
    'yes', 'no', 'maybe', 'ok', 'okay',
    # Conversational/meta filler that occasionally slips out of the regex
    'note', 'notes', 'question', 'questions', 'answer', 'answers',
    'thanks', 'please', 'sure', 'fine', 'good', 'bad',
    # Pure quantity / measurement words — meaningless without context
    'percentage', 'percentages', 'percent', 'share', 'shares',
    'ratio', 'ratios', 'fraction', 'fractions',
    'proportion', 'proportions', 'count', 'counts',
    'total', 'totals', 'sum', 'sums', 'average', 'averages',
    'mean', 'means', 'median', 'medians',
    'min', 'minimum', 'minimums', 'max', 'maximum', 'maximums',
    'size', 'sizes', 'length', 'lengths', 'height', 'heights',
    'width', 'widths', 'depth', 'depths', 'weight', 'weights',
    'quantity', 'quantities', 'measure', 'measures',
    # Bare adjectival qualifiers — too vague to stand as a concept on their
    # own ("Marine Biodiversity" is fine; "Marine" alone is a qualifier).
    'marine', 'coastal', 'oceanic', 'aquatic', 'terrestrial',
    'ecological', 'biological', 'environmental', 'physical', 'chemical',
    'natural', 'human', 'social', 'economic', 'cultural', 'political',
    'global', 'local', 'regional', 'national', 'international',
    'public', 'private', 'commercial', 'recreational', 'industrial',
    # Domain words that demand a qualifier to mean anything FCM-actionable
    # ("Climate Change" not "Climate"; "Marine Ecosystem" not "Ecosystem")
    'climate', 'ecosystem', 'ecosystems', 'species', 'catch', 'catches',
    'effort', 'efforts', 'availability', 'participation',
    # Generic verbal nouns / outcome words that are vague without a subject
    'help', 'helps', 'support',
    'recovery', 'recoveries', 'loss', 'losses',
    'decline', 'declines', 'growth', 'growths',
    'reduction', 'reductions', 'improvement', 'improvements',
    'depletion', 'depletions', 'restoration', 'restorations',
    # Abstract academic / report nouns — pure meta-vocabulary, never an FCM concept
    'finding', 'findings', 'outcome', 'outcomes', 'output', 'outputs',
    'development', 'developments', 'progress', 'progression',
    'aspect', 'aspects', 'feature', 'features', 'characteristic', 'characteristics',
    'attribute', 'attributes', 'property', 'properties',
    'implication', 'implications', 'consequence', 'consequences',
    'conclusion', 'conclusions', 'inference', 'inferences',
    'analysis', 'analyses', 'approach', 'approaches', 'method', 'methods',
    'methodology', 'methodologies', 'technique', 'techniques',
    'option', 'options', 'choice', 'choices', 'alternative', 'alternatives',
    'guideline', 'guidelines', 'principle', 'principles',
    'consideration', 'considerations', 'discussion', 'discussions',
    'review', 'reviews', 'evaluation', 'evaluations',
    'recommendation', 'recommendations', 'suggestion', 'suggestions',
    'observation', 'observations', 'perspective', 'perspectives',
    'approach', 'approaches', 'framework', 'frameworks',
    'context', 'contexts', 'scope', 'scopes',
    'objective', 'objectives', 'goal', 'goals', 'target', 'targets',
    'purpose', 'purposes', 'mission', 'missions',
}


# ---------------------------------------------------------------------------
# Causal pattern registry
# Each entry: (compiled_regex, base_weight, polarity_label)
# ---------------------------------------------------------------------------

def _pat(pattern_str: str) -> re.Pattern:
    return re.compile(pattern_str, re.I)


CAUSAL_PATTERNS: List[Tuple[re.Pattern, float, str]] = [

    # ── Strong positive: X increases / boosts / enhances Y ─────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:significantly\s+|substantially\s+|greatly\s+)?'
            r'(?:increases?|raises?|boosts?|improves?|strengthens?|enhances?|promotes?|'
            r'facilitates?|stimulates?|expands?|accelerates?|drives?|elevates?|amplifies?|'
            r'maximizes?|supports?)\s+' +
            _B.format(g='b')
        ), 0.75, 'positive',
    ),

    # ── Strong negative: X reduces / threatens / damages Y ──────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:significantly\s+|substantially\s+|greatly\s+)?'
            r'(?:decreases?|reduces?|weakens?|lowers?|depletes?|degrades?|'
            r'damages?|harms?|threatens?|destroys?|inhibits?|suppresses?|'
            r'diminishes?|impairs?|undermines?|hurts?|minimizes?|'
            r'disrupts?|disturbs?|erodes?|worsens?|aggravates?|exacerbates?|'
            r'endangers?|jeopardizes?|disrupted?|kills?|eliminates?|'
            r'conflicts?\s+with|interferes?\s+with|competes?\s+with)\s+' +
            _B.format(g='b')
        ), -0.75, 'negative',
    ),

    # ── "negatively / adversely affects|impacts|influences" ─────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:negatively|adversely|detrimentally|harmfully)\s+'
            r'(?:affects?|impacts?|influences?|alters?|changes?|shapes?|'
            r'modifies?|correlates?|relates?)\s+(?:with\s+|to\s+)?' +
            _B.format(g='b')
        ), -0.70, 'negative',
    ),

    # ── "X causes loss / decline / decrease in Y" ──────────────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:causes?|drives?|leads?\s+to|results?\s+in|produces?|triggers?)\s+'
            r'(?:a\s+|the\s+)?(?:loss|decline|decrease|drop|fall|reduction|'
            r'depletion|degradation|deterioration|collapse|extinction|'
            r'destruction|disruption)\s+(?:of|in)\s+' +
            _B.format(g='b')
        ), -0.70, 'negative',
    ),

    # ── Direct causation: X leads to / causes / triggers Y ─────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:leads?\s+to|results?\s+in|causes?|triggers?|induces?|generates?|'
            r'produces?|creates?|contributes?\s+to|gives?\s+rise\s+to|brings?\s+about)\s+' +
            _B.format(g='b')
        ), 0.65, 'positive',
    ),

    # ── Negative control: X prevents / limits Y ─────────────────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:prevents?|limits?|constrains?|restricts?|blocks?|'
            r'mitigates?|alleviates?|controls?|curbs?|halts?|stops?)\s+' +
            _B.format(g='b')
        ), -0.65, 'negative',
    ),

    # ── Affect / influence (direction inferred as positive, moderate weight) ─
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:affects?|impacts?|influences?|alters?|changes?|shapes?|'
            r'modifies?|determines?|governs?|drives?)\s+' +
            _B.format(g='b')
        ), 0.55, 'positive',
    ),

    # ── Association / linkage ───────────────────────────────────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:is\s+|are\s+)?(?:positively\s+)?'
            r'(?:associated\s+with|linked\s+to|correlated\s+with|'
            r'connected\s+to|related\s+to)\s+' +
            _B.format(g='b')
        ), 0.50, 'positive',
    ),

    # ── Negative association ────────────────────────────────────────────────
    (
        _pat(
            _A.format(g='a') +
            r'\s+(?:is\s+|are\s+)?negatively\s+'
            r'(?:associated\s+with|linked\s+to|correlated\s+with)\s+' +
            _B.format(g='b')
        ), -0.50, 'negative',
    ),

    # ── Passive positive: "B is increased/driven by A"  →  A causes B ───────
    (
        _pat(
            _FIRST.format(g='b') +
            r'\s+(?:is|are|was|were)\s+'
            r'(?:increased|driven|caused|triggered|induced|enhanced|'
            r'boosted|promoted|generated|elevated)\s+by\s+' +
            _LAST.format(g='a')
        ), 0.65, 'positive',
    ),

    # ── Passive negative: "B is reduced/threatened by A"  →  A harms B ─────
    (
        _pat(
            _FIRST.format(g='b') +
            r'\s+(?:is|are|was|were)\s+'
            r'(?:reduced|depleted|degraded|damaged|threatened|harmed|'
            r'decreased|limited|impaired|suppressed|undermined)\s+by\s+' +
            _LAST.format(g='a')
        ), -0.65, 'negative',
    ),

    # ── "B due to / because of / as a result of A"  →  A causes B ──────────
    (
        _pat(
            _FIRST.format(g='b') +
            r'\s+(?:due\s+to|because\s+of|as\s+a\s+result\s+of|'
            r'owing\s+to|in\s+response\s+to)\s+' +
            _LAST.format(g='a')
        ), 0.60, 'positive',
    ),
]

# ---------------------------------------------------------------------------
# Stop-words & domain keywords
# ---------------------------------------------------------------------------

STOPWORDS = {
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'they',
    'he', 'she', 'we', 'you', 'i', 'and', 'or', 'but', 'if', 'for',
    'to', 'of', 'in', 'on', 'with', 'by', 'from', 'at', 'as', 'its',
    'their', 'our', 'has', 'have', 'had', 'be', 'been', 'being',
    'is', 'are', 'was', 'were', 'can', 'could', 'may', 'might',
    'will', 'would', 'shall', 'should', 'such', 'which', 'when',
    # Interrogatives and auxiliaries — must never appear inside a concept name
    'how', 'what', 'why', 'where', 'does', 'did', 'do', 'not',
    'also', 'both', 'each', 'more', 'most', 'other', 'some',
    'than', 'then', 'there', 'so', 'very', 'just', 'about',
    # Empty intensifiers / qualifiers that add no FCM meaning
    'much', 'many', 'few', 'little', 'lots', 'plenty',
    'really', 'quite', 'rather', 'simply', 'only', 'even', 'still', 'yet',
    'almost', 'nearly', 'maybe', 'perhaps', 'probably', 'possibly',
    'always', 'never', 'often', 'sometimes', 'usually',
    'here', 'now', 'today', 'tomorrow', 'yesterday',
}

# Words that are pure modifiers/qualifiers — strip from the start or end of a
# concept regardless of position; never count as the meaningful head of a phrase.
_USELESS_MODIFIERS = {
    'very', 'much', 'more', 'most', 'less', 'least', 'many', 'few',
    'some', 'any', 'all', 'every', 'each', 'such', 'no',
    'just', 'only', 'even', 'still', 'yet', 'also',
    'really', 'quite', 'rather', 'simply', 'almost', 'nearly',
    'maybe', 'perhaps', 'probably', 'possibly',
    'always', 'never', 'often', 'sometimes', 'usually',
    'high', 'low', 'big', 'small', 'large', 'huge', 'tiny',  # bare adjectives w/o noun
    'good', 'bad', 'better', 'worse', 'best', 'worst',
    'new', 'old', 'recent',
    # Comparatives / superlatives — pure qualifiers that carry no noun content
    'higher', 'highest', 'lower', 'lowest',
    'bigger', 'biggest', 'smaller', 'smallest',
    'larger', 'largest', 'greater', 'greatest', 'lesser',
    'stronger', 'strongest', 'weaker', 'weakest',
    'faster', 'fastest', 'slower', 'slowest',
    'longer', 'longest', 'shorter', 'shortest',
    'wider', 'widest', 'narrower', 'narrowest',
    'deeper', 'deepest', 'shallower',
    'older', 'oldest', 'newer', 'newest',
    'main', 'major', 'minor', 'primary', 'secondary',
    'overall', 'general', 'specific',
    # Plurality / count qualifiers ("Various Measures", "Several Studies")
    'various', 'several', 'numerous', 'multiple', 'countless', 'innumerable',
    'certain', 'particular', 'individual', 'single', 'whole', 'entire', 'full',
    # Generic value adjectives ("Important Issue", "Key Factor")
    'important', 'significant', 'notable', 'remarkable', 'considerable',
    'key', 'central', 'critical', 'essential', 'fundamental', 'crucial',
    'common', 'typical', 'normal', 'usual', 'standard',
    'basic', 'simple', 'complex', 'broad', 'narrow',
    'similar', 'different', 'same', 'other', 'another',
    'possible', 'available', 'potential', 'likely', 'unlikely',
    'true', 'false', 'real', 'actual', 'apparent',
    'clear', 'obvious', 'evident', 'distinct',
    'complete', 'partial', 'final', 'initial', 'first', 'last',
    'previous', 'next', 'current', 'present', 'former', 'latter',
    # Intensity / hedging adverbs — they modulate edge WEIGHT, they are never
    # part of a concept name. Stripped from concept ends so the regex can't
    # leak them in ("Red Tide Dramatically" -> "Red Tide").
    'significantly', 'substantially', 'greatly', 'dramatically', 'drastically',
    'sharply', 'strongly', 'severely', 'markedly', 'considerably', 'vastly',
    'heavily', 'rapidly', 'profoundly', 'massively', 'enormously', 'hugely',
    'seriously', 'critically', 'slightly', 'marginally', 'somewhat', 'modestly',
    'minimally', 'moderately', 'partially', 'partly', 'occasionally',
    'again', 'further', 'directly', 'indirectly', 'largely', 'mainly', 'mostly',
}

# If ANY word inside a concept matches these, the span is a clause fragment,
# not a noun phrase — reject the whole concept.
_DISALLOWED_ANYWHERE = {
    # Interrogatives anywhere
    'how', 'what', 'why', 'when', 'where', 'who', 'whom', 'whose', 'which',
    # Auxiliaries / modals anywhere — signals a clause
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'do', 'does', 'did', 'doing', 'done',
    'has', 'have', 'had', 'having',
    'can', 'could', 'shall', 'should', 'will', 'would', 'may', 'might', 'must',
    # Pronouns anywhere
    'it', 'its', 'they', 'them', 'their', 'theirs',
    'he', 'him', 'his', 'she', 'her', 'hers',
    'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'i', 'me', 'my', 'mine',
    'this', 'that', 'these', 'those',
    # Negation anywhere — hides causal sign
    'not', 'never', "n't",
    # Present-tense / 3rd-person-s causal verbs anywhere — these are clause
    # markers (e.g. "Reef Fish Populations Decreases"), never noun-phrase
    # content. Past participles and gerunds are excluded so legitimate
    # adjective-noun combos like "Reduced Mortality" can still be cleaned by
    # the leading-word rule rather than blanket-rejected.
    'increases', 'increase', 'decreases', 'decrease',
    'reduces', 'reduce', 'raises', 'raise', 'lowers', 'lower',
    'boosts', 'boost', 'improves', 'improve',
    'strengthens', 'strengthen', 'weakens', 'weaken',
    'depletes', 'deplete', 'degrades', 'degrade',
    'causes', 'cause', 'drives', 'drive', 'leads', 'lead',
    'affects', 'affect', 'threatens', 'threaten',
    'promotes', 'promote', 'damages', 'damage',
    'prevents', 'prevent', 'limits', 'limit',
    'enhances', 'enhance', 'inhibits', 'inhibit',
    'produces', 'produce', 'creates', 'create',
    'generates', 'generate', 'contributes', 'contribute',
    'triggers', 'trigger', 'impacts', 'impact',
    'harms', 'harm', 'destroys', 'destroy',
    'suppresses', 'suppress', 'diminishes', 'diminish',
    'impairs', 'impair', 'undermines', 'undermine',
    'disrupts', 'disrupt', 'erodes', 'erode',
    'worsens', 'worsen', 'aggravates', 'aggravate', 'exacerbates', 'exacerbate',
    'endangers', 'endanger', 'jeopardizes', 'jeopardize',
    'kills', 'kill', 'eliminates', 'eliminate',
    'stops', 'stop', 'halts', 'halt', 'curbs', 'curb',
    'mitigates', 'mitigate', 'alleviates', 'alleviate',
    'controls', 'control', 'restricts', 'restrict', 'blocks', 'block',
    'facilitates', 'facilitate', 'stimulates', 'stimulate',
    'expands', 'expand', 'accelerates', 'accelerate',
    'elevates', 'elevate', 'amplifies', 'amplify',
    'influences', 'influence', 'alters', 'alter',
    'modifies', 'modify', 'determines', 'determine', 'governs', 'govern',
    'induces', 'induce', 'results', 'result',
    # Intent / purpose verbs — never valid noun-phrase content
    'aims', 'aim', 'seeks', 'seek', 'intends', 'intend',
    'attempts', 'attempt', 'tries', 'try',
    'proposes', 'propose', 'wishes', 'wish',
    'hopes', 'hope', 'wants', 'want',
    'expects', 'expect', 'requires', 'require',
    # Generic action / report verbs anywhere ("Implemented Various Measures")
    'implemented', 'implementing', 'implements', 'implement',
    'applied', 'applying', 'applies', 'apply',
    'used', 'using', 'uses',  # 'use' kept out (legit noun)
    'employed', 'employing', 'employs', 'employ',
    'adopted', 'adopting', 'adopts', 'adopt',
    'established', 'establishing', 'establishes', 'establish',
    'developed', 'developing', 'develops', 'develop',
    'designed', 'designing', 'designs', 'design',
    'considered', 'considering', 'considers', 'consider',
    'identified', 'identifying', 'identifies', 'identify',
    'observed', 'observing', 'observes', 'observe',
    'reported', 'reporting', 'reports',  # 'report' kept (noun)
    'suggested', 'suggesting', 'suggests', 'suggest',
    'recommended', 'recommending', 'recommends', 'recommend',
    'evaluated', 'evaluating', 'evaluates', 'evaluate',
    'assessed', 'assessing', 'assesses',  # 'assess'/'assessment' nouny
    'reviewed', 'reviewing', 'reviews',
    'analyzed', 'analyzing', 'analyzes',
    'analysed', 'analysing', 'analyses',
    'examined', 'examining', 'examines',
    'investigated', 'investigating', 'investigates',
    'studied', 'studying', 'studies',
    'addressed', 'addressing', 'addresses',
    'discussed', 'discussing', 'discusses',
    'highlighted', 'highlighting', 'highlights',
    'emphasized', 'emphasizing', 'emphasizes',
    'argued', 'arguing', 'argues',
    'concluded', 'concluding', 'concludes',
    'found', 'showed', 'shown',
    'demonstrated', 'demonstrating', 'demonstrates',
    'indicated', 'indicating', 'indicates',
    'noted', 'described', 'describes',
    'explained', 'explaining', 'explains',
    'stated', 'stating', 'states',
    'claimed', 'claiming', 'claims',
    'mentioned', 'mentioning', 'mentions',
    'presented', 'presenting', 'presents',
    'introduced', 'introducing', 'introduces',
    'provided', 'providing', 'provides',
    'offered', 'offering', 'offers',
    'allowed', 'allowing', 'allows',
    'enabled', 'enabling', 'enables',
    'helped', 'helping', 'helps',
}

# Sentence-level polarity cues: used by the co-occurrence fallback so weak
# edges inherit the negative tone of their sentence instead of being forced
# positive by default.
_NEG_MARKERS = re.compile(
    r'\b(?:decline|declin\w+|decrease|decreas\w+|reduc\w+|loss|losses|lost|'
    r'lower|lowered|damag\w+|threat\w+|destroy\w+|degrad\w+|deplet\w+|'
    r'disrupt\w+|harm\w+|kill\w+|collapse|overfish\w+|pollut\w+|'
    r'contaminat\w+|extinct\w+|endanger\w+|deterior\w+|impair\w+|'
    r'less|fewer|suppress\w+|inhibit\w+|prevent\w+|limit\w+|restrict\w+|'
    r'conflict\w*|interfere\w*|compete[ds]?|bycatch|hypoxi\w+|'
    r'negative\w*|adverse\w*|detriment\w*|worsen\w*|aggravat\w+|exacerbat\w+|'
    r'mortality|die-off|dying|extinction|crisis|crash|crashed)\b',
    re.IGNORECASE,
)

_POS_MARKERS = re.compile(
    r'\b(?:increase\w*|grow\w+|rise|rising|rose|boost\w*|improv\w+|enhanc\w+|'
    r'strengthen\w*|recover\w+|restore\w+|restoration|sustain\w+|benefit\w*|'
    r'support\w*|promot\w+|foster\w+|facilitat\w+|protect\w+|conserv\w+|'
    r'rebuild\w*|expand\w+|thriv\w+|flourish\w+|positive\w*)\b',
    re.IGNORECASE,
)


def _sentence_polarity(sentence: str) -> int:
    """Return -1 if the sentence leans negative, +1 otherwise."""
    neg = len(_NEG_MARKERS.findall(sentence))
    pos = len(_POS_MARKERS.findall(sentence))
    return -1 if neg > pos else 1


# ── Intensity / hedging weight modulation ──────────────────────────────────
# A research-grade FCM should let edge magnitude reflect HOW strongly a
# relation is asserted. We scale a pattern's base weight up when the linking
# language is emphatic ("significantly increases") and down when it is hedged
# or tentative ("may slightly reduce"), then clamp into the open (-1, 1)
# interval so no single sentence can claim absolute certainty.
_INTENSIFIERS = re.compile(
    r'\b(?:significantly|substantially|greatly|dramatically|drastically|'
    r'sharply|strongly|severely|markedly|considerably|vastly|heavily|'
    r'rapidly|profoundly|massively|enormously|hugely|seriously|critically)\b',
    re.IGNORECASE,
)
_DOWNTONERS = re.compile(
    r'\b(?:slightly|marginally|somewhat|partly|partially|modestly|minimally|'
    r'occasionally|sometimes|may|might|could|can|possibly|potentially|'
    r'perhaps|arguably|tends?\s+to|tend\s+to|appears?\s+to|seems?\s+to|'
    r'likely|moderately|a\s+little|to\s+some\s+extent)\b',
    re.IGNORECASE,
)


def _modulate_weight(base_weight: float, connector: str) -> float:
    """Scale a pattern's base weight by the strength of the linking phrase.

    `connector` is the span of text between the cause and effect concepts
    (the verb region). Emphatic adverbs boost the magnitude; hedges shrink it.
    Sign is always preserved.
    """
    factor = 1.0
    if _INTENSIFIERS.search(connector):
        factor *= 1.18
    if _DOWNTONERS.search(connector):
        factor *= 0.62
    if factor == 1.0:
        return base_weight
    w = base_weight * factor
    if w > 0.98:
        w = 0.98
    elif w < -0.98:
        w = -0.98
    return round(w, 3)


def _evidence_confidence_boost(distinct_mentions: int) -> float:
    """Extra confidence for relations corroborated by multiple sentences.

    +0.03 per independent mention beyond the first, capped at +0.15 — so an
    edge repeated across the discourse reads as stronger community/evidence
    support than a one-off assertion, without ever reaching certainty.
    """
    return min(0.15, 0.03 * max(0, distinct_mentions - 1))


def _interleave_polarity(edges: List['Edge']) -> List['Edge']:
    """Re-rank so positive and negative edges alternate at the top.

    A credible signed FCM must not be dominated by one polarity in the
    first N shown edges. This preserves the overall ranking within each
    polarity but interleaves them 2:1 (positive-heavy) so negatives are
    visible without overwhelming the typical positive signal.
    """
    pos = [e for e in edges if e.weight >= 0]
    neg = [e for e in edges if e.weight < 0]
    if not neg or not pos:
        return edges
    out: List['Edge'] = []
    pi = ni = 0
    # 2 positive : 1 negative cadence — honest balance without flipping tone
    while pi < len(pos) or ni < len(neg):
        for _ in range(2):
            if pi < len(pos):
                out.append(pos[pi]); pi += 1
        if ni < len(neg):
            out.append(neg[ni]); ni += 1
    return out


_DOMAIN_KW = re.compile(
    r'\b(fish(?:ing|eries?|stock)?|bycatch|trawl(?:ing)?|harvest(?:ing)?|'
    r'spawn(?:ing)?|recruit(?:ment)?|biomass|abundance|mortality|overfishing|'
    r'ecosystem|habitat|coral|reef|mangrove|seagrass|tuna|shrimp|lobster|crab|'
    r'snapper|grouper|marine|ocean(?:ic)?|gulf|coastal|aquaculture|climate|'
    r'temperature|salinity|pollution|nutrient|oxygen|hypox\w+|management|'
    r'regulation|quota|effort|vessel|gear|discard|population|species|'
    r'biodiversity|conservation|sustainability|stock|assessment|CPUE|MSY|'
    r'EBM|EBFM|ecosystem.based|fishing\s+pressure|sea\s+level|water\s+quality|'
    # ── Perception / management discourse vocabulary (Gulf Council topics) ──
    r'red\s+tide|algal\s+bloom|dead\s+zone|red\s+snapper|grouper|amberjack|'
    r'menhaden|oyster|seagrass|estuary|runoff|sewage|red\s+drum|'
    r'season|closure|bag\s+limit|size\s+limit|catch\s+limit|allocation|'
    r'enforcement|permit|licen[sc]e|charter|angler|recreational|commercial|'
    r'depredation|dolphin|shark|turtle|manatee|storm|hurricane|'
    r'tourism|economy|community|access|price|market|demand|import)\b',
    re.I,
)


def _is_clean_concept(concept: str) -> bool:
    """Return False if the concept looks like a captured clause fragment."""
    if not concept:
        return False
    words = concept.lower().split()
    if len(words) > 4:
        return False
    # Must be at least 4 characters long (rejects "Up", "A", "On", etc.)
    if len(concept) < 4:
        return False
    # Reject if the first word is a causal verb, gerund, past tense, pronoun or preposition
    if words[0] in _VERB_FIRST_WORDS:
        return False
    # Reject if the last word is a dangling preposition, conjunction, causal
    # marker or time/sequence preposition. Mirrors _TRAILING_STRIP — anything
    # listed here would have been stripped if it had been preceded by a word,
    # so finding it as the last word means the concept is incomplete.
    _TRAILING_PREPS = {
        'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'into', 'onto',
        'to', 'the', 'a', 'an', 'and', 'or', 'but',
        'after', 'before', 'during', 'while', 'since', 'until', 'till', 'once',
        'due', 'because', 'as', 'than', 'then', 'though', 'although', 'whereas',
        'through', 'throughout', 'via', 'against', 'around', 'among', 'amongst',
        'between', 'within', 'without', 'across', 'along', 'behind',
        'below', 'above', 'beneath', 'beside', 'near', 'off', 'out',
        'up', 'down', 'over', 'under', 'inside', 'outside', 'toward', 'towards',
        'so', 'yet', 'also', 'too', 'even', 'still', 'just', 'only',
        'more', 'most', 'less', 'least',
    }
    if words[-1] in _TRAILING_PREPS:
        return False
    # Reject single-word concepts that are pure past-participle adjectives
    # ("increased", "reducing") UNLESS the word is a domain noun-gerund
    # ("overfishing", "fishing", "spawning") — those are legitimate FCM concepts.
    if (len(words) == 1
            and words[0].endswith(('ed', 'ing', 'ise', 'ize'))
            and not _DOMAIN_KW.search(words[0])):
        return False
    # Reject single-word concepts that are too vague to mean anything alone
    if len(words) == 1 and words[0] in _VAGUE_SINGLES:
        return False
    # Reject if the concept is entirely made of stopwords
    non_stop = [w for w in words if w not in STOPWORDS]
    if not non_stop:
        return False
    # Reject pure-numeric concepts ("2", "100", "3.5")
    if re.fullmatch(r'[\d.,\-/]+', concept.replace(' ', '')):
        return False
    # Reject if any word is a clear passive/relational verb marker
    if set(words) & _CLAUSE_VERBS:
        return False
    # Reject if any word is an interrogative, auxiliary, pronoun, or negation
    # (these are clause markers, never noun-phrase content) — applies to ANY
    # position, not just the first word.
    word_set = set(words)
    if word_set & _DISALLOWED_ANYWHERE:
        return False
    # Reject if every word is a useless modifier (e.g. "Very High", "Much More")
    if all(w in _USELESS_MODIFIERS or w in STOPWORDS for w in words):
        return False
    # Reject if more than half the words are stopwords — concept is mostly filler
    stop_count = sum(1 for w in words if w in STOPWORDS)
    if len(words) >= 2 and stop_count * 2 > len(words):
        return False
    # Reject if any token is a sub-2-letter junk fragment (regex sometimes
    # captures stray "a", "I", initials, etc. that survived earlier filters)
    if any(len(w) < 2 for w in words):
        return False
    # Reject duplicate-word concepts ("Fish Fish", "Stock Stock")
    if len(words) >= 2 and len(set(words)) == 1:
        return False
    # ── Substantive-content gate ────────────────────────────────────────
    # A defensible FCM concept must contain at least one *substantive* word.
    # Substantive = either a domain keyword (matches _DOMAIN_KW) OR a
    # 4+ char content word that is not in any of the weak-vocabulary sets.
    # Without this gate, multi-word phrases composed entirely of generic
    # vocabulary slip through — e.g. "Implemented Various Measures"
    # (verb + qualifier + vague-noun, none of them domain-relevant).
    if not _has_substantive_word(words):
        return False
    return True


# Words that carry no FCM meaning on their own. Used by the substantive-content
# gate to detect pure-meta phrases ("Implemented Various Measures",
# "Important Recent Findings", "Several Possible Outcomes") whose individual
# tokens are all generic vocabulary.
_WEAK_WORDS_FOR_SUBSTANTIVE = (
    STOPWORDS
    | _USELESS_MODIFIERS
    | _VERB_FIRST_WORDS
    | _DISALLOWED_ANYWHERE
    | _CLAUSE_VERBS
    | _VAGUE_SINGLES
)


def _has_substantive_word(words: List[str]) -> bool:
    """True if at least one token is a real content word.

    A token counts as substantive when ANY of:
      • the token (or a suffix-stem of it) matches the domain-keyword regex,
      • the token is ≥ 4 chars AND not in the weak-vocabulary union.

    This intentionally allows known compounds like "Climate Change" — both
    tokens are individually weak (`climate` is a vague single, `change` is a
    vague action noun) but `climate` matches `_DOMAIN_KW`, so the phrase
    survives. "Implemented Various Measures" has no domain match and every
    token is weak → rejected.
    """
    for w in words:
        wl = w.lower()
        if _DOMAIN_KW.search(wl):
            return True
        if len(wl) >= 4 and wl not in _WEAK_WORDS_FOR_SUBSTANTIVE:
            return True
    return False


def _smart_title(text: str) -> str:
    """Title-case text but preserve domain acronyms (CPUE, MSY, NOAA, ...)."""
    out: List[str] = []
    for w in text.split():
        up = w.upper()
        if up in _ACRONYMS:
            out.append(up)
        else:
            out.append(w.capitalize())
    return ' '.join(out)


def normalize_concept(text: str) -> str:
    cleaned = re.sub(r'\s+', ' ', text.strip(" .,:;\n\t'\"()[]"))
    # Strip leading articles
    cleaned = re.sub(r'^(?:the|a|an)\s+', '', cleaned, flags=re.I)
    # Strip leading vague-quantity prefixes ("percentage of fish stock"
    # → "fish stock", "amount of bycatch" → "bycatch"). The quantity word
    # alone carries no domain meaning; the noun it qualifies does.
    cleaned = re.sub(
        r'^(?:percentage|percent|share|fraction|proportion|ratio|'
        r'amount|number|count|total|sum|average|mean|level|rate|'
        r'quantity|measure|size|set|group|kind|sort|type|range)s?'
        r'\s+of\s+',
        '',
        cleaned,
        flags=re.I,
    )
    # Strip trailing prepositions / conjunctions / articles iteratively
    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = _TRAILING_STRIP.sub('', cleaned).strip()
    # Remove internal stopwords but keep domain multi-word phrases intact
    words = cleaned.split()
    # Only filter stopwords from single-word results; keep multi-word as-is
    # so that "Fish Stock" and "Water Quality" survive
    if len(words) == 1 and words[0].lower() in STOPWORDS:
        return ''
    # Strip leading stopwords / useless modifiers iteratively
    # ("very high fish stock" → "fish stock", "more of overfishing" → "overfishing")
    while len(words) >= 2 and (
        words[0].lower() in STOPWORDS
        or words[0].lower() in _USELESS_MODIFIERS
    ):
        words = words[1:]
    # Strip trailing useless modifiers / bare adjectives that were not caught
    # by _TRAILING_STRIP ("fish stock high" → "fish stock")
    while len(words) >= 2 and words[-1].lower() in _USELESS_MODIFIERS:
        words = words[:-1]
    # Collapse consecutive duplicate tokens ("fish fish stock" → "fish stock")
    deduped: List[str] = []
    for w in words:
        if not deduped or deduped[-1].lower() != w.lower():
            deduped.append(w)
    words = deduped
    cleaned = ' '.join(words)
    return _smart_title(cleaned) if len(cleaned) >= 4 else ''


# ---------------------------------------------------------------------------
# Lemmatization + fuzzy-merge for duplicate consolidation
# ---------------------------------------------------------------------------

def _lemma_token(word: str) -> str:
    """Crude stemmer — strips plural endings. Acronyms untouched."""
    if word.upper() in _ACRONYMS:
        return word.upper()
    w = word.lower()
    if len(w) > 4 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 4 and w.endswith('es') and not w.endswith(('ses', 'ches', 'shes', 'xes', 'zes')):
        return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


def _lemma_key(name: str) -> str:
    """Canonical key for duplicate detection: lemmatized + sorted tokens."""
    tokens = [_lemma_token(w) for w in name.split()]
    return ' '.join(sorted(tokens))


def fuzzy_merge_concepts(
    concepts: List[str],
    edges: List[Edge],
    similarity: float = 0.88,
) -> Tuple[List[str], List[Edge]]:
    """Consolidate near-duplicate concepts and merge their edges.

    Two concepts are collapsed when they share a lemma key OR their string
    similarity ≥ `similarity`. The canonical form is the longest name in the
    cluster (more informative), preserving acronyms. Edges between the same
    (source, target) post-merge are combined by confidence-weighted mean.
    """
    if len(concepts) < 2:
        return concepts, edges

    # Stage 1: bucket by lemma key
    lemma_buckets: Dict[str, List[str]] = defaultdict(list)
    for c in concepts:
        lemma_buckets[_lemma_key(c)].append(c)

    # Stage 2: fuzzy-match any remaining singletons against other lemma keys
    canon: Dict[str, str] = {}        # original name -> canonical name
    canonical_names: List[str] = []
    for names in lemma_buckets.values():
        head = max(names, key=len)    # prefer longer, more informative form
        canonical_names.append(head)
        for n in names:
            canon[n] = head

    # Stage 1b: subsume short concepts whose token set is a strict subset of
    # exactly one longer concept ("Pollution" → "Pollution From Agricultural
    # Runoff", "Biomass" → "Fish Biomass"). When the short concept is a
    # subset of MULTIPLE longer concepts (e.g. "Habitat" appears in both
    # "Habitat Loss" and "Habitat Restoration"), it is left alone — it is a
    # genuinely independent general concept, not a duplicate.
    by_tokens: Dict[str, set] = {
        c: set(c.lower().split()) for c in canonical_names
    }
    for short_name, short_tokens in list(by_tokens.items()):
        supersets = [
            long_name for long_name, long_tokens in by_tokens.items()
            if long_name != short_name and short_tokens < long_tokens
        ]
        if len(supersets) == 1:
            target = supersets[0]
            # Re-route every original name that pointed at the short canonical
            for orig, head in list(canon.items()):
                if head == short_name:
                    canon[orig] = target
            # Drop the short name from the working canonical list
            if short_name in canonical_names:
                canonical_names.remove(short_name)

    # Collapse near-duplicate canonical names by string similarity
    merged_heads: Dict[str, str] = {}
    resolved: List[str] = []
    for name in canonical_names:
        chosen = None
        for existing in resolved:
            ratio = SequenceMatcher(None, name.lower(), existing.lower()).ratio()
            if ratio >= similarity:
                chosen = existing if len(existing) >= len(name) else name
                if chosen != existing:
                    merged_heads[existing] = chosen
                    resolved[resolved.index(existing)] = chosen
                else:
                    merged_heads[name] = chosen
                break
        if chosen is None:
            resolved.append(name)
            merged_heads[name] = name

    # Chase merged_heads chains to final canonical
    def _final(n: str) -> str:
        seen = set()
        while n in merged_heads and merged_heads[n] != n and n not in seen:
            seen.add(n)
            n = merged_heads[n]
        return n

    for orig, head in list(canon.items()):
        canon[orig] = _final(head)

    # Stage 3: re-map edges and merge duplicates
    grouped: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
    for e in edges:
        src = canon.get(e.source, e.source)
        tgt = canon.get(e.target, e.target)
        if src == tgt:
            continue
        grouped[(src, tgt)].append(e)

    new_edges: List[Edge] = []
    for (src, tgt), group in grouped.items():
        total_conf = sum(x.confidence for x in group)
        mean_w = sum(x.weight * x.confidence for x in group) / total_conf
        mean_c = total_conf / len(group)
        # Corroboration bonus from distinct supporting evidence sentences.
        mean_c = min(0.98, mean_c + _evidence_confidence_boost(len({x.evidence for x in group})))
        evidence = ' | '.join(dict.fromkeys(x.evidence for x in group))
        # Preserve the strongest edge_type label (pattern > cooccurrence > transitive)
        type_priority = {'pattern': 3, 'cooccurrence': 2, 'transitive': 1}
        best_type = max((x.edge_type for x in group), key=lambda t: type_priority.get(t, 0))
        min_hops = min(x.hops for x in group)
        new_edges.append(Edge(
            source=src,
            target=tgt,
            weight=round(mean_w, 3),
            polarity='positive' if mean_w >= 0 else 'negative',
            confidence=round(mean_c, 3),
            evidence=evidence,
            evidence_doc_id=group[0].evidence_doc_id,
            edge_type=best_type,
            hops=min_hops,
        ))

    new_edges = _dedupe_bidirectional(new_edges)
    new_edges.sort(key=lambda e: e.confidence * abs(e.weight), reverse=True)
    new_concepts = list(dict.fromkeys(canon[c] for c in concepts))
    return new_concepts, new_edges


def _is_valid_relation(source: str, target: str, weight: float = 1.0) -> bool:
    """Single source of truth: is this (source, target, weight) a defensible
    FCM edge? Applied at every edge-creation site so no clause fragment,
    morphological duplicate, substring pair, or vanishing-weight edge can
    enter the final graph.

    Rejects when:
      • either endpoint fails `_is_clean_concept` (verb fragment, vague single,
        interrogative anywhere, all-modifier phrase, etc.);
      • source equals target;
      • |weight| < 0.10 (noise from transitive damping or weak co-occurrence);
      • the two endpoints share a lemma key (plural / morphological variant
        of the same concept — would be merged anyway);
      • one endpoint's word set is a subset of the other's
        (e.g. "Fish" → "Fish Stock", "Climate" → "Climate Change") — almost
        always means the regex captured related fragments of the same phrase.
    """
    if not source or not target or source == target:
        return False
    if abs(weight) < 0.10:
        return False
    if not _is_clean_concept(source) or not _is_clean_concept(target):
        return False
    if _lemma_key(source) == _lemma_key(target):
        return False
    s_words = set(source.lower().split())
    t_words = set(target.lower().split())
    if s_words.issubset(t_words) or t_words.issubset(s_words):
        return False
    return True


def _dedupe_bidirectional(edges: List[Edge]) -> List[Edge]:
    """Collapse A→B and B→A when they carry the same polarity.

    Same-polarity reverse pairs encode the same causal meaning (especially
    from the undirected co-occurrence fallback), so we keep only the
    stronger direction (confidence × |weight|). Opposing polarity is left
    intact — that's a legitimate feedback loop in an FCM.
    """
    by_pair: Dict[Tuple[str, str], Edge] = {(e.source, e.target): e for e in edges}
    dropped: set = set()
    for (src, tgt), edge in list(by_pair.items()):
        rev_key = (tgt, src)
        if rev_key in dropped or (src, tgt) in dropped:
            continue
        rev = by_pair.get(rev_key)
        if rev is None:
            continue
        same_sign = (edge.weight >= 0) == (rev.weight >= 0)
        if not same_sign:
            continue  # genuine opposing feedback — keep both
        score_fwd = edge.confidence * abs(edge.weight)
        score_rev = rev.confidence * abs(rev.weight)
        loser = rev_key if score_fwd >= score_rev else (src, tgt)
        dropped.add(loser)
    return [e for e in edges if (e.source, e.target) not in dropped]


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class CausalExtractor:
    def extract(self, chunks: Iterable[dict]) -> Tuple[List[str], List[Edge]]:
        concepts: Dict[str, None] = {}
        grouped: Dict[Tuple[str, str], List[Edge]] = defaultdict(list)
        all_text_parts: List[str] = []

        for chunk in chunks:
            text = chunk.get('text', '')
            all_text_parts.append(text)

            for sentence in re.split(r'(?<=[.!?])\s+', text):
                for pattern, base_weight, polarity in CAUSAL_PATTERNS:
                    for match in pattern.finditer(sentence):
                        source = normalize_concept(match.group('a'))
                        target = normalize_concept(match.group('b'))
                        # Single relation-validity gate — covers cleanliness,
                        # equality, lemma duplicates, substring pairs and
                        # vanishing weight in one place.
                        if not _is_valid_relation(source, target, base_weight):
                            continue
                        # Scale magnitude by the emphatic/hedging language in the
                        # matched span (sign preserved). Scanning the whole span
                        # catches adverbs the greedy capture may have absorbed.
                        weight = _modulate_weight(base_weight, match.group(0))
                        if not _is_valid_relation(source, target, weight):
                            continue
                        concepts[source] = None
                        concepts[target] = None
                        edge = Edge(
                            source=source,
                            target=target,
                            weight=weight,
                            polarity=polarity,
                            confidence=min(0.95, 0.55 + float(chunk.get('score', 0.0))),
                            evidence=sentence.strip(),
                            evidence_doc_id=chunk.get('doc_id'),
                            edge_type='pattern',
                            hops=1,
                        )
                        grouped[(source, target)].append(edge)

        # Merge duplicate edges by confidence-weighted mean
        merged_edges: List[Edge] = []
        for (source, target), edges in grouped.items():
            total_conf = sum(e.confidence for e in edges)
            mean_weight = sum(e.weight * e.confidence for e in edges) / total_conf
            mean_conf = total_conf / len(edges)
            # Corroboration bonus: distinct supporting sentences raise confidence.
            distinct = len({e.evidence for e in edges})
            mean_conf = min(0.98, mean_conf + _evidence_confidence_boost(distinct))
            evidence = ' | '.join(dict.fromkeys(e.evidence for e in edges))
            merged_edges.append(
                Edge(
                    source=source,
                    target=target,
                    weight=round(mean_weight, 3),
                    polarity='positive' if mean_weight >= 0 else 'negative',
                    confidence=round(mean_conf, 3),
                    evidence=evidence,
                    evidence_doc_id=edges[0].evidence_doc_id,
                    edge_type='pattern',
                    hops=1,
                )
            )

        # Sort by (confidence × |weight|) descending — best pattern edges first
        merged_edges.sort(key=lambda e: e.confidence * abs(e.weight), reverse=True)

        # ── Always supplement with co-occurrence edges ───────────────────────
        # When pattern matches are few (or zero), co-occurrence fills the gap
        # so that higher relation-length requests return more results.
        supp_edges, supp_concepts = self._cooccurrence_fallback(all_text_parts, dict(concepts))

        if not merged_edges:
            # No explicit patterns at all — use co-occurrence only
            sc, se = fuzzy_merge_concepts(list(supp_concepts.keys()), supp_edges)
            se = [e for e in se if _is_valid_relation(e.source, e.target, e.weight)]
            sc = self._prune_unused_concepts(sc, se)
            return sc, _interleave_polarity(se)
        else:
            # Append co-occurrence edges that don't duplicate a pattern edge
            existing_pairs = {(e.source, e.target) for e in merged_edges}
            for e in supp_edges:
                if (e.source, e.target) not in existing_pairs:
                    merged_edges.append(e)
                    existing_pairs.add((e.source, e.target))
                    concepts[e.source] = None
                    concepts[e.target] = None
            # Re-sort: pattern-matched (high confidence) first, then interleave
            merged_edges.sort(key=lambda e: e.confidence * abs(e.weight), reverse=True)

        # Consolidate near-duplicate concepts (Plurals, acronyms, fuzzy matches)
        concept_list = list(concepts.keys())
        concept_list, merged_edges = fuzzy_merge_concepts(concept_list, merged_edges)

        # ── Transitive closure: A→B + B→C ⇒ A→C (damped, 2-hop inferred) ──
        # Densifies the graph and surfaces second-order causal pathways that
        # researchers can spot-check against the cited evidence.
        inferred = self._transitive_closure(merged_edges)
        if inferred:
            existing = {(e.source, e.target) for e in merged_edges}
            for e in inferred:
                if (e.source, e.target) not in existing:
                    merged_edges.append(e)
                    existing.add((e.source, e.target))

        merged_edges = _dedupe_bidirectional(merged_edges)
        # ── Final relation-validity pass ─────────────────────────────────
        # Belt-and-braces: re-validate every edge after fuzzy-merge could
        # have rewritten endpoints, after transitive closure widened the
        # graph, and after bidirectional dedup. Any edge that fails the
        # unified gate at this point is silently dropped.
        merged_edges = [
            e for e in merged_edges
            if _is_valid_relation(e.source, e.target, e.weight)
        ]
        # Drop concepts that no longer appear in any surviving edge so the
        # adjacency matrix doesn't carry orphan rows/columns.
        concept_list = self._prune_unused_concepts(concept_list, merged_edges)
        merged_edges.sort(key=lambda e: e.confidence * abs(e.weight), reverse=True)
        merged_edges = _interleave_polarity(merged_edges)
        return concept_list, merged_edges

    @staticmethod
    def _prune_unused_concepts(concepts: List[str], edges: List[Edge]) -> List[str]:
        used: set = set()
        for e in edges:
            used.add(e.source)
            used.add(e.target)
        return [c for c in concepts if c in used]

    # ------------------------------------------------------------------
    # Fallback: domain keyword co-occurrence (3-sentence rolling window,
    # all-pairs within the window). Wider context = denser, more
    # informative graphs; weights scale by window proximity.
    # ------------------------------------------------------------------
    def _cooccurrence_fallback(
        self,
        text_parts: List[str],
        concepts: Dict[str, None],
        window: int = 3,
    ) -> Tuple[List[Edge], Dict[str, None]]:
        """
        Pair domain keywords inside a `window`-sentence rolling context.
        Polarity inherits the *window's* dominant sentiment so co-occurrence
        edges aren't blindly positive — a faithful FCM contains both signs.
        Distance-weighted: same-sentence pairs count fully, neighboring
        sentences count proportionally less.
        """
        # Per-pair tallies split by polarity, weighted by proximity
        pos_count: Dict[Tuple[str, str], float] = defaultdict(float)
        neg_count: Dict[Tuple[str, str], float] = defaultdict(float)
        evidence_pool: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        full_text = ' '.join(text_parts)
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', full_text) if s.strip()]

        # Pre-extract domain keywords per sentence — keywords must clear the
        # cleanliness gate so weak / vague tokens never enter the pool.
        per_sent: List[List[str]] = []
        for sent in sentences:
            found = list(dict.fromkeys(
                normalize_concept(m.group(0))
                for m in _DOMAIN_KW.finditer(sent)
            ))
            per_sent.append([
                f for f in found
                if f
                and (' ' in f or len(f) >= 6)
                and _is_clean_concept(f)
            ])

        for i, anchor_sent in enumerate(sentences):
            anchor_keywords = per_sent[i]
            if not anchor_keywords:
                continue
            window_text_parts = [anchor_sent]
            polarity = _sentence_polarity(anchor_sent)

            # Slide window forward up to `window-1` sentences
            for offset in range(window):
                j = i + offset
                if j >= len(sentences):
                    break
                # weight halves for each sentence further away
                proximity = 1.0 / (1 + offset)
                neighbor_kw = per_sent[j]
                if j != i:
                    window_text_parts.append(sentences[j])
                    polarity = polarity if polarity < 0 else _sentence_polarity(sentences[j])

                # All-pairs across anchor + neighbor (skip self-pair, skip backwards
                # within same sentence to avoid duplicate work — but keep both
                # directions since FCM is directed)
                for a in anchor_keywords:
                    for b in neighbor_kw:
                        if a == b:
                            continue
                        # Avoid double-tallying within the same sentence
                        if j == i and a >= b:
                            continue
                        if polarity < 0:
                            neg_count[(a, b)] += proximity
                        else:
                            pos_count[(a, b)] += proximity
                        concepts[a] = None
                        concepts[b] = None
                        if len(evidence_pool[(a, b)]) < 2:
                            evidence_pool[(a, b)].append(anchor_sent.strip())

        combined = {k: pos_count.get(k, 0) + neg_count.get(k, 0)
                    for k in set(pos_count) | set(neg_count)}

        if not combined:
            for item in re.findall(
                r'\b[A-Za-z][A-Za-z\-/]{3,}\b(?:\s+[A-Za-z][A-Za-z\-/]{3,}\b){0,2}',
                full_text,
            )[:12]:
                c = normalize_concept(item)
                if c:
                    concepts[c] = None
            return [], concepts

        max_count = max(combined.values())
        # Pairs must co-occur with enough proximity weight to be informative.
        # 0.50 corresponds to a single neighbouring-sentence hit; 1.0 is a
        # same-sentence hit. Lowering to 0.50 brings back richer detail
        # without re-admitting random background pairs.
        strong = [(k, c) for k, c in combined.items() if c >= 0.5]
        if not strong:
            # Fall back to the strongest tier if no pair clears the floor
            strong = sorted(combined.items(), key=lambda x: -x[1])[:8]
        edges: List[Edge] = []
        # Cap co-occurrence at the top-25 by count — generous enough for a
        # detailed graph, tight enough to keep the bottom-rank noise out.
        for (source, target), count in sorted(strong, key=lambda x: -x[1])[:25]:
            pos = pos_count.get((source, target), 0)
            neg = neg_count.get((source, target), 0)
            is_neg = neg > pos
            magnitude = 0.3 + 0.4 * (count / max_count)
            weight = round(-magnitude if is_neg else magnitude, 3)
            # Apply the unified relation gate so co-occurrence cannot bypass
            # the cleanliness, lemma-duplicate, or substring-pair checks.
            if not _is_valid_relation(source, target, weight):
                continue
            evidence_text = ' || '.join(evidence_pool.get((source, target), [])) \
                            or f'co-occurrence ({count:.1f} weighted, {pos:.1f}+ / {neg:.1f}-)'
            edges.append(Edge(
                source=source,
                target=target,
                weight=weight,
                polarity='negative' if is_neg else 'positive',
                confidence=0.40,
                evidence=evidence_text,
                evidence_doc_id=None,
                edge_type='cooccurrence',
                hops=1,
            ))
        return edges, concepts

    # ------------------------------------------------------------------
    # Transitive closure: derive A→C from A→B + B→C with damped weight
    # ------------------------------------------------------------------
    def _transitive_closure(
        self,
        edges: List[Edge],
        damping: float = 0.7,
        min_weight: float = 0.20,
        max_new: int = 20,
    ) -> List[Edge]:
        """
        Generate 2-hop inferred edges. For each chain A→B→C with no direct
        A→C edge present, add A→C with `damping × w_AB × w_BC`. Marked
        `edge_type='transitive', hops=2` so researchers can filter or
        visually distinguish them from primary evidence.
        """
        # Build outgoing adjacency from existing edges
        out: Dict[str, List[Edge]] = defaultdict(list)
        existing: set = set()
        for e in edges:
            out[e.source].append(e)
            existing.add((e.source, e.target))

        candidates: Dict[Tuple[str, str], Dict] = {}
        for e1 in edges:
            for e2 in out.get(e1.target, []):
                a, c = e1.source, e2.target
                if a == c or (a, c) in existing:
                    continue
                w = damping * e1.weight * e2.weight
                if abs(w) < min_weight:
                    continue
                # Same unified gate — inferred edges must also be valid
                # relations (e.g. don't infer "Fish Stock → Fish")
                if not _is_valid_relation(a, c, w):
                    continue
                conf = damping * min(e1.confidence, e2.confidence)
                key = (a, c)
                # Keep the strongest derivation if multiple chains exist
                if key not in candidates or abs(w) > abs(candidates[key]['weight']):
                    candidates[key] = {
                        'weight': w,
                        'confidence': conf,
                        'via': e1.target,
                        'evidence_a': e1.evidence,
                        'evidence_b': e2.evidence,
                    }

        # Sort by absolute weight × confidence and cap
        ranked = sorted(
            candidates.items(),
            key=lambda kv: abs(kv[1]['weight']) * kv[1]['confidence'],
            reverse=True,
        )[:max_new]

        new_edges: List[Edge] = []
        for (a, c), info in ranked:
            new_edges.append(Edge(
                source=a,
                target=c,
                weight=round(info['weight'], 3),
                polarity='positive' if info['weight'] >= 0 else 'negative',
                confidence=round(info['confidence'], 3),
                evidence=f"inferred via {info['via']} :: {info['evidence_a']} → {info['evidence_b']}",
                evidence_doc_id=None,
                edge_type='transitive',
                hops=2,
            ))
        return new_edges
