"""Pull hard constraints out of a free-text car query.

Why this exists: the search used to embed the whole query and rank by cosine
similarity alone, so "I want automatic transmission. Electric car." was just
tokens averaged into one vector. Requirements got diluted to noise — a four-clause
query pushed the best electric car to rank #38 behind 37 gas cars, even though
"electric car" on its own returned electrics in the top 8.

So we do two things here:
  1. Lift explicit attribute requirements out of the text into real filters.
  2. Strip the matched phrases from the text handed to the embedder, so the
     semantic ranking is spent on the descriptive part ("family car with good
     fuel mileage") instead of re-litigating constraints already enforced exactly.
"""
import re

# Field values are a small closed set (see cars_with_vibes.csv), so patterns map
# phrasings onto exactly those values. Order matters within a field: the first
# match wins, so put the more specific pattern first (mini-van before van).
FUEL_PATTERNS = [
    ("electric", r"\b(?:all[-\s]?electric|battery[-\s]?electric|electric|bev|ev)\b"),
    ("hybrid", r"\b(?:plug[-\s]?in\s+hybrid|phev|hybrid)\b"),
    ("diesel", r"\b(?:diesel|tdi)\b"),
    ("gas", r"\b(?:gasoline|petrol|gas(?:\s+powered)?)\b"),
]

# The optional trailing "transmission" is consumed so stripping doesn't leave the
# orphan word behind in the text handed to the embedder.
TRANSMISSION_PATTERNS = [
    ("manual", r"\b(?:manual|stick[-\s]?shift|stick|standard|"
                r"\d[-\s]?speed\s+manual)(?:\s+transmission)?\b"),
    ("automatic", r"\b(?:automatic|auto)(?:\s+transmission)?\b"),
]

TYPE_PATTERNS = [
    ("mini-van", r"\b(?:mini[-\s]?van|minivan|people\s+carrier)\b"),
    ("SUV", r"\b(?:suv|sport\s+utility|crossover)\b"),
    ("pickup", r"\b(?:pick[-\s]?up|pickup)\b"),
    ("truck", r"\b(?:truck|lorry)\b"),
    ("sedan", r"\b(?:sedan|saloon|(?:4|four)[-\s]?door)\b"),
    ("coupe", r"\b(?:coupe|(?:2|two)[-\s]?door)\b"),
    ("convertible", r"\b(?:convertible|cabriolet|drop[-\s]?top|roadster)\b"),
    ("hatchback", r"\b(?:hatchback|hatch)\b"),
    ("wagon", r"\b(?:wagon|estate|station\s+wagon)\b"),
    ("van", r"\b(?:cargo\s+van|van)\b"),
    ("bus", r"\bbus\b"),
    ("offroad", r"\b(?:off[-\s]?road)\b"),
]

DRIVE_PATTERNS = [
    ("4wd", r"\b(?:4wd|4x4|awd|all[-\s]?wheel[-\s]?drive|four[-\s]?wheel[-\s]?drive)\b"),
    ("fwd", r"\b(?:fwd|front[-\s]?wheel[-\s]?drive)\b"),
    ("rwd", r"\b(?:rwd|rear[-\s]?wheel[-\s]?drive)\b"),
]

FIELD_PATTERNS = {
    "fuel": FUEL_PATTERNS,
    "transmission": TRANSMISSION_PATTERNS,
    "type": TYPE_PATTERNS,
    "drive": DRIVE_PATTERNS,
}

# "good gas mileage" / "easy on gas" is a request for efficiency, not a demand for
# a gasoline engine. Without this the single most common way people ask for a
# fuel-efficient car would silently filter out every hybrid and EV.
GAS_FALSE_POSITIVES = re.compile(
    r"\b(?:good|great|better|best|excellent|decent|easy|light|save|saving|saver|"
    r"cheap|low|efficient)\b[^.]{0,20}\bgas\b"
    r"|\bgas\s+(?:mileage|saver|savings|efficient|efficiency|economy|sipper)\b",
    re.IGNORECASE,
)

# --- budget -----------------------------------------------------------------
#
# Price has to be lifted out of the text for the same reason the other constraints
# do, plus one of its own: the raw digits actively poison the ranking. Embedding
# "cars under 5000 dollars" matched the *Fiat 500* and *Ford 500* model names, so
# the old behaviour was both unfiltered and skewed toward the wrong cars.

_AMOUNT = r"\$?\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s?(k\b)?"

# A number is only money if something says so — a bare "under 2015" is a model year,
# and "under 50,000 miles" is mileage. Requiring a currency cue keeps those out.
_CURRENCY_BEFORE = r"(?:\$|usd\s*)"
_CURRENCY_AFTER = r"(?:\s*(?:dollars?|bucks?|usd))"
_MILEAGE_UNIT = re.compile(r"^\s*(?:miles?|mi|mileage|km|kilometers?|odo\w*)\b",
                           re.IGNORECASE)

_UNDER = re.compile(
    r"\b(?:under|below|less\s+than|no\s+more\s+than|not?\s+more\s+than|at\s+most|"
    r"max(?:imum)?(?:\s+of)?|up\s+to|cheaper\s+than|within|budget\s+(?:of|is|:)?|"
    r"for)\s+" + _AMOUNT, re.IGNORECASE)

_OVER = re.compile(
    r"\b(?:over|above|more\s+than|at\s+least|min(?:imum)?(?:\s+of)?|"
    r"starting\s+(?:at|from)|from)\s+" + _AMOUNT, re.IGNORECASE)

_BETWEEN = re.compile(
    r"\b(?:between|from)\s+" + _AMOUNT + r"\s*(?:and|to|-|–)\s*" + _AMOUNT,
    re.IGNORECASE)

# A negated constraint is dropped rather than inverted: "not a truck" reliably means
# "don't filter to trucks", but turning it into "exclude trucks" over-reads a
# regex-only parser. Excluding is handled explicitly below where it's unambiguous.
NEGATION = re.compile(
    r"\b(?:not|no|non|never|without|avoid|except|excluding|don'?t\s+want|"
    r"do\s+not\s+want|rather\s+not|anything\s+but)\b[\s\w-]{0,15}$",
    re.IGNORECASE,
)


def _is_negated(query, match_start):
    """Look at the ~40 chars before the match for a negation cue."""
    window = query[max(0, match_start - 40):match_start]
    return bool(NEGATION.search(window))


def _to_amount(query, match, digits_group):
    """Turn a matched number into dollars, or None if it isn't money at all.

    Rejects mileage ("under 50,000 miles") and bare model years ("under 2015"),
    either of which would otherwise be read as a budget.
    """
    raw = match.group(digits_group)
    if raw is None:
        return None
    suffix = match.group(digits_group + 1)

    tail = query[match.end():]
    if _MILEAGE_UNIT.match(tail):
        return None

    value = float(raw.replace(",", ""))
    if suffix:                       # "5k" / "10 k"
        value *= 1000
    else:
        # No currency marker anywhere near it and it reads like a year — not a price.
        head = query[max(0, match.start() - 6):match.start()]
        cued = (re.search(_CURRENCY_BEFORE, head + match.group(0), re.IGNORECASE)
                or re.match(_CURRENCY_AFTER, tail, re.IGNORECASE)
                or "," in raw)
        if 1900 <= value <= 2100 and not cued:
            return None
    return int(value)


def parse_price(query):
    """Extract a (min_price, max_price) budget and the spans it occupied.

    Either bound may be None. Returns (None, None, []) when no budget is stated.
    """
    spans = []
    low = high = None

    match = _BETWEEN.search(query)
    if match:
        a = _to_amount(query, match, 1)
        b = _to_amount(query, match, 3)
        if a is not None and b is not None:
            low, high = min(a, b), max(a, b)
            spans.append(match.span())
            return low, high, spans

    match = _UNDER.search(query)
    if match:
        amount = _to_amount(query, match, 1)
        if amount is not None:
            high = amount
            spans.append(match.span())

    match = _OVER.search(query)
    if match and match.span() not in spans:
        amount = _to_amount(query, match, 1)
        if amount is not None:
            low = amount
            spans.append(match.span())

    # A nonsensical pair ("over 20k under 5k") is dropped rather than guessed at.
    if low is not None and high is not None and low > high:
        return None, None, []
    return low, high, spans


def parse_query(query):
    """Extract hard constraints from free text.

    Returns (constraints, semantic_query, negated, price):
      constraints    — {field: value} to filter on exactly
      semantic_query — query with matched constraint phrases removed, for embedding
      negated        — {field: value} the user explicitly ruled out, to exclude
      price          — (min_price, max_price), either bound possibly None
    """
    constraints = {}
    negated = {}
    spans = []

    # Blank out efficiency phrasings first so the "gas" pattern can't see them.
    maskable = GAS_FALSE_POSITIVES.sub(lambda m: " " * len(m.group()), query)

    # Budget first, so its digits are removed before the other patterns run and
    # before the remainder reaches the embedder.
    low, high, price_spans = parse_price(maskable)
    maskable = _blank(maskable, price_spans)

    for field, patterns in FIELD_PATTERNS.items():
        for value, pattern in patterns:
            match = re.search(pattern, maskable, re.IGNORECASE)
            if not match:
                continue
            if _is_negated(maskable, match.start()):
                negated.setdefault(field, value)
            else:
                constraints[field] = value
            spans.append(match.span())
            break  # first match wins for this field

    semantic_query = _strip_spans(query, spans, always=price_spans)
    return constraints, semantic_query, negated, (low, high)


# Words that carry no ranking signal once the constraints are lifted out. "car" and
# friends are included because "Electric car." reduces to a bare "car" after
# stripping, which is pure noise in the embedding.
_FILLER = {
    "a", "an", "the", "i", "want", "need", "looking", "look", "for", "with", "and",
    "or", "but", "some", "something", "any", "anything", "that", "this", "is", "are",
    "be", "would", "like", "me", "my", "we", "our", "it", "in", "of", "to", "please",
    "car", "cars", "vehicle", "vehicles", "auto", "one", "thing", "prefer", "get",
    "buy", "find", "show", "give", "am", "'m",
    # money words: whatever survives a budget strip is never a ranking signal
    "dollars", "dollar", "bucks", "buck", "usd", "budget", "price", "priced",
    "cost", "costs", "under", "over", "max", "maximum", "min", "minimum", "range",
}


NEUTRAL_QUERY = "used car for sale"


def _content_words(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return [w for w in words if w not in _FILLER]


def _blank(text, spans):
    out = list(text)
    for start, end in spans:
        for i in range(start, min(end, len(out))):
            out[i] = " "
    return "".join(out)


def _tidy(text):
    """Drop clauses left with no content of their own ("I want a with", "car")."""
    clauses = [c for c in re.split(r"[.;]", text) if _content_words(c)]
    out = ". ".join(re.sub(r"\s+", " ", c).strip(" ,") for c in clauses)
    out = re.sub(r"\s+([.,;])", r"\1", out)
    return re.sub(r",\s*(?=,)", "", out).strip(" .,;")


def _strip_spans(query, spans, always=()):
    """Remove matched constraint phrases, leaving the descriptive remainder.

    Falls back when stripping leaves nothing meaningful — a bare "electric automatic
    sedan" would otherwise embed an empty string. That fallback is safe for the
    attribute phrases because they are enforced as filters either way.

    `always` spans are never restored by the fallback. Budget text lives there: the
    digits in "under 5000 dollars" match the *Fiat 500* and *Ford 500* model names,
    so putting them back would skew the ranking toward the wrong cars entirely.
    """
    base = _blank(query, always) if always else query

    def fallback():
        tidied = _tidy(base)
        if tidied:
            return tidied
        # Everything was a constraint. Embedding a neutral token beats restoring the
        # original, which for a budget query would put the stripped digits back and
        # skew ranking toward model names like "500". The filters have already run,
        # so ranking within the survivors is the only thing at stake here.
        return NEUTRAL_QUERY if always else query

    if not spans:
        return fallback()

    stripped = _tidy(_blank(base, spans))
    if len(_content_words(stripped)) < 2:
        return fallback()
    return stripped


def describe(constraints):
    """Human-readable constraint list for the UI."""
    labels = {"fuel": "fuel", "transmission": "transmission",
              "type": "body type", "drive": "drivetrain"}
    return ", ".join(f"{labels.get(k, k)}={v}" for k, v in constraints.items())
