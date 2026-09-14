"""Shared parser: extract canonical attribute levels from a vignette's literal
question_text. Used by every recomputation script so there is one source of
truth for "what level did this profile actually show," rather than each
script re-deriving it independently.

The seven attributes and their canonical levels (see the paper's Table 1):
  efficacy: 50 | 70 | 90            (reference: 50)
  duration: 1 | 5                   (reference: 1)
  major_side_effect: 1/10k | 1/1m   (reference: 1/10k)
  minor_side_effect: 1/10 | 1/30    (reference: 1/10)
  fda_status: full | emergency      (reference: full)
  origin: US | UK | China           (reference: US)
  endorsement: Trump | Biden | CDC | WHO   (reference: Trump)
"""
import re

REFERENCE = {
    "efficacy": 50,
    "duration": 1,
    "major_side_effect": "1/10k",
    "minor_side_effect": "1/10",
    "fda_status": "full",
    "origin": "US",
    "endorsement": "Trump",
}

ATTR_ORDER = ["efficacy", "duration", "major_side_effect", "minor_side_effect", "fda_status", "origin", "endorsement"]


def _efficacy(s):
    return int(re.search(r"Efficacy:\s*(\d+)%", s).group(1))


def _duration(s):
    return int(re.search(r"Duration of protection:\s*(\d+)\s*year", s).group(1))


def _major(s):
    return "1/1m" if "1 in 1,000,000" in s else "1/10k"


def _minor(s):
    return "1/30" if "1 in 30" in s else "1/10"


def _fda(s):
    return "emergency" if "emergency use authorization" in s else "full"


def _origin(s):
    m = re.search(r"Country of origin:\s*([^;.]+)", s)
    v = m.group(1).strip()
    return {"United States": "US", "United Kingdom": "UK", "China": "China"}[v]


def _endorsement(s):
    m = re.search(r"Endorsed by:\s*([^;.\n]+)", s)
    v = m.group(1).strip()
    if "Trump" in v:
        return "Trump"
    if "Biden" in v:
        return "Biden"
    if "Centers for Disease Control" in v or v.strip() == "US CDC" or "CDC" in v:
        return "CDC"
    if "World Health Organization" in v or "WHO" in v:
        return "WHO"
    raise ValueError(f"unrecognized endorsement: {v!r}")


_EXTRACTORS = {
    "efficacy": _efficacy,
    "duration": _duration,
    "major_side_effect": _major,
    "minor_side_effect": _minor,
    "fda_status": _fda,
    "origin": _origin,
    "endorsement": _endorsement,
}


def parse_profile_block(block_text):
    """Parse one profile's attribute levels from its slice of the vignette text."""
    return {attr: fn(block_text) for attr, fn in _EXTRACTORS.items()}


def parse_pair(question_text):
    """Split a two-profile vignette into (profile_a, profile_b) attribute dicts."""
    a_start = question_text.index("Vaccine A:")
    b_start = question_text.index("Vaccine B:")
    q_start = question_text.index("\n\nIf you had to choose")
    a_block = question_text[a_start:b_start]
    b_block = question_text[b_start:q_start]
    return parse_profile_block(a_block), parse_profile_block(b_block)


def parse_single_profile(question_text):
    """Parse a one-profile vignette (used by the anchor-profile instrument)."""
    end = question_text.index("\n\nHow likely")
    return parse_profile_block(question_text[:end])
