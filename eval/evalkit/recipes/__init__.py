"""P0 ground-truth recipes as a package: `_base` holds the @recipe registry
and shared helpers; one module per suite family registers its recipes on
import. Importing this package populates RECIPES exactly as the old
single-module recipes.py did."""

from ._base import (  # noqa: F401
    PARTICIPANT_REACTIONS_MIN_MESSAGES,
    RECIPES,
    _connect,
    _qt,
    _real,
    _scope_kwargs,
    _v,
    _year_bounds,
    recipe,
)
from . import (  # noqa: E402,F401  (imported for their @recipe side effects)
    s1_activity,
    s2_response,
    s3_reactions,
    s4_words,
    s5_milestones,
    s6_s8_stats,
    s9_retrieval,
    s11_conflict,
    s12_nudges,
    s13_drift,
    s14_voiceprint,
    s15_persona,
)
