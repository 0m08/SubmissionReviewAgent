"""
strategy_manager.py — Feedback-Driven Strategy Selector
=========================================================
Strategies are selected based on the IDENTIFIED PROBLEM in the reviewer's feedback,
not just round count. This ensures the right mindset shift is applied to the actual
failure, not just the next item in an escalation queue.

SELECTION LOGIC (two-tier):
  Tier 1 — Feedback-driven (primary):
    Classify last_feedback → failure_class.
    Find strategies where failure_class matches AND consecutive_failures >= 1.
    Pick highest-priority match → activates immediately on the identified problem.

  Tier 2 — Round-count fallback (safety net):
    Used when feedback is unclassifiable ('unknown') or no class-matched strategy
    is available. Falls back to threshold-based escalation (original behaviour).

FAILURE CLASSES (accuracy domain):
  'factual'       — wrong value, wrong state, incorrect reading, missing element
  'visibility'    — component present but unclear, too small, obscured, poorly lit
  'scene_mismatch'— fundamentally wrong scene, wrong subject category, wrong context

FAILURE CLASSES (copyright domain):
  'structural'    — composition/angle/form-factor similarity to reference
  'color'         — colour-palette similarity to reference
  'any'           — matches any failure class (last-resort strategies)

ZERO-IMPACT GUARANTEE:
  When select_strategy() returns None, the pipeline is UNCHANGED.
  _accuracy_strategy_directive and _copyright_strategy_directive remain ''.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR    = os.path.dirname(_THIS_DIR)
STRATEGIES_DIR = os.path.join(_PARENT_DIR, "strategies")


# ===========================================================================
# DATA MODELS
# ===========================================================================

@dataclass
class StrategyCondition:
    """
    Snapshot of pipeline state used to select the right strategy.

    Attributes
    ----------
    is_copyright_round      : True when the failure is from the copyright reviewer
    consecutive_failures    : How many rounds in a row this reviewer said No
    stale_similarity_rounds : Rounds copyright similarity did not improve
    tried_strategy_ids      : IDs already applied this session (to skip)
    regression_detected     : Whether the accuracy reviewer flagged a regression
    last_feedback           : Last reviewer feedback text — used for problem classification
    """
    is_copyright_round:      bool
    consecutive_failures:    int
    stale_similarity_rounds: int       = 0
    tried_strategy_ids:      List[str] = field(default_factory=list)
    regression_detected:     bool      = False
    last_feedback:           str       = ""


@dataclass
class Strategy:
    """
    A parsed strategy definition loaded from a .md file.

    Attributes
    ----------
    id                       : Unique identifier
    name                     : Human-readable name
    applicable               : "copyright" | "accuracy" | "both"
    trigger                  : "stale_similarity" | "consecutive_failures" | "always"
    failure_class            : Problem class this strategy solves (see module docstring)
    priority                 : Lower number = tried first within the same class
    min_consecutive_failures : Threshold for round-count fallback selection
    min_stale_rounds         : Stale similarity rounds required (copyright only)
    accuracy_directive       : Text injected into the accuracy reviewer's prompt
    copyright_directive      : Text injected into the copyright reviewer's context
    description              : Human-readable description
    source_file              : Absolute path to the .md file
    """
    id:                       str
    name:                     str
    applicable:               str
    trigger:                  str
    failure_class:            str
    priority:                 int
    min_consecutive_failures: int
    min_stale_rounds:         int
    accuracy_directive:       str
    copyright_directive:      str
    description:              str
    source_file:              str


# ===========================================================================
# PROBLEM CLASSIFIER
# ===========================================================================

# Keyword sets for each failure class.
# Order matters: check scene_mismatch first (highest specificity), then
# visibility, then factual. Unknown is the final fallback.

_SCENE_MISMATCH_KEYWORDS = [
    "wrong scene", "different context", "unrelated", "not related",
    "completely different", "wrong subject", "wrong type", "wrong environment",
    "different device", "different equipment", "does not show", "doesn't show",
    "not depicting", "shows office", "shows unrelated", "wrong category",
    "different machine", "fundamentally wrong", "not what",
    # additional synonyms observed in Gemini reviewer output
    "incorrect subject", "depicts a", "showing a", "irrelevant",
    "not the same", "mismatched", "wrong equipment", "not applicable",
    "unrelated object", "wrong item", "shows something", "incorrect scene",
]

_VISIBILITY_KEYWORDS = [
    "cannot see", "can't see", "unclear", "not visible", "too small",
    "obscured", "hard to confirm", "cannot confirm", "barely visible",
    "not clearly", "poorly lit", "blocked", "hidden", "not legible",
    "illegible", "not readable", "not discernible", "not distinguishable",
    "difficult to see", "can't confirm", "unable to confirm",
    # additional synonyms observed in Gemini reviewer output
    "not clear", "low resolution", "blurry", "out of focus",
    "too dark", "underexposed", "overexposed", "washed out",
    "not enough detail", "hard to read", "hard to identify",
    "cannot be confirmed", "cannot verify", "hard to distinguish",
]

_FACTUAL_KEYWORDS = [
    "wrong value", "incorrect value", "should read", "must read",
    "incorrect reading", "wrong reading", "wrong state", "incorrect state",
    "should show", "must show", "should be", "must be", "wrong position",
    "incorrect position", "wrong direction", "facing wrong", "wrong label",
    "missing label", "wrong number", "incorrect number", "should display",
    "displays incorrectly", "wrong gauge", "wrong setting", "incorrect setting",
    "missing component", "missing element", "absent", "not present",
    "technically incorrect", "factually incorrect",
    # additional synonyms observed in Gemini reviewer output
    "incorrectly shown", "shown incorrectly", "incorrect detail",
    "wrong detail", "incorrect component", "shows the wrong",
    "not accurate", "inaccurate", "misrepresented", "shows incorrect",
    "does not match", "doesn't match", "incorrect orientation",
]

# Copyright-specific keyword sets
_COPYRIGHT_COLOR_KEYWORDS = [
    "colour", "color", "palette", "hue", "tint", "shade", "orange", "red",
    "blue", "green", "signature colour", "brand colour", "branded colour",
    "distinctive colour", "recognisable colour",
    # additional synonyms observed in copyright reviewer output
    "same colour", "same color", "similar colour", "similar color",
    "yellow", "purple", "signature", "identifiable colour", "distinctive hue",
    "color scheme", "colour scheme", "color palette", "colour palette",
]

_COPYRIGHT_STRUCTURAL_KEYWORDS = [
    "angle", "perspective", "composition", "shape", "form", "silhouette",
    "profile", "structure", "layout", "arrangement", "viewpoint", "camera",
    "identical", "same angle", "same view", "recognisable", "identifiable",
    "distinctive form", "distinctive shape",
    # additional synonyms observed in copyright reviewer output
    "too similar", "nearly identical", "mirror image",
    "same shape", "same form", "same structure", "resembles",
    "unmistakably", "recognizable", "same composition", "same layout",
]


def _classify_accuracy_failure(feedback: str) -> str:
    """
    Classify an accuracy reviewer rejection into a problem class.
    Returns: 'scene_mismatch' | 'visibility' | 'factual' | 'unknown'

    Iterates explicitly (vs any()) so we can log which keyword triggered
    the classification — essential for tuning the keyword lists.
    """
    f = feedback.lower()
    for kw in _SCENE_MISMATCH_KEYWORDS:
        if kw in f:
            print(f"   [classifier] scene_mismatch \u2190 '{kw}'")
            return "scene_mismatch"
    for kw in _VISIBILITY_KEYWORDS:
        if kw in f:
            print(f"   [classifier] visibility \u2190 '{kw}'")
            return "visibility"
    for kw in _FACTUAL_KEYWORDS:
        if kw in f:
            print(f"   [classifier] factual \u2190 '{kw}'")
            return "factual"
    print(f"   [classifier] unknown \u2014 no keyword matched "
          f"(feedback snippet: {f[:80]!r})")
    return "unknown"


def _classify_copyright_failure(feedback: str) -> str:
    """
    Classify a copyright reviewer rejection into a problem class.
    Returns: 'color' | 'structural' | 'unknown'
    """
    f = feedback.lower()
    for kw in _COPYRIGHT_COLOR_KEYWORDS:
        if kw in f:
            print(f"   [classifier] color \u2190 '{kw}'")
            return "color"
    for kw in _COPYRIGHT_STRUCTURAL_KEYWORDS:
        if kw in f:
            print(f"   [classifier] structural \u2190 '{kw}'")
            return "structural"
    print(f"   [classifier] unknown \u2014 no copyright keyword matched "
          f"(feedback snippet: {f[:80]!r})")
    return "unknown"


def classify_failure(feedback: str, is_copyright_round: bool) -> str:
    """
    Public entry point: classify reviewer feedback into a problem class.
    The returned class is used to select the matching strategy.
    """
    if not feedback or not feedback.strip():
        return "unknown"
    if is_copyright_round:
        return _classify_copyright_failure(feedback)
    return _classify_accuracy_failure(feedback)


# ===========================================================================
# PARSER
# ===========================================================================

def _parse_strategy_file(filepath: str) -> Optional[Strategy]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        print(f"[strategy_manager] ⚠ Cannot read {filepath}: {e}")
        return None

    fm_match = re.search(r"^---\s*\n(.*?)\n---", raw, re.DOTALL)
    if not fm_match:
        print(f"[strategy_manager] ⚠ No frontmatter in {filepath}")
        return None

    fm: Dict[str, str] = {}
    for line in fm_match.group(1).splitlines():
        kv = re.match(r"^\s*(\w+)\s*:\s*(.+)", line)
        if kv:
            fm[kv.group(1).strip()] = kv.group(2).strip()

    required = ["id", "name", "applicable", "trigger", "failure_class",
                "priority", "min_consecutive_failures", "min_stale_rounds"]
    missing = [k for k in required if k not in fm]
    if missing:
        print(f"[strategy_manager] ⚠ Missing fields {missing} in {filepath}")
        return None

    desc_match = re.search(r"##\s+Description\s*\n(.*?)(?=\n##|\Z)", raw, re.DOTALL)
    acc_match  = re.search(r"##\s+Accuracy Reviewer Directive\s*\n(.*?)(?=\n##|\Z)",
                            raw, re.DOTALL)
    cp_match   = re.search(r"##\s+Copyright Reviewer Directive\s*\n(.*?)(?=\n##|\Z)",
                            raw, re.DOTALL)

    return Strategy(
        id                       = fm["id"],
        name                     = fm["name"],
        applicable               = fm["applicable"].lower(),
        trigger                  = fm["trigger"].lower(),
        failure_class            = fm["failure_class"].lower(),
        priority                 = int(fm["priority"]),
        min_consecutive_failures = int(fm["min_consecutive_failures"]),
        min_stale_rounds         = int(fm["min_stale_rounds"]),
        accuracy_directive       = acc_match.group(1).strip() if acc_match else "",
        copyright_directive      = cp_match.group(1).strip() if cp_match else "",
        description              = desc_match.group(1).strip() if desc_match else "",
        source_file              = filepath,
    )


# ===========================================================================
# LOADER
# ===========================================================================

_CACHED_STRATEGIES: Optional[List[Strategy]] = None


def load_strategies(force_reload: bool = False) -> List[Strategy]:
    """Load and cache all strategy .md files, sorted by priority."""
    global _CACHED_STRATEGIES
    if _CACHED_STRATEGIES is not None and not force_reload:
        return _CACHED_STRATEGIES

    if not os.path.isdir(STRATEGIES_DIR):
        print(f"[strategy_manager] ⚠ Strategies folder not found: {STRATEGIES_DIR}")
        _CACHED_STRATEGIES = []
        return []

    strategies = []
    for fname in sorted(os.listdir(STRATEGIES_DIR)):
        if not fname.endswith(".md"):
            continue
        s = _parse_strategy_file(os.path.join(STRATEGIES_DIR, fname))
        if s:
            strategies.append(s)
            print(f"[strategy_manager] ✓ {s.id} "
                  f"(class={s.failure_class}, applicable={s.applicable}, pri={s.priority})")

    strategies.sort(key=lambda s: s.priority)
    _CACHED_STRATEGIES = strategies
    print(f"[strategy_manager] {len(strategies)} strategies ready.")
    return strategies


# ===========================================================================
# SELECTOR  (two-tier: feedback-driven → round-count fallback)
# ===========================================================================

def _threshold_met_fallback(s: Strategy, condition: StrategyCondition) -> bool:
    """
    Check that a strategy's minimum thresholds are met for ROUND-COUNT fallback.
    This is the conservative check used when the problem class is 'unknown'.
    """
    if s.trigger == "stale_similarity":
        if not condition.is_copyright_round:
            return False
        return (condition.stale_similarity_rounds >= s.min_stale_rounds and
                condition.consecutive_failures >= s.min_consecutive_failures)
    elif s.trigger == "consecutive_failures":
        return condition.consecutive_failures >= s.min_consecutive_failures
    elif s.trigger == "always":
        return True
    return False


def select_strategy(condition: StrategyCondition) -> Optional[Strategy]:
    """
    Select the best strategy for the current condition.

    TIER 1 — Feedback-driven (primary):
      Classify last_feedback → problem_class.
      If problem_class is known, find strategies where:
        - failure_class == problem_class (or 'any')
        - applicable domain matches
        - not already tried
        - consecutive_failures >= 1 (one confirmed failure is enough when we know the problem)
      Return the highest-priority match.

    TIER 2 — Round-count fallback (safety net):
      Used when problem_class == 'unknown' (feedback was unclassifiable) OR
      when no Tier-1 match was found.
      Applies full min_consecutive_failures thresholds from strategy frontmatter.
      Returns the highest-priority remaining match.

    Returns None → pipeline runs unchanged.
    """
    strategies = load_strategies()
    if not strategies:
        return None

    domain = "copyright" if condition.is_copyright_round else "accuracy"

    # Classify the actual problem from reviewer feedback
    problem_class = classify_failure(condition.last_feedback, condition.is_copyright_round)
    print(f"[strategy_manager] Problem classified as: '{problem_class}' "
          f"(failures={condition.consecutive_failures}, "
          f"stale={condition.stale_similarity_rounds})")

    tier1_candidates = []   # feedback-matched
    tier2_candidates = []   # round-count fallback

    for s in strategies:
        # Domain and already-tried filters (apply to both tiers)
        if s.applicable not in (domain, "both"):
            continue
        if s.id in condition.tried_strategy_ids:
            continue

        # TIER 1: known problem class + strategy's own min_consecutive_failures threshold.
        # We honour min_consecutive_failures from the frontmatter so that:
        #   - Round 1 failures always get normal (unconstrained) reviewer diagnosis.
        #   - Surgical Fix (min=2) only fires after 2 consecutive factual failures.
        #   - Visibility Boost (min=3) only fires after 3 consecutive visibility failures.
        #   - Scene Reframe (min=4) only fires after 4 consecutive scene-mismatch failures.
        # This prevents the strategy from narrowing the reviewer's output too early,
        # which was causing MORE rounds by addressing one field at a time from round 1.
        if (problem_class != "unknown"
                and s.failure_class in (problem_class, "any")
                and condition.consecutive_failures >= s.min_consecutive_failures):
            # Copyright strategies using stale_similarity need at least 1 round of
            # similarity data before they are meaningful. Without it the directive
            # would fire before any transformation attempt has produced a score.
            if s.trigger == "stale_similarity" and condition.stale_similarity_rounds < 1:
                # Defer to Tier 2 (threshold-based) — not enough data yet
                pass
            else:
                tier1_candidates.append(s)
                continue

        # TIER 2: full threshold fallback
        if _threshold_met_fallback(s, condition):
            tier2_candidates.append(s)

    # --- Tier 1: feedback-driven (mindset shift matched to identified problem) ---
    if tier1_candidates:
        # Prefer exact class match over 'any' wildcard — prevents a last-resort
        # strategy from jumping the queue solely because of a lower priority number.
        exact_matches = [s for s in tier1_candidates if s.failure_class == problem_class]
        pool = exact_matches if exact_matches else tier1_candidates
        chosen = min(pool, key=lambda s: s.priority)
        print(f"🎯 [strategy_manager] FEEDBACK-MATCHED: '{chosen.name}' "
              f"(failure_class={problem_class} → strategy.failure_class={chosen.failure_class})")
        return chosen

    # --- Tier 2: round-count fallback ---
    if tier2_candidates:
        chosen = min(tier2_candidates, key=lambda s: s.priority)
        print(f"🔄 [strategy_manager] ROUND-COUNT FALLBACK: '{chosen.name}' "
              f"(problem_class=unknown, failures={condition.consecutive_failures})")
        return chosen

    print(f"[strategy_manager] No strategy qualifies. Pipeline continues unchanged.")
    return None


# ===========================================================================
# APPLIER
# ===========================================================================

def apply_strategy(strategy: Strategy, is_copyright_round: bool) -> str:
    """
    Return the reviewer-prompt directive string for the active strategy.

    For copyright rounds → strategy.copyright_directive (into previous_context)
    For accuracy rounds  → strategy.accuracy_directive  (into accuracy prompt text)

    Returns '' if the relevant directive is not defined (safe no-op).
    """
    directive = (strategy.copyright_directive if is_copyright_round
                 else strategy.accuracy_directive)
    if not directive:
        print(f"[strategy_manager] ⚠ '{strategy.id}' has no "
              f"{'copyright' if is_copyright_round else 'accuracy'} directive.")
        return ""
    reviewer = "copyright" if is_copyright_round else "accuracy"
    print(f"   [strategy_manager] '{strategy.name}' directive → {reviewer} reviewer prompt.")
    return f"\n\n{directive}"


# ===========================================================================
# DEBUG
# ===========================================================================

def describe_strategies() -> str:
    strategies = load_strategies()
    if not strategies:
        return "No strategies loaded."
    h = f"{'ID':<35} {'NAME':<26} {'DOM':<10} {'CLASS':<16} {'TRIGGER':<22} {'PRI'}"
    rows = [h, "-" * 120]
    for s in strategies:
        rows.append(f"{s.id:<35} {s.name:<26} {s.applicable:<10} "
                    f"{s.failure_class:<16} {s.trigger:<22} {s.priority}")
    return "\n".join(rows)
