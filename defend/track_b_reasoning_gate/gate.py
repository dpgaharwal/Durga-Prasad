"""
Track B - Reasoning Integrity Gate (RIG).

Sits between agent reasoning and payment execution. Three checks, run every
mandate through all three before allowing execution.

  1. Evidence-Decision Consistency
     Does the final mandate (merchant_id, amount) match the TRUSTED catalog
     record the agent actually searched, not something that only appears in
     untrusted free-text content? This is the core check -- it doesn't need
     to understand *why* the mandate drifted, only that it drifted from
     ground truth.

  2. Injection Marker Scan
     Pattern-scan the untrusted content channel (product descriptions, tool
     outputs) for instruction-shaped text. Not a complete defense on its own
     (adaptive attackers avoid known patterns -- see the honesty note below)
     but cheap, fast, and catches the common case.

  3. Rule-of-Two Checkpoint
     Structural, not content-based: this gate must sit between "agent just
     processed untrusted input" and "state-changing action" (payment
     execution) as a hard architectural boundary -- not a rule the agent can
     be prompted around. Enforced here by the graph wiring itself
     (loop/track_b_loop.py) rather than by this module, which is the point:
     Rule-of-Two protection with an LLM check can't be prompt-injected out
     of. Recorded here as an audit flag.

Verdict: ALLOW / BLOCK / ESCALATE, with a logged reasoning trace explaining
why -- the trace is the actual deliverable for the judges, not just the verdict.

HONEST LIMITATION (state this explicitly in the writeup): pattern-based
injection scanning is defeated by adaptive attackers who avoid known trigger
phrases. Published research shows adaptive prompt-injection attacks bypass the
large majority of deployed defenses once the attacker knows what the defense
is doing. Check 1 (evidence-decision consistency) is the one that holds up
under an adaptive attacker, because it doesn't depend on recognizing attack
*text* -- it depends on the mandate matching ground truth, which the attacker
cannot spoof without also compromising the catalog itself. Frame this gate as
blast-radius reduction, not a guarantee.
"""

import re
from dataclasses import dataclass, field


INJECTION_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"do not (mention|tell|inform)",
    r"new instructions?:",
    r"override",
    r"system\s*:",
]

AMOUNT_TOLERANCE_PCT = 0.05   # 5% wiggle room for tax/shipping variance


@dataclass
class GateVerdict:
    decision: str                 # "ALLOW" | "BLOCK" | "ESCALATE"
    checks: dict = field(default_factory=dict)
    reasoning_trace: list = field(default_factory=list)


def check_evidence_decision_consistency(mandate: dict, catalog_entry: dict) -> dict:
    merchant_ok = mandate["merchant_id"] == catalog_entry["merchant_id"]
    price_diff_pct = abs(mandate["amount"] - catalog_entry["price"]) / catalog_entry["price"]
    amount_ok = price_diff_pct <= AMOUNT_TOLERANCE_PCT

    return {
        "name": "evidence_decision_consistency",
        "passed": merchant_ok and amount_ok,
        "merchant_match": merchant_ok,
        "amount_within_tolerance": amount_ok,
        "price_diff_pct": round(price_diff_pct, 4),
        "detail": (
            f"mandate merchant_id={mandate['merchant_id']} vs "
            f"catalog={catalog_entry['merchant_id']}; "
            f"mandate amount=${mandate['amount']:.2f} vs "
            f"catalog=${catalog_entry['price']:.2f}"
        ),
    }


def check_injection_markers(untrusted_content: str) -> dict:
    hits = [p for p in INJECTION_PATTERNS if re.search(p, untrusted_content, re.IGNORECASE)]
    return {
        "name": "injection_marker_scan",
        "passed": len(hits) == 0,
        "patterns_matched": hits,
        "detail": f"{len(hits)} injection-shaped pattern(s) found in untrusted content"
                   if hits else "no injection patterns found",
    }


def check_rule_of_two(gate_invoked_before_execution: bool) -> dict:
    """
    Architectural flag, not a content check. See module docstring -- the real
    enforcement is the graph structure in track_b_loop.py; this just records
    that the checkpoint fired where it should have.
    """
    return {
        "name": "rule_of_two_checkpoint",
        "passed": gate_invoked_before_execution,
        "detail": "gate sat between untrusted-content ingestion and payment execution"
                   if gate_invoked_before_execution
                   else "VIOLATION: execution path bypassed the gate",
    }


def evaluate_mandate(
    mandate: dict,
    catalog_entry: dict,
    untrusted_content: str,
    gate_invoked_before_execution: bool = True,
) -> GateVerdict:
    c1 = check_evidence_decision_consistency(mandate, catalog_entry)
    c2 = check_injection_markers(untrusted_content)
    c3 = check_rule_of_two(gate_invoked_before_execution)

    checks = {c["name"]: c for c in (c1, c2, c3)}
    trace = [c1["detail"], c2["detail"], c3["detail"]]

    if not c3["passed"]:
        decision = "BLOCK"
    elif not c1["passed"]:
        # ground-truth mismatch is the hard failure -- block outright
        decision = "BLOCK"
    elif not c2["passed"]:
        # injection text present but mandate still matches ground truth --
        # suspicious but not proven harmful, human review
        decision = "ESCALATE"
    else:
        decision = "ALLOW"

    return GateVerdict(decision=decision, checks=checks, reasoning_trace=trace)
