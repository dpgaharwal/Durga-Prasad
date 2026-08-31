"""Track B - Reasoning Integrity Gate."""

import re
from dataclasses import dataclass, field

INJECTION_PATTERNS = [
    r"ignore (all|previous|prior) instructions",
    r"do not (mention|tell|inform)",
    r"new instructions?:",
    r"override",
    r"system\s*:",
]
AMOUNT_TOLERANCE_PCT = 0.05


@dataclass
class GateVerdict:
    decision: str
    checks: dict = field(default_factory=dict)
    reasoning_trace: list = field(default_factory=list)


def check_evidence_decision_consistency(mandate: dict, catalog_entry: dict) -> dict:
    merchant_ok = mandate["merchant_id"] == catalog_entry["merchant_id"]
    price_diff_pct = abs(mandate["amount"] - catalog_entry["price"]) / catalog_entry["price"]
    amount_ok = price_diff_pct <= AMOUNT_TOLERANCE_PCT
    return {
        "name": "evidence_decision_consistency",
        "passed": merchant_ok and amount_ok,
        "detail": (
            f"mandate merchant_id={mandate['merchant_id']} vs catalog={catalog_entry['merchant_id']}; "
            f"mandate amount=${mandate['amount']:.2f} vs catalog=${catalog_entry['price']:.2f}"
        ),
    }


def check_injection_markers(untrusted_content: str) -> dict:
    hits = [p for p in INJECTION_PATTERNS if re.search(p, untrusted_content, re.IGNORECASE)]
    return {
        "name": "injection_marker_scan",
        "passed": len(hits) == 0,
        "detail": f"{len(hits)} injection-shaped pattern(s) found in untrusted content" if hits else "no injection patterns found",
    }


def check_rule_of_two(gate_invoked_before_execution: bool) -> dict:
    return {
        "name": "rule_of_two_checkpoint",
        "passed": gate_invoked_before_execution,
        "detail": "gate sat between untrusted-content ingestion and payment execution" if gate_invoked_before_execution else "VIOLATION: execution path bypassed the gate",
    }


def evaluate_mandate(mandate, catalog_entry, untrusted_content, gate_invoked_before_execution=True) -> GateVerdict:
    c1 = check_evidence_decision_consistency(mandate, catalog_entry)
    c2 = check_injection_markers(untrusted_content)
    c3 = check_rule_of_two(gate_invoked_before_execution)
    checks = {c["name"]: c for c in (c1, c2, c3)}
    trace = [c1["detail"], c2["detail"], c3["detail"]]

    if not c3["passed"] or not c1["passed"]:
        decision = "BLOCK"
    elif not c2["passed"]:
        decision = "ESCALATE"
    else:
        decision = "ALLOW"

    return GateVerdict(decision=decision, checks=checks, reasoning_trace=trace)
