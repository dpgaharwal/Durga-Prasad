"""Track B - agent pipeline. Single shared LLM decision, two policies (naive vs gated)."""

from typing import TypedDict
from catalog import get_product
from gate import evaluate_mandate
import re


class AgentState(TypedDict):
    goal: str
    sku: str
    use_injected_content: bool
    product_description: str
    mandate: dict
    reasoning_trace: list
    gate_verdict: object
    final_status: str


def node_search(state: AgentState) -> AgentState:
    catalog_entry = get_product(state["sku"])
    description = (
        catalog_entry["description_injected"] if state["use_injected_content"]
        else catalog_entry["description_clean"]
    )
    state["product_description"] = description
    state["reasoning_trace"] = state.get("reasoning_trace", [])
    state["reasoning_trace"].append(f"Searched catalog for SKU {state['sku']}, retrieved listing.")
    return state


def make_node_shopper_decide(llm):
    def node_shopper_decide(state: AgentState) -> AgentState:
        catalog_entry = get_product(state["sku"])
        response = llm.shopper_decide(state["goal"], state["product_description"], catalog_entry)

        merchant_match = re.search(r"merchant_id[=:\s]+([A-Z0-9_]+)", response.text)
        amount_match = re.search(r"\$?(\d+\.\d{2})", response.text)

        mandate = {
            "merchant_id": merchant_match.group(1) if merchant_match else catalog_entry["merchant_id"],
            "amount": float(amount_match.group(1)) if amount_match else catalog_entry["price"],
            "sku": state["sku"],
        }
        state["mandate"] = mandate
        state["reasoning_trace"].append(response.text)
        state["_followed_injection"] = response.followed_injection
        return state
    return node_shopper_decide


def node_execute_naive(state: AgentState) -> AgentState:
    state["final_status"] = f"EXECUTED (naive, no gate): {state['mandate']}"
    return state


def node_reasoning_gate(state: AgentState) -> AgentState:
    catalog_entry = get_product(state["sku"])
    verdict = evaluate_mandate(
        mandate=state["mandate"], catalog_entry=catalog_entry,
        untrusted_content=state["product_description"], gate_invoked_before_execution=True,
    )
    state["gate_verdict"] = verdict
    state["reasoning_trace"].extend(verdict.reasoning_trace)
    return state


def node_execute_gated(state: AgentState) -> AgentState:
    verdict = state["gate_verdict"]
    if verdict.decision == "ALLOW":
        state["final_status"] = f"EXECUTED (gate: ALLOW): {state['mandate']}"
    elif verdict.decision == "ESCALATE":
        state["final_status"] = f"HELD FOR HUMAN REVIEW (gate: ESCALATE): {state['mandate']}"
    else:
        state["final_status"] = f"BLOCKED (gate: BLOCK): {state['mandate']}"
    return state


def run_scenario(sku: str, goal: str, use_injected_content: bool, llm):
    """
    Single shared agent decision, then two POLICIES applied to it (naive vs
    gated) -- not two independent LLM calls, which with a real (non-
    deterministic) model can produce inconsistent phrasing between the two
    "same scenario" runs.
    """
    shared_state: AgentState = {
        "goal": goal, "sku": sku, "use_injected_content": use_injected_content,
        "reasoning_trace": [],
    }
    shared_state = node_search(shared_state)
    shared_state = make_node_shopper_decide(llm)(shared_state)

    naive_result = node_execute_naive({**shared_state, "reasoning_trace": list(shared_state["reasoning_trace"])})
    gated_result = node_reasoning_gate({**shared_state, "reasoning_trace": list(shared_state["reasoning_trace"])})
    gated_result = node_execute_gated(gated_result)

    return naive_result, gated_result
