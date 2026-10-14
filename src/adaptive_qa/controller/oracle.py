"""Hindsight upper bound for the sampled finite tree, not an achievable policy."""
from adaptive_qa.llm.model import Usage

ACTION_ORDER = ("ANSWER", "ABSTAIN", "DELIBERATE", "RETRIEVE")


def available_actions(node):
    actions = ["ANSWER"] if node["parse_valid"] else []
    actions.append("ABSTAIN")
    if node["remaining_steps"] > 0:
        actions.extend(a for a in ("DELIBERATE", "RETRIEVE") if a in node["children"])
    return actions


def best_action(values, available=None):
    allowed = set(available if available is not None else values)
    return max((a for a in ACTION_ORDER if a in allowed), key=lambda a: values[a])


def oracle_values(nodes, root_id, cost_model, lambda_, memo=None):
    memo = {} if memo is None else memo
    if root_id in memo:
        return memo[root_id]
    node = nodes[root_id]
    values = {"ABSTAIN": cost_model.reward(answered=False)}
    if node["parse_valid"]:
        values["ANSWER"] = cost_model.reward(node["correct"])
    for action, child_id in node["children"].items():
        child_values = oracle_values(nodes, child_id, cost_model, lambda_, memo)
        values[action] = max(child_values.values()) - lambda_ * cost_model.cost(Usage(**nodes[child_id]["incremental_usage"]))
    memo[root_id] = values
    return values


def immediate_action_values(nodes, node_id, cost_model, lambda_):
    """Training targets: observed action, then answer. No clairvoyant abstention.

    Keep these separate from the hindsight tree bound: clipping a failed action's
    reward to zero would teach a systematically optimistic value estimate.
    """
    node = nodes[node_id]
    values = {"ABSTAIN":cost_model.reward(answered=False)}
    if node["parse_valid"]:
        values["ANSWER"] = cost_model.reward(node["correct"])
    for action, child_id in node["children"].items():
        child = nodes[child_id]
        values[action] = cost_model.utility(child["correct"], True, Usage(**child["incremental_usage"]), lambda_)
    return values


def action_outcome_rows(nodes, root_ids, cost_model, lambda_):
    """Both immediate answer-after-action outcomes and finite-horizon oracle targets."""
    memo, rows = {}, []
    for root in root_ids:
        oracle_values(nodes, root, cost_model, lambda_, memo)
    for node_id, node in nodes.items():
        q = memo[node_id]
        for action in available_actions(node):
            resulting = nodes[node["children"][action]] if action in node["children"] else node
            extra = Usage(**resulting["incremental_usage"]) if action in node["children"] else Usage()
            answered = action != "ABSTAIN"
            rows.append({"question_id": node["question_id"], "state_id": node["state_id"], "node_id": node_id,
                         "split": node["split"], "action": action, "resulting_answer": resulting["prediction"] if answered else None,
                         "correct": resulting["correct"] if answered else None, "additional_usage": extra.__dict__,
                         "cost": cost_model.cost(extra), "reward": cost_model.reward(resulting["correct"], answered),
                         "immediate_utility": cost_model.utility(resulting["correct"], answered, extra, lambda_),
                         "oracle_q": q[action], "oracle_action": best_action(q), "lambda": lambda_})
    return rows
