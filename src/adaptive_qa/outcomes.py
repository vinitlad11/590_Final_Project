"""Observed finite action trees. Gold labels are added AFTER model execution."""
from dataclasses import asdict
from adaptive_qa.baseline import collect_candidate, prepare, seed_for
from adaptive_qa.config import path_for
from adaptive_qa.data.preprocess import make_states
from adaptive_qa.evaluation.answers import exact_match, answer_f1
from adaptive_qa.io import write_json, write_jsonl
from adaptive_qa.llm.answer import parse_answer
from adaptive_qa.llm.model import Usage, make_llm
from adaptive_qa.retrieval.evidence import evidence_features
from adaptive_qa.retrieval.retriever import BM25


def make_retriever(c):
    return BM25(path_for(c, c["retrieval"]["corpus"]), path_for(c, c["cache_dir"]) / "retrieval",
                c["retrieval"]["max_doc_words"])


def observe_node(llm, retriever, c, state, history="", parent=None):
    action = {"R": "RETRIEVE", "D": "DELIBERATE"}[history[-1]] if history else "ANSWER"
    documents = list(parent["documents"]) if parent else []
    retrieval_usage = Usage()
    if action == "RETRIEVE":
        # The query contains exactly the current revealed prefix, never the gold answer.
        documents, retrieval_usage = retriever.retrieve(state.question_text, c["retrieval"]["top_k"])
    candidate = collect_candidate(llm, c, state, action, parent["prediction"] if parent else "",
                                  documents if action == "RETRIEVE" else (), history)
    incremental = Usage(**candidate["usage"]) + retrieval_usage
    total = incremental + (Usage(**parent["total_usage"]) if parent else Usage())
    node = {**candidate, **asdict(state), "node_id": state.state_id + "|" + (history or "root"),
            "depth": len(history), "remaining_steps": c["controller"]["max_steps"] - len(history),
            "documents": documents, "incremental_usage": asdict(incremental), "total_usage": asdict(total),
            "retrieval_usage": asdict(retrieval_usage), "children": {},
            "previous_retrieves": history.count("R"), "previous_deliberations": history.count("D")}
    node.update(evidence_features(candidate["prediction"], documents))
    return node


def collect_tree(llm, retriever, c, state, history="", parent=None):
    node = observe_node(llm, retriever, c, state, history, parent)
    nodes = [node]
    if node["remaining_steps"]:
        for action in c["controller"]["actions"]:
            suffix = {"RETRIEVE": "R", "DELIBERATE": "D"}[action]
            branch = collect_tree(llm, retriever, c, state, history + suffix, node)
            node["children"][action] = branch[0]["node_id"]
            nodes.extend(branch)
    return nodes


def collect_outcomes(c, out):
    questions, mapping = prepare(c, out)
    llm, retriever = make_llm(c), make_retriever(c)
    rows, roots, plain_rows = [], [], []
    for qi, q in enumerate(questions, 1):
        for state in make_states(q, c["fractions"]):
            # Independent answer-only prompt: the direct baseline is not forced to buy confidence.
            plain_gen = llm.answer(state.question_text, c["model"]["temperature"], seed_for(c, state.state_id, "plain", 0))
            plain = parse_answer(plain_gen.raw, confidence=False)
            plain_record = asdict(plain_gen)
            plain_record.pop("cache_hit")
            plain_rows.append({**asdict(state), "prediction": plain.answer, "parse_valid": plain.valid,
                               "correct": exact_match(plain.answer, q.answers), "answer_f1": answer_f1(plain.answer, q.answers),
                               "split": mapping[q.question_id], "generation": plain_record})
            nodes = collect_tree(llm, retriever, c, state)
            roots.append(nodes[0]["node_id"])
            for node in nodes:
                node.update({"ground_truth": list(q.answers), "correct": exact_match(node["prediction"], q.answers),
                             "answer_f1": answer_f1(node["prediction"], q.answers), "split": mapping[q.question_id],
                             "experiment_id": c["experiment_id"]})
            rows.extend(nodes)
        if qi % 10 == 0 or qi == len(questions):
            print(f"Collected {qi}/{len(questions)} questions; {llm.new_calls} new generation calls, {llm.hits} cache hits", flush=True)
    write_jsonl(out / "action_nodes.jsonl", rows)
    write_jsonl(out / "direct_predictions.jsonl", plain_rows)
    write_json(out / "root_ids.json", roots)
    write_json(out / "collection_usage.json", {**llm.usage(), "retrieval_new_calls": retriever.new_calls,
               "retrieval_cache_hits": retriever.cache_hits, "status": c["dataset"]["name"],
               "nodes": len(rows), "states": len(roots), "corpus": retriever.identity})
    return rows, roots, plain_rows
