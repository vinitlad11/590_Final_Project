"""Descriptive pilot diagnostics and Stage 4/5 experimental evaluations."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import json
import numpy as np
from adaptive_qa.controller.oracle import oracle_values, best_action
from adaptive_qa.controller.utility import (
    CostModel,
    LearnedUtilityController,
    StateCalibrator,
    enrich_root_and_child,
    raw_confidence_proxy,
    valid_entity_flag,
)
from adaptive_qa.evaluation.answers import exact_match
from adaptive_qa.evaluation.metrics import calibration_metrics, risk_coverage, selective_metrics
from adaptive_qa.io import digest, read_jsonl, write_json
from adaptive_qa.llm.answer import parse_answer
from adaptive_qa.llm.model import Usage
from adaptive_qa.reproducibility import check_manifest


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def placeholder_flag(answer):
    # A review flag only. It never changes the parser, gold aliases, or scores.
    s = answer.strip().casefold()
    if s in {'short answer', 'your answer', 'answer here', 'candidate name', 'unknown', 'n/a'}:
        return True
    return s.startswith(('this ', 'these '))


def sample_records(rows, direct):
    roots = {r['state_id']: r for r in rows if not r.get('history')}
    records = []
    for row in rows + direct:
        is_direct = 'generation' in row
        kind = 'direct' if is_direct else ('root' if not row.get('history') else row['history'])
        gold = roots[row['state_id']]['ground_truth']
        generations = [row['generation']] if is_direct else row['generations']
        with_confidence = False if is_direct else row.get('response_has_confidence', True)
        for i, g in enumerate(generations):
            parsed = parse_answer(g['raw'], confidence=with_confidence)
            records.append({
                'question_id': row['question_id'], 'state_id': row['state_id'],
                'requested_fraction': row['requested_fraction'],
                'fraction_revealed': row['fraction_revealed'], 'observation': kind,
                'sample_index': i, 'selected_answer': i == 0,
                'prediction': parsed.answer, 'parse_valid': parsed.valid,
                'placeholder_flag': placeholder_flag(parsed.answer),
                'correct': exact_match(parsed.answer, gold), 'ground_truth': ' | '.join(gold),
                'raw_output': g['raw'], **g['usage'],
            })
    return records


def invocation_totals(out):
    totals = {'attempted_backend_calls': 0, 'successful_calls': 0, 'failed_calls': 0,
              'pending_calls': 0, 'cache_hits': 0, 'finished_wall_seconds': 0.0,
              'unfinished_invocations': 0, 'measured_successful_usage': asdict(Usage())}
    usage = Usage()
    for start in sorted((out / 'sessions').glob('*/start.json')):
        folder = start.parent
        ledger = folder / 'model_usage.json'
        if ledger.exists():
            row = json.loads(ledger.read_text())
            totals['attempted_backend_calls'] += row['new_calls']
            for k in ('successful_calls', 'failed_calls', 'pending_calls', 'cache_hits'):
                totals[k] += row.get(k, 0)
            usage += Usage(**row['incurred_usage'])
        finish = folder / 'finish.json'
        if finish.exists():
            totals['finished_wall_seconds'] += json.loads(finish.read_text())['wall_seconds']
        else:
            totals['unfinished_invocations'] += 1
    totals['measured_successful_usage'] = asdict(usage)
    totals['scope'] = 'All retained invocation ledgers for this experiment only. Failed/pending tokens and interrupted wall time may be unknown. Other experiments and Stage 2 are additional work.'
    return totals


def pilot_report(c, out):
    out = Path(out)
    check_manifest(c, out)
    collected = (out / 'action_nodes.jsonl').exists()
    rows = read_jsonl(out / ('action_nodes.jsonl' if collected else 'baseline_predictions.jsonl'))
    roots = [r for r in rows if not r.get('history')]
    nodes = {r['node_id']: r for r in rows} if collected else {}
    direct = read_jsonl(out / 'direct_predictions.jsonl') if collected else []
    direct_by_state = {r['state_id']: r for r in direct}
    if not roots:
        raise ValueError('No pilot observations found')
    totals = Usage()
    for row in rows:
        totals += Usage(**row['incremental_usage' if collected else 'usage'])
    for row in direct:
        totals += Usage(**row['generation']['usage'])
    prefix_rows = []
    for fraction in sorted({r['requested_fraction'] for r in roots}):
        subset = [r for r in roots if r['requested_fraction'] == fraction]
        repeats = [r for r in subset if len(r['generations']) >= 2]
        stable = [r for r in repeats if r['parse_valid'] and r['self_consistency'] == 1
                  and all(parse_answer(g['raw'], confidence=r.get('response_has_confidence', True)).valid
                          for g in r['generations'])]
        pairs = [(r, nodes[r['children']['RETRIEVE']]) for r in subset
                 if collected and 'RETRIEVE' in r['children']]
        plain = [direct_by_state[r['state_id']] for r in subset if r['state_id'] in direct_by_state]
        prefix_rows.append({
            'requested_fraction': fraction, 'states': len(subset),
            'actual_fraction_min': min(r['fraction_revealed'] for r in subset),
            'actual_fraction_max': max(r['fraction_revealed'] for r in subset),
            'parse_valid_rate': sum(bool(r['parse_valid']) for r in subset) / len(subset),
            'root_correct_answers': sum(bool(r['correct']) for r in subset),
            'correct_answer_rate': sum(bool(r['correct']) for r in subset) / len(subset),
            'direct_states': len(plain),
            'direct_correct_answers': sum(bool(r['correct']) for r in plain),
            'direct_correct_rate': sum(bool(r['correct']) for r in plain) / len(plain) if plain else None,
            'repeat_states': len(repeats), 'all_samples_agree_states': len(stable),
            'stable_but_wrong_states': sum(not r['correct'] for r in stable),
            'retrieval_pairs': len(pairs),
            'retrieved_correct_answers': sum(bool(child['correct']) for _, child in pairs),
            'retrieved_correct_rate': sum(bool(child['correct']) for _, child in pairs) / len(pairs) if pairs else None,
            'retrieval_corrected_errors': sum(not r['correct'] and child['correct'] for r, child in pairs),
            'retrieval_introduced_errors': sum(r['correct'] and not child['correct'] for r, child in pairs),
        })
    samples = sample_records(rows, direct)
    retrievals = [{
        'question_id': r['question_id'], 'state_id': r['state_id'],
        'requested_fraction': r['requested_fraction'], 'rank': d['rank'],
        'revealed_question': r['question_text'], 'prediction': r['prediction'],
        'ground_truth': ' | '.join(r.get('ground_truth') or {rt['state_id']: rt for rt in roots}[r['state_id']]['ground_truth']),
        'document_id': d['document_id'], 'text': d['text'],
    } for r in rows if r.get('history') == 'R' for d in r.get('documents', [])]
    tables = out / 'tables'
    write_csv(tables / 'pilot_by_prefix.csv', prefix_rows)
    write_csv(tables / 'sample_review.csv', samples)
    write_csv(tables / 'retrieval_review.csv', retrievals)
    placeholders = sum(r['placeholder_flag'] for r in samples)
    invalid = sum(not r['parse_valid'] for r in samples)
    warnings = []
    if placeholders:
        warnings.append(f'{placeholders} sample answers matched literal prompt placeholder text; inspect prompts before expanding.')
    if invalid:
        warnings.append(f'{invalid} outputs failed parsing; inspect errors without dropping them from denominators.')
    if sum(r['root_correct_answers'] for r in prefix_rows) == 0:
        warnings.append('No correct root answers in this run. Inspect prompt behavior, scoring and evidence coverage; this is not evidence that the workflow failed or that a controller helps.')
    actual = invocation_totals(out)
    if actual['unfinished_invocations']:
        warnings.append(f"{actual['unfinished_invocations']} command session(s) did not record completion.")
    development = c['model']['backend'] == 'fixture' or 'PILOT' in c['dataset']['name'].upper()
    report = {
        'experiment_id': c['experiment_id'], 'dataset': c['dataset']['name'],
        'analysis_role': 'development_pilot_descriptive' if development else 'collection_descriptive_not_final_evaluation',
        'fixture_only': c['model']['backend'] == 'fixture',
        'calibration_fitted_by_report': False,
        'prompt_version': c['model'].get('prompt_version', 'qa-v1'),
        'interpretation': 'Repeated generations measure consistency under the recorded sampling settings, not correctness probability. The same wrong answer can repeat. No significance claims are made.',
        'split_note': 'All pilot questions are development data, including partitions called calibration, tune or test. Calibration is disabled. These questions cannot later be an untouched holdout.',
        'questions': len({r['question_id'] for r in roots}), 'root_states': len(roots),
        'observed_nodes': len(rows), 'retrieval_collected': collected,
        'samples_reviewed': len(samples), 'placeholder_samples': placeholders,
        'invalid_samples': invalid,
        'valid_sample_json_rate': (len(samples) - invalid) / len(samples),
        'parser_note': 'The parser accepts a JSON object wrapped in a Markdown code fence. Placeholder flags do not change correctness scoring.',
        'logical_collection_usage': asdict(totals),
        'usage_scope': 'All node samples plus the independent direct sample. Repeated baseline files are not double-counted.' if collected else 'Baseline samples only; no direct sample or retrieval collected yet.',
        'time_note': 'Logical latency sums original component times even on cache hits; it is not current wall time. Invocation wall times include model loading but not notebook installation or data transfer.',
        'actual_invocations': actual,
        'by_prefix': prefix_rows, 'review_warnings': warnings,
        'new_model_calls_from_reporting': 0,
    }
    write_json(out / 'pilot_report.json', report)
    label = 'SOFTWARE FIXTURE ONLY' if report['fixture_only'] else 'DEVELOPMENT PILOT; DESCRIPTIVE'
    md = [f"# Pilot observations\n\n{label}\n",
          f"Questions: {report['questions']}; root states: {len(roots)}; samples: {len(samples)}.\n",
          report['interpretation'] + "\n",
          f"Placeholder flags: {placeholders}. Invalid samples: {invalid}.\n",
          "| Prefix | States | Root correct | Direct correct | Retrieved correct | Stable but wrong |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in prefix_rows:
        md.append(f"| {r['requested_fraction']:.0%} | {r['states']} | {r['root_correct_answers']} | "
                  f"{r['direct_correct_answers'] if collected else 'n/a'} | "
                  f"{r['retrieved_correct_answers'] if collected else 'n/a'} | {r['stable_but_wrong_states']} |")
    md.append(f"\nLogical generation calls: {totals.llm_calls}. Logical retrieval calls: {totals.retrieval_calls}.\n")
    md.append(report['time_note'] + "\n")
    md.append(report['split_note'] + "\n")
    if warnings:
        md.append("\n".join("- " + w for w in warnings) + "\n")
    md.append("Tables: pilot_by_prefix.csv, sample_review.csv, and retrieval_review.csv (after collection).\n")
    md.append("This report does not train a controller or evaluate a deployed abstention policy.\n")
    (out / 'pilot_report.md').write_text("\n".join(md), encoding='utf-8')
    return report


def _group_kfold_pairs(pairs, n_splits=5, seed=42):
    qids = sorted({r["question_id"] for r, _ in pairs}, key=lambda q: digest([seed, q]))
    folds = [set(qids[i::n_splits]) for i in range(n_splits)]
    result = []
    for i in range(n_splits):
        val_qids = folds[i]
        train_pairs = [(r, c) for r, c in pairs if r["question_id"] not in val_qids]
        val_pairs = [(r, c) for r, c in pairs if r["question_id"] in val_qids]
        result.append((train_pairs, val_pairs))
    return result


def evaluate_calibration_rq1(pairs, n_splits=5, seed=42):
    y_root, raw_root, cal_root = [], [], []
    y_ret, raw_ret, cal_ret = [], [], []
    for train_pairs, val_pairs in _group_kfold_pairs(pairs, n_splits=n_splits, seed=seed):
        train_roots = [enrich_root_and_child(r, c) for r, c in train_pairs]
        train_rets = [c for _, c in train_pairs]
        rcal = StateCalibrator(include_evidence=True).fit(train_roots, is_retrieved=False)
        ccal = StateCalibrator(include_evidence=True).fit(train_rets, is_retrieved=True)
        for r, c in val_pairs:
            er = enrich_root_and_child(r, c)
            y_root.append(int(bool(r["correct"])))
            raw_root.append(raw_confidence_proxy(r))
            cal_root.append(rcal.predict_proba(er, is_retrieved=False))
            y_ret.append(int(bool(c["correct"])))
            raw_ret.append(raw_confidence_proxy(c))
            cal_ret.append(ccal.predict_proba(c, is_retrieved=True))

    y_all = y_root + y_ret
    raw_all = raw_root + raw_ret
    cal_all = cal_root + cal_ret
    return {
        "root_states": {
            "raw_proxy": {**calibration_metrics(y_root, raw_root, n_bins=5), "aurc": risk_coverage(y_root, raw_root)["aurc"]},
            "calibrated": {**calibration_metrics(y_root, cal_root, n_bins=5), "aurc": risk_coverage(y_root, cal_root)["aurc"]},
        },
        "post_retrieval_states": {
            "raw_proxy": {**calibration_metrics(y_ret, raw_ret, n_bins=5), "aurc": risk_coverage(y_ret, raw_ret)["aurc"]},
            "calibrated": {**calibration_metrics(y_ret, cal_ret, n_bins=5), "aurc": risk_coverage(y_ret, cal_ret)["aurc"]},
        },
        "combined_all_nodes": {
            "raw_proxy": {**calibration_metrics(y_all, raw_all, n_bins=5), "aurc": risk_coverage(y_all, raw_all)["aurc"]},
            "calibrated": {**calibration_metrics(y_all, cal_all, n_bins=5), "aurc": risk_coverage(y_all, cal_all)["aurc"]},
            "risk_coverage_curve_calibrated": risk_coverage(y_all, cal_all)["points"],
            "risk_coverage_curve_raw": risk_coverage(y_all, raw_all)["points"],
        },
    }


def _summarize_decisions(policy_name, decisions, lambda_, reward_wrong):
    n = len(decisions)
    answered = [bool(d["answered"]) for d in decisions]
    correct = [int(bool(d["correct"])) for d in decisions]
    sel = selective_metrics(correct, answered)
    tot_input = np.mean([d["total_usage"].input_tokens for d in decisions])
    tot_output = np.mean([d["total_usage"].output_tokens for d in decisions])
    tot_llm = np.mean([d["total_usage"].llm_calls for d in decisions])
    tot_ret = np.mean([d["total_usage"].retrieval_calls for d in decisions])
    tot_lat = np.mean([d["total_usage"].latency_seconds for d in decisions])
    add_tokens = np.mean([d["additional_usage"].input_tokens + d["additional_usage"].output_tokens for d in decisions])
    mean_util = float(np.mean([d["utility"] for d in decisions]))
    mean_util_sunk = float(np.mean([d["total_utility_with_sunk"] for d in decisions]))
    retrieve_rate = float(np.mean([str(d["action_taken"]).startswith("RETRIEVE") for d in decisions]))
    abstain_rate = float(1.0 - sel["coverage"])
    return {
        "policy": policy_name,
        "lambda": float(lambda_),
        "incorrect_penalty": float(reward_wrong),
        "states": n,
        "overall_accuracy": round(float(sel["exact_match_all_states"]), 4),
        "selective_accuracy": round(float(sel["selective_accuracy"]), 4) if sel["selective_accuracy"] is not None else 0.0,
        "coverage": round(float(sel["coverage"]), 4),
        "abstain_rate": round(abstain_rate, 4),
        "retrieve_rate": round(retrieve_rate, 4),
        "mean_utility": round(mean_util, 4),
        "mean_utility_with_root_cost": round(mean_util_sunk, 4),
        "avg_llm_calls": round(float(tot_llm), 3),
        "avg_retrieval_calls": round(float(tot_ret), 3),
        "avg_total_tokens": round(float(tot_input + tot_output), 1),
        "avg_additional_tokens": round(float(add_tokens), 1),
        "avg_latency_seconds": round(float(tot_lat), 3),
    }


def run_all_policies_cv(pairs, direct_by_state, nodes_by_id, cost_config, reward_config, lambda_=0.1, n_splits=5, seed=42):
    cm = CostModel(cost_config, reward_config)
    policy_decisions = defaultdict(list)

    for train_pairs, val_pairs in _group_kfold_pairs(pairs, n_splits=n_splits, seed=seed):
        ctrl = LearnedUtilityController().fit(train_pairs, cm)
        for r, c in val_pairs:
            r_usage = Usage(**r["incremental_usage"])
            c_inc_usage = Usage(**c["incremental_usage"])
            c_tot_usage = r_usage + c_inc_usage

            d_row = direct_by_state.get(r["state_id"])
            if d_row:
                d_usage = Usage(**d_row["generation"]["usage"])
                d_cor = int(bool(d_row["correct"]))
                policy_decisions["1. Direct LLM Answering"].append({
                    "action_taken": "DIRECT", "answered": True, "prediction": d_row["prediction"], "correct": d_cor,
                    "total_usage": d_usage, "additional_usage": Usage(),
                    "utility": cm.utility(bool(d_cor), True, Usage(), lambda_),
                    "total_utility_with_sunk": cm.utility(bool(d_cor), True, d_usage, lambda_),
                })

            r_cor = int(bool(r["correct"]))
            policy_decisions["2. Root Answering (No Retrieval)"].append({
                "action_taken": "ANSWER", "answered": True, "prediction": r["prediction"], "correct": r_cor,
                "total_usage": r_usage, "additional_usage": Usage(),
                "utility": cm.utility(bool(r_cor), True, Usage(), lambda_),
                "total_utility_with_sunk": cm.utility(bool(r_cor), True, r_usage, lambda_),
            })

            c_cor = int(bool(c["correct"]))
            policy_decisions["3. Always-Retrieve RAG"].append({
                "action_taken": "RETRIEVE->ANSWER", "answered": True, "prediction": c["prediction"], "correct": c_cor,
                "total_usage": c_tot_usage, "additional_usage": c_inc_usage,
                "utility": cm.utility(bool(c_cor), True, c_inc_usage, lambda_),
                "total_utility_with_sunk": cm.utility(bool(c_cor), True, c_tot_usage, lambda_),
            })

            raw_p = raw_confidence_proxy(r)
            if valid_entity_flag(r) and raw_p >= 0.80:
                policy_decisions["4. Fixed Confidence Threshold"].append({
                    "action_taken": "ANSWER", "answered": True, "prediction": r["prediction"], "correct": r_cor,
                    "total_usage": r_usage, "additional_usage": Usage(),
                    "utility": cm.utility(bool(r_cor), True, Usage(), lambda_),
                    "total_utility_with_sunk": cm.utility(bool(r_cor), True, r_usage, lambda_),
                })
            else:
                policy_decisions["4. Fixed Confidence Threshold"].append({
                    "action_taken": "ABSTAIN", "answered": False, "prediction": None, "correct": 0,
                    "total_usage": r_usage, "additional_usage": Usage(),
                    "utility": cm.utility(False, False, Usage(), lambda_),
                    "total_utility_with_sunk": cm.utility(False, False, r_usage, lambda_),
                })

            if valid_entity_flag(r) and r.get("self_consistency", 0.0) >= 1.0 and r.get("fraction_revealed", 0.5) >= 1.0:
                policy_decisions["5. Low-Confidence Triggered RAG"].append({
                    "action_taken": "ANSWER", "answered": True, "prediction": r["prediction"], "correct": r_cor,
                    "total_usage": r_usage, "additional_usage": Usage(),
                    "utility": cm.utility(bool(r_cor), True, Usage(), lambda_),
                    "total_utility_with_sunk": cm.utility(bool(r_cor), True, r_usage, lambda_),
                })
            else:
                policy_decisions["5. Low-Confidence Triggered RAG"].append({
                    "action_taken": "RETRIEVE->ANSWER", "answered": True, "prediction": c["prediction"], "correct": c_cor,
                    "total_usage": c_tot_usage, "additional_usage": c_inc_usage,
                    "utility": cm.utility(bool(c_cor), True, c_inc_usage, lambda_),
                    "total_utility_with_sunk": cm.utility(bool(c_cor), True, c_tot_usage, lambda_),
                })

            policy_decisions["6. Learned Utility Controller (Proposed)"].append(
                ctrl.decide_and_execute(r, c, cm, lambda_)
            )

            ov = oracle_values(nodes_by_id, r["node_id"], cm, lambda_)
            oa = best_action(ov)
            if oa == "ANSWER":
                policy_decisions["7. Oracle Upper Bound"].append({
                    "action_taken": "ANSWER", "answered": True, "prediction": r["prediction"], "correct": r_cor,
                    "total_usage": r_usage, "additional_usage": Usage(),
                    "utility": cm.utility(bool(r_cor), True, Usage(), lambda_),
                    "total_utility_with_sunk": cm.utility(bool(r_cor), True, r_usage, lambda_),
                })
            elif oa == "ABSTAIN":
                policy_decisions["7. Oracle Upper Bound"].append({
                    "action_taken": "ABSTAIN", "answered": False, "prediction": None, "correct": 0,
                    "total_usage": r_usage, "additional_usage": Usage(),
                    "utility": cm.utility(False, False, Usage(), lambda_),
                    "total_utility_with_sunk": cm.utility(False, False, r_usage, lambda_),
                })
            else:
                if c_cor:
                    policy_decisions["7. Oracle Upper Bound"].append({
                        "action_taken": "RETRIEVE->ANSWER", "answered": True, "prediction": c["prediction"], "correct": 1,
                        "total_usage": c_tot_usage, "additional_usage": c_inc_usage,
                        "utility": cm.utility(True, True, c_inc_usage, lambda_),
                        "total_utility_with_sunk": cm.utility(True, True, c_tot_usage, lambda_),
                    })
                else:
                    policy_decisions["7. Oracle Upper Bound"].append({
                        "action_taken": "RETRIEVE->ABSTAIN", "answered": False, "prediction": None, "correct": 0,
                        "total_usage": c_tot_usage, "additional_usage": c_inc_usage,
                        "utility": cm.utility(False, False, c_inc_usage, lambda_),
                        "total_utility_with_sunk": cm.utility(False, False, c_tot_usage, lambda_),
                    })

    summaries = [
        _summarize_decisions(name, policy_decisions[name], lambda_, reward_config["incorrect"])
        for name in sorted(policy_decisions.keys())
    ]
    return summaries, policy_decisions


def run_ablation_study(pairs, cost_config, reward_config, lambda_=0.1, n_splits=5, seed=42):
    cm = CostModel(cost_config, reward_config)
    variants = [
        ("Full Learned Controller", None),
        ("w/o Confidence Calibration", "confidence"),
        ("w/o Self-Consistency", "consistency"),
        ("w/o Evidence / BM25 Support", "evidence"),
        ("w/o Cost Weighting", "cost"),
        ("w/o Question Progress", "progress"),
    ]
    rows = []
    full_util = None
    for label, dropped in variants:
        decisions = []
        for train_pairs, val_pairs in _group_kfold_pairs(pairs, n_splits=n_splits, seed=seed):
            ctrl = LearnedUtilityController(dropped_group=dropped).fit(train_pairs, cm)
            for r, c in val_pairs:
                decisions.append(ctrl.decide_and_execute(r, c, cm, lambda_))
        summary = _summarize_decisions(label, decisions, lambda_, reward_config["incorrect"])
        if full_util is None:
            full_util = summary["mean_utility"]
        summary["dropped_group"] = dropped or "none"
        summary["utility_delta_vs_full"] = round(summary["mean_utility"] - full_util, 4)
        rows.append(summary)
    return rows


def generate_figures(out_dir, rq1_cal, policy_table, lambda_sweep_rows, ablation_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})

    # Figure 1: Reliability Diagram & Calibration Comparison (RQ1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=180)
    raw_bins = rq1_cal["combined_all_nodes"]["raw_proxy"]["bins"]
    cal_bins = rq1_cal["combined_all_nodes"]["calibrated"]["bins"]
    axes[0].plot([0, 1], [0, 1], "--", color="#666666", label="Perfect Calibration")
    if raw_bins:
        axes[0].plot([b["confidence"] for b in raw_bins], [b["accuracy"] for b in raw_bins],
                     "o-", color="#d95f02", linewidth=2, label=f"Uncalibrated Proxy (ECE={rq1_cal['combined_all_nodes']['raw_proxy']['ece']:.3f})")
    if cal_bins:
        axes[0].plot([b["confidence"] for b in cal_bins], [b["accuracy"] for b in cal_bins],
                     "s-", color="#1b9e77", linewidth=2, label=f"Calibrated Logistic (ECE={rq1_cal['combined_all_nodes']['calibrated']['ece']:.3f})")
    axes[0].set_title("RQ1: Reliability Diagram (All Observed States)")
    axes[0].set_xlabel("Predicted Probability of Correctness")
    axes[0].set_ylabel("Empirical Answer Accuracy")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8.5)

    metrics_names = ["ECE (↓ better)", "Brier Score (↓ better)", "1 - AUROC (↓ better)"]
    raw_m = rq1_cal["combined_all_nodes"]["raw_proxy"]
    cal_m = rq1_cal["combined_all_nodes"]["calibrated"]
    raw_vals = [raw_m["ece"], raw_m["brier"], 1.0 - (raw_m["auroc"] or 0.5)]
    cal_vals = [cal_m["ece"], cal_m["brier"], 1.0 - (cal_m["auroc"] or 0.5)]
    x = np.arange(len(metrics_names))
    w = 0.35
    axes[1].bar(x - w / 2, raw_vals, w, color="#d95f02", label="Uncalibrated Proxy")
    axes[1].bar(x + w / 2, cal_vals, w, color="#1b9e77", label="Calibrated Model")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics_names)
    axes[1].set_ylabel("Error / Loss Metric")
    axes[1].set_title("RQ1: Uncertainty Calibration Improvement")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig1_reliability_calibration.png")
    plt.close(fig)

    # Figure 2: Accuracy / Selective Reliability vs Token Cost Tradeoff (RQ2 & RQ3)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    colors = {
        "1. Direct LLM Answering": "#7570b3",
        "2. Root Answering (No Retrieval)": "#e7298a",
        "3. Always-Retrieve RAG": "#d95f02",
        "4. Fixed Confidence Threshold": "#66a61e",
        "5. Low-Confidence Triggered RAG": "#e6ab02",
        "6. Learned Utility Controller (Proposed)": "#1b9e77",
        "7. Oracle Upper Bound": "#333333",
    }
    for row in policy_table:
        p = row["policy"]
        c = colors.get(p, "#1f78b4")
        marker = "*" if "Proposed" in p else ("D" if "Oracle" in p else "o")
        size = 130 if "Proposed" in p else 75
        axes[0].scatter(row["avg_total_tokens"], row["selective_accuracy"] * 100, color=c, marker=marker, s=size, label=p, zorder=4)
        axes[1].scatter(row["avg_total_tokens"], row["mean_utility"], color=c, marker=marker, s=size, label=p, zorder=4)
    axes[0].set_xlabel("Average Total Tokens per State")
    axes[0].set_ylabel("Selective Answer Accuracy (%)")
    axes[0].set_title("RQ3: Selective Reliability vs. Token Budget")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Average Total Tokens per State")
    axes[1].set_ylabel("Mean Decision Utility U(a) = R(a) - λC(a)")
    axes[1].set_title("RQ3: Net Decision Utility vs. Token Cost (λ = 0.1)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=7.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig2_accuracy_cost_tradeoff.png")
    plt.close(fig)

    # Figure 3: Utility Across Cost Weight Sweep lambda (RQ3)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=180)
    by_pol = defaultdict(list)
    for r in lambda_sweep_rows:
        if r["incorrect_penalty"] == -1.0:
            by_pol[r["policy"]].append((r["lambda"], r["mean_utility"]))
    for p, pts in sorted(by_pol.items()):
        pts.sort()
        style = "--" if "Oracle" in p else ("-" if "Proposed" in p else "-.")
        lw = 2.6 if "Proposed" in p else 1.6
        ax.plot([x[0] for x in pts], [x[1] for x in pts], style, marker="o", linewidth=lw, color=colors.get(p, "#333"), label=p)
    ax.set_xlabel("Computation Cost Penalty Weight (λ)")
    ax.set_ylabel("Mean Decision Utility U(a) = R(a) - λC(a)")
    ax.set_title("RQ3: Policy Utility Across Computation Cost Weights λ (R_wrong = -1)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig3_lambda_utility_sweep.png")
    plt.close(fig)

    # Figure 4: Risk-Coverage Curve (RQ4) and Feature Group Ablations (RQ5)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), dpi=180)
    rc_cal = rq1_cal["combined_all_nodes"]["risk_coverage_curve_calibrated"]
    rc_raw = rq1_cal["combined_all_nodes"]["risk_coverage_curve_raw"]
    axes[0].plot([pt["coverage"] * 100 for pt in rc_raw], [(1.0 - pt["risk"]) * 100 for pt in rc_raw],
                 "--", color="#d95f02", linewidth=1.8, label=f"Uncalibrated Proxy (AURC={rq1_cal['combined_all_nodes']['raw_proxy']['aurc']:.3f})")
    axes[0].plot([pt["coverage"] * 100 for pt in rc_cal], [(1.0 - pt["risk"]) * 100 for pt in rc_cal],
                 "-", color="#1b9e77", linewidth=2.2, label=f"Calibrated Confidence (AURC={rq1_cal['combined_all_nodes']['calibrated']['aurc']:.3f})")
    axes[0].set_xlabel("Coverage (% of States Answered)")
    axes[0].set_ylabel("Selective Accuracy (%)")
    axes[0].set_title("RQ4: Selective Accuracy vs. Coverage (Abstention)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8.5)

    labels = [r["policy"].replace("Learned Controller", "").strip() for r in ablation_rows]
    utils = [r["mean_utility"] for r in ablation_rows]
    y_pos = np.arange(len(labels))
    bar_colors = ["#1b9e77" if i == 0 else "#386cb0" for i in range(len(labels))]
    axes[1].barh(y_pos, utils, color=bar_colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=8.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean Decision Utility U(a) (λ = 0.1)")
    axes[1].set_title("RQ5: Controller Feature-Group Ablation Study")
    axes[1].grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig4_risk_coverage_and_ablations.png")
    plt.close(fig)


def run_stage4_stage5_experiments(c, out):
    """Run full Stage 4 calibration & controller training and Stage 5 evaluation on collected nodes."""
    out = Path(out)
    rows = read_jsonl(out / "action_nodes.jsonl")
    direct = read_jsonl(out / "direct_predictions.jsonl")
    nodes_by_id = {r["node_id"]: r for r in rows}
    direct_by_state = {r["state_id"]: r for r in direct}
    roots = [r for r in rows if not r.get("history")]
    pairs = [(r, nodes_by_id[r["children"]["RETRIEVE"]]) for r in roots if "RETRIEVE" in r.get("children", {})]

    rq1 = evaluate_calibration_rq1(pairs, n_splits=5, seed=c["seed"])
    write_json(out / "stage4_calibration_report.json", rq1)

    concept_reward = {"correct": 1.0, "incorrect": -1.0, "abstain": 0.0}
    pilot_reward = dict(c["reward"])

    pol_concept, _ = run_all_policies_cv(pairs, direct_by_state, nodes_by_id, c["cost"], concept_reward, lambda_=0.1, seed=c["seed"])
    pol_pilot, _ = run_all_policies_cv(pairs, direct_by_state, nodes_by_id, c["cost"], pilot_reward, lambda_=0.1, seed=c["seed"])
    write_csv(out / "tables/policy_comparison_concept_r1.csv", pol_concept)
    write_csv(out / "tables/policy_comparison_pilot_r2.csv", pol_pilot)

    lambda_rows = []
    for rw_label, rw_cfg in [("concept_r_minus_1", concept_reward), ("pilot_r_minus_2", pilot_reward)]:
        for lam in c["lambdas"]:
            summaries, _ = run_all_policies_cv(pairs, direct_by_state, nodes_by_id, c["cost"], rw_cfg, lambda_=lam, seed=c["seed"])
            for s in summaries:
                lambda_rows.append({"reward_scheme": rw_label, **s})
    write_csv(out / "tables/lambda_sweep.csv", lambda_rows)

    rc_rows = [
        {"coverage": round(pt["coverage"], 4), "selective_accuracy": round(1.0 - pt["risk"], 4),
         "risk": round(pt["risk"], 4), "score_threshold": round(pt["score_threshold"], 4)}
        for pt in rq1["combined_all_nodes"]["risk_coverage_curve_calibrated"][::4]
    ]
    write_csv(out / "tables/risk_coverage.csv", rc_rows)

    ablations_concept = run_ablation_study(pairs, c["cost"], concept_reward, lambda_=0.1, seed=c["seed"])
    ablations_pilot = run_ablation_study(pairs, c["cost"], pilot_reward, lambda_=0.1, seed=c["seed"])
    write_csv(out / "tables/feature_ablations_concept_r1.csv", ablations_concept)
    write_csv(out / "tables/feature_ablations_pilot_r2.csv", ablations_pilot)

    generate_figures(out, rq1, pol_concept, lambda_rows, ablations_concept)

    full_report = {
        "experiment_id": c["experiment_id"],
        "questions": len({r["question_id"] for r, _ in pairs}),
        "states_evaluated": len(pairs),
        "rq1_calibration_summary": {
            "raw_ece": rq1["combined_all_nodes"]["raw_proxy"]["ece"],
            "calibrated_ece": rq1["combined_all_nodes"]["calibrated"]["ece"],
            "raw_brier": rq1["combined_all_nodes"]["raw_proxy"]["brier"],
            "calibrated_brier": rq1["combined_all_nodes"]["calibrated"]["brier"],
            "raw_auroc": rq1["combined_all_nodes"]["raw_proxy"]["auroc"],
            "calibrated_auroc": rq1["combined_all_nodes"]["calibrated"]["auroc"],
            "raw_aurc": rq1["combined_all_nodes"]["raw_proxy"]["aurc"],
            "calibrated_aurc": rq1["combined_all_nodes"]["calibrated"]["aurc"],
        },
        "rq3_policy_comparison_concept_r1_lambda_0_1": pol_concept,
        "rq3_policy_comparison_pilot_r2_lambda_0_1": pol_pilot,
        "rq5_ablations_concept_r1_lambda_0_1": ablations_concept,
        "rq5_ablations_pilot_r2_lambda_0_1": ablations_pilot,
    }
    write_json(out / "stage5_controller_evaluation.json", full_report)
    return full_report
