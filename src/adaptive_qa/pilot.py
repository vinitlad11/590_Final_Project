"""Descriptive pilot diagnostics. Agreement is not a correctness probability."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
from adaptive_qa.io import read_jsonl, write_json
from adaptive_qa.llm.answer import parse_answer
from adaptive_qa.llm.model import Usage
from adaptive_qa.reproducibility import check_manifest


def pilot_report(c, out):
    out = Path(out)
    check_manifest(c, out)
    collected = (out / 'action_nodes.jsonl').exists()
    rows = read_jsonl(out / ('action_nodes.jsonl' if collected else 'baseline_predictions.jsonl'))
    roots = [r for r in rows if not r.get('history')]
    nodes = {r['node_id']:r for r in rows} if collected else {}
    if not roots:
        raise ValueError('No pilot observations found')
    totals = Usage()
    for row in rows:
        totals += Usage(**row['incremental_usage' if collected else 'usage'])
    if collected:
        for row in read_jsonl(out / 'direct_predictions.jsonl'):
            totals += Usage(**row['generation']['usage'])
    prefix_rows = []
    for fraction in sorted({r['fraction_revealed'] for r in roots}):
        subset = [r for r in roots if r['fraction_revealed'] == fraction]
        repeats = [r for r in subset if len(r['generations']) >= 2]
        stable = [r for r in repeats if r['parse_valid'] and r['self_consistency'] == 1
                  and all(parse_answer(g['raw']).valid for g in r['generations'])]
        pairs = [(r, nodes[r['children']['RETRIEVE']]) for r in subset
                 if collected and 'RETRIEVE' in r['children']]
        prefix_rows.append({
            'fraction_revealed':fraction, 'states':len(subset),
            'parse_valid_rate':sum(bool(r['parse_valid']) for r in subset)/len(subset),
            'correct_answer_rate':sum(bool(r['correct']) for r in subset)/len(subset),
            'repeat_states':len(repeats), 'all_samples_agree_states':len(stable),
            'stable_but_wrong_states':sum(not r['correct'] for r in stable),
            'retrieval_pairs':len(pairs),
            'retrieval_corrected_errors':sum(not r['correct'] and child['correct'] for r,child in pairs),
            'retrieval_introduced_errors':sum(r['correct'] and not child['correct'] for r,child in pairs),
        })
    generations = [g for r in rows for g in r['generations']]
    development = c['model']['backend'] == 'fixture' or 'PILOT' in c['dataset']['name'].upper()
    report = {
        'experiment_id':c['experiment_id'], 'dataset':c['dataset']['name'],
        'analysis_role':'development_pilot_descriptive' if development else 'collection_descriptive_not_final_evaluation',
        'fixture_only':c['model']['backend'] == 'fixture',
        'calibration_fitted_by_report':False,
        'interpretation':'Repeated generations measure consistency under the recorded sampling settings, not correctness probability. The same wrong answer can repeat.',
        'reserved_calibration_split':('Kept for split compatibility; unused for fitting when calibration.enabled is false. All pilot questions remain development data.'
                                      if development else 'A calibration partition is optional and unused when calibration.enabled is false. Preserve the final holdout from tuning.'),
        'questions':len({r['question_id'] for r in roots}), 'root_states':len(roots),
        'observed_nodes':len(rows), 'retrieval_collected':collected,
        'valid_sample_json_rate':sum(parse_answer(g['raw']).valid for g in generations)/len(generations),
        'logical_collection_usage':asdict(totals),
        'usage_scope':'All node samples plus direct baseline, without double-counting the separate baseline file' if collected else 'Baseline samples only; no direct baseline or retrieval collected yet',
        'time_note':'latency_seconds sums recorded component times, including cached observations. It is not current wall time. Keep notebook stage logs and failure records.',
        'latest_invocation_ledgers':{}, 'by_prefix':prefix_rows,
        'new_model_calls_from_reporting':0,
    }
    for name in ('baseline_collection_usage.json','collection_usage.json'):
        if (out/name).exists():
            report['latest_invocation_ledgers'][name] = json.loads((out/name).read_text())
    tables = out/'tables'; tables.mkdir(parents=True,exist_ok=True)
    with (tables/'pilot_by_prefix.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=prefix_rows[0]);writer.writeheader();writer.writerows(prefix_rows)
    write_json(out/'pilot_report.json', report)
    label = 'SOFTWARE FIXTURE ONLY' if report['fixture_only'] else ('DEVELOPMENT PILOT' if development else 'COLLECTION DIAGNOSTICS; NOT FINAL EVALUATION')
    text = f"# Pilot observations\n\n{label}\n\nQuestions: {report['questions']}; root states: {len(roots)}.\n\n"
    text += report['interpretation']+'\n\n'
    text += 'Inspect tables/pilot_by_prefix.csv and pilot_report.json. All repeated samples count toward usage.\n'
    text += f"\nLogical generation calls in the reported collection: {totals.llm_calls}.\n"
    text += '\nThis report does not fit calibration, train a controller or establish final-study performance.\n'
    (out/'pilot_report.md').write_text(text,encoding='utf-8')
    return report
