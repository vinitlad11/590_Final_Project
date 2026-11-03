"""Descriptive pilot diagnostics. Agreement is not a correctness probability."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
from adaptive_qa.evaluation.answers import exact_match
from adaptive_qa.io import read_jsonl, write_json
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
    return answer.strip().casefold() in {'short answer', 'your answer', 'answer here', 'candidate name'}


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
            'direct_states': len(plain), 'direct_correct_answers': sum(bool(r['correct']) for r in plain),
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
    flagged = sum(s['placeholder_flag'] for s in samples)
    invalid = sum(not s['parse_valid'] for s in samples)
    warnings = []
    if flagged:
        warnings.append(f'{flagged} outputs match placeholder text; inspect raw outputs before expanding.')
    if invalid:
        warnings.append(f'{invalid} outputs failed parsing; inspect errors without dropping them from denominators.')
    if not any(r['correct'] for r in roots):
        warnings.append('No correct root answers in this run. Inspect prompt behavior, scoring and evidence coverage; this is not evidence that the workflow failed or that a controller helps.')
    development = c['model']['backend'] == 'fixture' or 'PILOT' in c['dataset']['name'].upper()
    report = {
        'experiment_id': c['experiment_id'], 'dataset': c['dataset']['name'],
        'analysis_role': 'development_pilot_descriptive' if development else 'collection_descriptive_not_final_evaluation',
        'fixture_only': c['model']['backend'] == 'fixture', 'calibration_fitted_by_report': False,
        'prompt_version': c['model'].get('prompt_version', 'qa-v1'),
        'interpretation': 'Repeated generations measure consistency under the recorded sampling settings, not correctness probability. The same wrong answer can repeat. No significance claims are made.',
        'split_note': 'All pilot questions are development data, including partitions called calibration, tune or test. Calibration is disabled. These questions cannot later be an untouched holdout.',
        'questions': len({r['question_id'] for r in roots}), 'root_states': len(roots),
        'observed_nodes': len(rows), 'retrieval_collected': collected,
        'samples_reviewed': len(samples), 'placeholder_samples': flagged,
        'invalid_samples': invalid, 'valid_sample_json_rate': (len(samples) - invalid) / len(samples),
        'parser_note': 'The parser accepts a JSON object wrapped in a Markdown code fence. Placeholder flags do not change correctness scoring.',
        'logical_collection_usage': asdict(totals),
        'usage_scope': 'All node samples plus the independent direct sample. Repeated baseline files are not double-counted.' if collected else 'Repeated baseline samples only; direct baseline and retrieval not yet collected.',
        'time_note': 'Logical latency sums original component times even on cache hits; it is not current wall time. Invocation wall times include model loading but not notebook installation or data transfer.',
        'actual_invocations': invocation_totals(out), 'by_prefix': prefix_rows,
        'review_warnings': warnings, 'new_model_calls_from_reporting': 0,
    }
    tables = out / 'tables'
    write_csv(tables / 'pilot_by_prefix.csv', prefix_rows)
    write_csv(tables / 'sample_review.csv', samples)
    evidence = []
    for row in rows:
        if row.get('history') != 'R':
            continue
        for rank, doc in enumerate(row['documents'], 1):
            evidence.append({'question_id': row['question_id'], 'state_id': row['state_id'],
                             'requested_fraction': row['requested_fraction'], 'rank': rank,
                             'revealed_question': row['question_text'], 'prediction': row['prediction'],
                             'ground_truth': ' | '.join(row['ground_truth']),
                             'document_id': doc['document_id'], 'text': doc['text']})
    write_csv(tables / 'retrieval_review.csv', evidence)
    write_json(out / 'pilot_report.json', report)
    label = 'SOFTWARE FIXTURE ONLY' if report['fixture_only'] else 'DEVELOPMENT PILOT; DESCRIPTIVE'
    text = f'# Pilot observations\n\n{label}\n\nQuestions: {report["questions"]}; root states: {len(roots)}; samples: {len(samples)}.\n\n'
    text += report['interpretation'] + '\n\n'
    text += f'Placeholder flags: {flagged}. Invalid samples: {invalid}.\n\n'
    text += '| Prefix | States | Root correct | Direct correct | Retrieved correct | Stable but wrong |\n|---|---:|---:|---:|---:|---:|\n'
    for r in prefix_rows:
        text += f'| {r["requested_fraction"]:.0%} | {r["states"]} | {r["root_correct_answers"]} | {r["direct_correct_answers"] if collected else "n/a"} | {r["retrieved_correct_answers"] if collected else "n/a"} | {r["stable_but_wrong_states"]} |\n'
    text += f'\nLogical generation calls: {totals.llm_calls}. Logical retrieval calls: {totals.retrieval_calls}.\n\n'
    text += report['time_note'] + '\n\n' + report['split_note'] + '\n\n'
    text += '\n'.join('- ' + w for w in warnings)
    text += '\n\nTables: pilot_by_prefix.csv, sample_review.csv, and retrieval_review.csv (after collection).\n'
    text += '\nThis report does not train a controller or evaluate a deployed abstention policy.\n'
    (out / 'pilot_report.md').write_text(text, encoding='utf-8')
    return report
