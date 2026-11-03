"""Sum retained Stage 3 invocation records without executing the model."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adaptive_qa.pilot import invocation_totals
from adaptive_qa.io import write_json


def summarize(root):
    rows = {p.name: invocation_totals(p) for p in sorted((root / 'results/runs').glob('qanta2021_stage03_*')) if p.is_dir()}
    keys = ('attempted_backend_calls', 'successful_calls', 'failed_calls', 'pending_calls',
            'cache_hits', 'finished_wall_seconds', 'unfinished_invocations')
    result = {'experiments': rows, 'totals': {k: sum(r[k] for r in rows.values()) for k in keys},
              'scope': 'Only retained Stage 3 sessions. Stage 2 work is additional. Interrupted wall time and failed/pending token use may be unknown. Cache hits do not erase original computational cost.'}
    result['successful_input_tokens'] = sum(r['measured_successful_usage']['input_tokens'] for r in rows.values())
    result['successful_output_tokens'] = sum(r['measured_successful_usage']['output_tokens'] for r in rows.values())
    write_json(root / 'results/stage03_cost_summary.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(Path(__file__).resolve().parents[1]), indent=2))
