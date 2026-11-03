"""Per-command wall time and attempt records, retained across retries."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from adaptive_qa.io import write_json


@contextmanager
def record_invocation(c, out, command):
    invocation = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    folder = Path(out) / 'sessions' / invocation
    folder.mkdir(parents=True)
    c['_usage_ledger'] = str(folder / 'model_usage.json')
    write_json(folder / 'start.json', {
        'command': command, 'experiment_id': c['experiment_id'],
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'budget_scope': 'per invocation', 'max_new_calls': c['budget']['max_new_calls'],
    })
    started, status, error = time.perf_counter(), 'completed', None
    try:
        yield
    except BaseException as exc:
        status, error = 'failed_or_interrupted', type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        write_json(folder / 'finish.json', {
            'status': status, 'error': error,
            'wall_seconds': time.perf_counter() - started,
            'finished_utc': datetime.now(timezone.utc).isoformat(),
        })
        c.pop('_usage_ledger', None)
