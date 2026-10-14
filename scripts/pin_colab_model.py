"""Resolve one checkpoint SHA for both Colab pilot configs, before inference."""
from pathlib import Path
import json
import re
import yaml
from huggingface_hub import HfApi

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    paths = [p for n in (10,50) if (p := root/f'configs/colab_qanta2021_pilot{n}.yaml').exists()]
    if not paths:
        raise ValueError('No pilot config exists; import stage 2 first')
    configs = [yaml.safe_load(p.read_text()) for p in paths]
    revisions = {c['model']['revision'] for c in configs if c['model']['revision'] != 'SET_COMMIT_SHA'}
    if len(revisions) > 1:
        raise ValueError('Pilot configs disagree; restore the original pinned model revision')
    revision = next(iter(revisions)) if revisions else HfApi().model_info(configs[0]['model']['name']).sha
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Expected a 40-character immutable checkpoint SHA')
    for path,c in zip(paths,configs):
        if c['model']['revision'] != revision:
            run = root/'results/runs'/c['experiment_id']/'manifest.json'
            if run.exists():
                raise ValueError('A run already exists; do not change its model revision')
            path.write_text(path.read_text().replace('SET_COMMIT_SHA',revision))
    record = root/'results/logs/colab_model_pin.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({'model':configs[0]['model']['name'], 'revision':revision},indent=2)+'\n')
    print(revision)
