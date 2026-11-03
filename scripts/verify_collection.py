"""Check a collected pilot without requiring a trained controller or calibrator."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adaptive_qa.config import load_config, path_for
from adaptive_qa.io import read_jsonl, write_json
from adaptive_qa.llm.model import Usage
from adaptive_qa.reproducibility import check_manifest


def verify(c, out):
    check_manifest(c,out)
    rows=read_jsonl(out/'action_nodes.jsonl')
    nodes={r['node_id']:r for r in rows}
    roots=json.loads((out/'root_ids.json').read_text())
    splits=json.loads((out/'split_manifest.json').read_text())
    from adaptive_qa.data.load_qanta import load_questions
    from adaptive_qa.data.preprocess import make_states
    from adaptive_qa.io import digest
    questions, _ = load_questions(path_for(c, c['dataset']['path']), c['dataset']['text_only'])
    questions = sorted(questions, key=lambda q: digest([c['seed'], q.question_id]))
    if c['dataset']['limit_questions']:
        questions = questions[:c['dataset']['limit_questions']]
    expected_states = {s.state_id for q in questions for s in make_states(q, c['fractions'])}
    if len(nodes) != len(rows) or len(roots) != len(set(roots)):
        raise ValueError('Duplicate nodes or roots')
    if {nodes[n]['state_id'] for n in roots} != expected_states or len(roots) != len(expected_states):
        raise ValueError('Collection does not cover every configured question/prefix')
    branching = len(c['controller']['actions'])
    expected_nodes = len(expected_states) * sum(branching**d for d in range(c['controller']['max_steps'] + 1))
    if len(rows) != expected_nodes:
        raise ValueError('Incomplete collection tree')
    for nid in roots:
        assert nid in nodes and nodes[nid]['depth']==0
    for row in rows:
        assert splits[row['question_id']]==row['split']
        assert row['depth']==len(row['history'])
        assert row['remaining_steps']==c['controller']['max_steps']-row['depth']
        assert len(row['generations'])==c['model']['samples']
        usage=Usage(**row['retrieval_usage'])
        for generation in row['generations']:
            usage+=Usage(**generation['usage'])
        assert usage==Usage(**row['incremental_usage'])
        allowed=set(c['controller']['actions']) if row['remaining_steps'] else set()
        assert set(row['children'])==allowed
        for child_id in row['children'].values():
            child=nodes[child_id]
            assert child['state_id']==row['state_id'] and child['depth']==row['depth']+1
            assert Usage(**row['total_usage'])+Usage(**child['incremental_usage'])==Usage(**child['total_usage'])
    plain=read_jsonl(out/'direct_predictions.jsonl')
    assert len(plain)==len(roots)
    assert {r['state_id'] for r in plain}=={nodes[n]['state_id'] for n in roots}
    result={'status':'PASSED','root_states':len(roots),'nodes':len(rows),
            'calibration_required':False,'dataset':c['dataset']['name']}
    write_json(out/'collection_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/fixture_core_check.yaml')
    c=load_config(p.parse_args().config)
    print(json.dumps(verify(c,path_for(c,c['output_dir'])/c['experiment_id']),indent=2))
