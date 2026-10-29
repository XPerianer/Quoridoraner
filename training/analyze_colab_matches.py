"""Reproducible full-wall screening against a shared local reference."""
import json
from pathlib import Path
import torch
from core import model_from_config
from benchmark_depth import evaluate

def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = model_from_config({**saved['config'], 'precision': 'fp32'}).to('mps').eval()
    model.load_state_dict(saved['model'])
    return model, saved

def main():
    torch.set_num_threads(1)
    root = Path('training/runs')
    out = root / 'colab-strength'; out.mkdir(exist_ok=True)
    result_path = out / 'matches.json'
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    imports = sorted((root/'colab-import').rglob('selfplay.pt'))
    candidates = [p.parent for p in imports]
    candidates += [root/n for n in ('rolling-fixed-01','lattice-fixed-01','gated-balanced-full','curriculum-64-u32')]
    for mode in ('selfplay','negamax'):
        reference_path = root/'both-values-full-curriculum'/f'{mode}.pt'
        reference, reference_saved = load(reference_path)
        for directory in candidates:
            path = directory/f'{mode}.pt'
            if not path.exists() or any(r['candidate']==str(path) for r in rows): continue
            model, saved = load(path)
            print('START',directory.name,mode,flush=True)
            result = evaluate(model, saved['config'], 'mps', 'standard-10', 20, reference)
            row = dict(candidate=str(path),reference=str(reference_path),iteration=saved.get('iteration'),
                       reference_iteration=reference_saved.get('iteration'),simulations=64,precision='fp32',
                       opening_seed=917,max_plies=100,result=result)
            rows.append(row);result_path.write_text(json.dumps(rows,indent=2)+'\n')
            print('RESULT',directory.name,mode,result['wins'],result['losses'],result['cutoffs'],flush=True)
            del model, saved
        del reference, reference_saved

if __name__=='__main__': main()
