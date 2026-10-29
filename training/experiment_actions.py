"""Small controlled goal-tactic probe; does not modify production models or search."""
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from core import PolicyValue, encode_batch, key, search_batch
from direct import DirectBridge

OFFSETS = [(0,-1),(1,0),(0,1),(-1,0),(0,-2),(2,0),(0,2),(-2,0),(-1,-1),(1,-1),(-1,1),(1,1)]
OUT = Path(__file__).parent / 'measurements/action-encoding'


def immediate_win(position, legal):
    goal = 0 if position['turn'] == 0 else 8
    return next((m for m in legal if m < 81 and m // 9 == goal), None)


def mapping(p):
    origin = p['pawns'][p['turn']]
    x, y = origin % 9, origin // 9
    result = {}
    for i, (dx,dy) in enumerate(OFFSETS):
        if 0 <= x+dx < 9 and 0 <= y+dy < 9:
            result[(y+dy)*9+x+dx] = i
    result.update({m: m-81+12 for m in range(81,209)})
    return result


def dataset(bridge, count, kind, rng, seen):
    rows = []
    while len(rows) < count:
        turn, x = rng.randrange(2), rng.randrange(9)
        goal, direction = (0,1) if turn == 0 else (8,-1)
        mine = (goal+direction*(2 if kind == 'straight' else 1))*9+x
        other = goal*9+x if kind == 'diagonal' else ((goal+direction)*9+x if kind == 'straight' else rng.randrange(81))
        pawns = [mine,other] if turn == 0 else [other,mine]
        if mine == other or pawns[0] < 9 or pawns[1] >= 72:
            continue
        p = dict(pawns=pawns,wallsLeft=[10,10],hWalls=[],vWalls=[],turn=turn)
        # Each added wall passes the engine's fit and two-player path checks.
        for _ in range(rng.randrange(1,9)):
            walls = [m for m in bridge.describe(p)['legal'] if m >= 81]
            if not walls: break
            m = rng.choice(walls)
            p['vWalls' if m >= 145 else 'hWalls'].append(m-(145 if m >= 145 else 81))
            p['wallsLeft'][turn] -= 1
        p['hWalls'].sort(); p['vWalls'].sort()
        legal = bridge.describe(p)['legal']
        wins = [m for m in legal if m < 81 and m//9 == goal]
        if not wins: continue
        if kind == 'step' and any(abs(m%9-x)+abs(m//9-mine//9) != 1 for m in wins): continue
        signature = key(p)
        if signature in seen: continue
        seen.add(signature)
        mp = mapping(p)
        assert all(m in mp for m in legal)
        assert len({mp[m] for m in legal}) == len(legal)
        rows.append(dict(position=p,legal=legal,wins=wins,kind=kind))
    return rows


class AbsoluteAdapter(nn.Module):
    def __init__(self, model, relative):
        super().__init__(); self.model=model; self.relative=relative
    def forward(self,x):
        logits, values = self.model(x)
        if not self.relative: return logits, values
        out = logits.new_full((len(x),209), -1e9)
        out[:,81:] = logits[:,12:]
        turns = x[:,6,0,0].long()
        origins = x[torch.arange(len(x)),turns].flatten(1).argmax(1)
        for i,(dx,dy) in enumerate(OFFSETS):
            xx, yy = origins%9+dx, origins//9+dy
            valid = (xx>=0)&(xx<9)&(yy>=0)&(yy<9)
            ids = torch.arange(len(x))[valid]
            out[ids, yy[valid]*9+xx[valid]] = logits[ids,i]
        return out,values


@torch.inference_mode()
def evaluate(model, relative, rows):
    adapter=AbsoluteAdapter(model,relative)
    logits,_=adapter(encode_batch([(r['position'],1,1.) for r in rows]))
    result={}
    for kind in ('step','straight','diagonal'):
        selected=[(i,r) for i,r in enumerate(rows) if r['kind']==kind]
        result[kind]=sum(r['legal'][int(logits[i,r['legal']].argmax())] in r['wins'] for i,r in selected)/len(selected)
    return result


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    bridge=DirectBridge(workers=1,mps_inference='eager')
    rng=random.Random(731); seen=set()
    pools={k:dataset(bridge,1200,k,rng,seen) for k in ('step','straight','diagonal')}
    test=sum([dataset(bridge,100,k,rng,seen) for k in pools],[])
    (OUT/'positions.json').write_text(json.dumps(dict(pools=pools,test=test))+'\n')
    initial = dict(pawns=[76,4],wallsLeft=[10,10],hWalls=[],vWalls=[],turn=0)
    assert immediate_win(initial,bridge.describe(initial)['legal']) is None
    for r in test:
        move = immediate_win(r['position'],r['legal'])
        assert bridge.advance(bridge.identifier('guard-check'),r['position'],move)['winner'] == r['position']['turn']
    bridge.ask(op='clear')
    results=[]
    try:
        for fraction in (0.,.05,1/3):
            nd=round(1200*fraction); ns=(1200-nd)//2
            rows=pools['step'][:ns]+pools['straight'][:1200-nd-ns]+pools['diagonal'][:nd]
            x=encode_batch([(r['position'],1,1.) for r in rows])
            for seed in (1,2,3):
                for relative in (False,True):
                    torch.manual_seed(seed)
                    model=PolicyValue('small')
                    if relative: model.policy=nn.Linear(128,140)
                    target=torch.zeros(len(rows),140 if relative else 209)
                    for i,r in enumerate(rows):
                        ids=[mapping(r['position'])[m] if relative else m for m in r['wins']]
                        target[i,ids]=1/len(ids)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                    gen=torch.Generator().manual_seed(seed+100)
                    start=time.perf_counter()
                    for _ in range(150):
                        ids=torch.randint(len(rows),(64,),generator=gen)
                        logits,values=model(x[ids])
                        loss=-(target[ids]*logits.log_softmax(1)).sum(1).mean()+(values-1).square().mean()
                        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step()
                    model.eval()
                    row=dict(diagonal_fraction=fraction,seed=seed,encoding='relative' if relative else 'destination',seconds=time.perf_counter()-start,accuracy=evaluate(model,relative,test),search={})
                    # Fresh trees, no exploration noise; evaluate identical diagonal cases.
                    cases=[r for r in test if r['kind']=='diagonal'][:30]
                    games=[dict(id=str(i),state=bridge.describe(r['position']),history={key(r['position']):1},ply=0) for i,r in enumerate(cases)]
                    for budget in (8,32):
                        bridge.ask(op='clear')
                        policy=search_batch(AbsoluteAdapter(model,relative),bridge,games,100,budget,random.Random(7),'cpu')
                        row['search'][str(budget)]=sum(int(policy[str(i)].argmax()) in r['wins'] for i,r in enumerate(cases))/len(cases)
                    # Exact tactical guard uses only legal moves and the goal row.
                    guarded=[immediate_win(r['position'],r['legal']) for r in test]
                    assert all(m in r['wins'] for m,r in zip(guarded,test))
                    row['immediate_win_guard_accuracy']=1.
                    results.append(row)
                    (OUT/'results.json').write_text(json.dumps(dict(updates=150,batch_size=64,training_positions=1200,test_per_kind=100,search_diagonal_positions=30,seeds=[1,2,3],results=results),indent=2)+'\n')
                    print(json.dumps(row),flush=True)
    finally: bridge.close()

if __name__=='__main__': main()
