"""Frozen replay capacity and equal-wall-budget self-play architecture experiments.

Never modifies the source checkpoint. Run sequentially without other trainers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time

import torch

from core import ARCHITECTURES, PolicyValue, choose, key, search_batch, update
from curriculum import evaluation_opening, gate_opening
from direct import DirectBridge
from inference import MpsInference

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def replay_split(state, mode):
    """Recover game-pair groups from append-ordered uniform replay, validate alignment.

    Balanced replay is intentionally rejected: its retention changes row ordering.
    Both-value supervision means exactly one replay row per played ply in either arm.
    """
    cfg = state['config']
    if cfg.get('replay_sampling', 'uniform') != 'uniform' or cfg.get('negamax_value_turns') != 'both':
        raise ValueError('Frozen source requires uniform replay and both-turn value targets')
    replay = state['arms'][mode]['replay']
    cursor = len(replay)
    groups = [None] * cursor
    for record in reversed(state['records']):
        if record['mode'] != mode or record['phase'] != 'train':
            continue
        count = min(cursor, record['plies'])
        first_ply = record['plies'] - count
        group = f"{record['iteration']}:{record['game']//2}"
        for offset in range(count):
            row = replay[cursor-count+offset]
            ply = first_ply + offset
            turn = (record['opening']['turn'] + ply) % 2
            expected = 0. if record['winner'] == -1 else (1. if record['winner'] == turn else -1.)
            assert row[2] == expected, 'Outcome/replay ordering mismatch'
            assert abs(float(row[0][8, 0, 0]) - (cfg['max_plies'] - ply)/cfg['max_plies']) < 1e-6
            assert float(row[0][6, 0, 0]) == turn
            groups[cursor-count+offset] = group
        cursor -= count
        if not cursor:
            break
    assert cursor == 0 and all(g is not None for g in groups)
    is_validation = {g: int.from_bytes(hashlib.sha256(('depth-split:'+g).encode()).digest()[:4], 'big') % 5 == 0
                     for g in groups}
    validation = [row for row, group in zip(replay, groups) if is_validation[group]]
    fingerprints = {hashlib.sha256(row[0].numpy().tobytes()).digest() for row in validation}
    training = [row for row, group in zip(replay, groups) if not is_validation[group]
                and hashlib.sha256(row[0].numpy().tobytes()).digest() not in fingerprints]
    assert training and validation
    metadata = dict(training_rows=len(training), validation_rows=len(validation),
                    removed_duplicate_training_states=len(replay)-len(validation)-len(training),
                    training_game_pairs=sum(not v for v in is_validation.values()),
                    validation_game_pairs=sum(is_validation.values()),
                    value_distribution={str(v):sum(row[2] == v for row in replay) for v in (-1.,0.,1.)})
    return training, validation, metadata


@torch.inference_mode()
def loss_metrics(model, rows, device):
    model.eval()
    policy_ce = policy_entropy = value_error = correct = 0.
    policy_count = 0
    for start in range(0, len(rows), 256):
        batch = rows[start:start+256]
        logits, predictions = model(torch.stack([r[0] for r in batch]).to(device))
        logits, predictions = logits.cpu(), predictions.cpu()
        values = torch.tensor([r[2] for r in batch])
        value_error += float((predictions-values).square().sum())
        ids = [i for i, row in enumerate(batch) if row[1] is not None]
        if ids:
            targets = torch.stack([batch[i][1] for i in ids])
            policy_ce += float(-(targets*logits[ids].log_softmax(1)).sum())
            policy_entropy += float(-(targets*targets.clamp_min(1e-12).log()).sum())
            correct += float((targets.argmax(1) == logits[ids].argmax(1)).sum())
            policy_count += len(ids)
    return dict(policy_cross_entropy=policy_ce/max(1,policy_count),
                policy_kl=(policy_ce-policy_entropy)/max(1,policy_count),
                policy_top1=correct/max(1,policy_count), value_mse=value_error/len(rows),
                policy_rows=policy_count, rows=len(rows))


@torch.inference_mode()
def latency(model, device):
    model.eval()
    runner = MpsInference(model, 'hybrid') if device == 'mps' else None
    result = {}
    for n in (1, 16, 64, 128):
        x = torch.zeros(n,9,9,9)
        samples=[]
        for i in range(15):
            start=time.perf_counter()
            if runner is not None:
                runner(x)
            else:
                logits, values=model(x.to(device)); torch.cat((logits,values[:,None]),1).cpu()
            if i >= 5:
                samples.append((time.perf_counter()-start)*1000)
        result[str(n)]=statistics.median(samples)
    return result


def frozen(args):
    state=torch.load(args.source, map_location='cpu', weights_only=True)
    data={mode:replay_split(state,mode) for mode in ('selfplay','negamax')}
    manifest=dict(source=str(args.source.resolve()), sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                  source_iteration=state['iteration'], source_config=state['config'],
                  split={mode:rows[2] for mode,rows in data.items()}, updates=args.updates,
                  batch_size=128, device=args.device)
    save_json(args.out/'frozen-data.json',manifest)
    del state
    results=[]
    for seed_index,seed in enumerate(args.seeds):
        architectures=list(args.architectures)
        if seed_index%2: architectures.reverse()
        for architecture in architectures:
            for mode,(training,validation,_) in data.items():
                torch.manual_seed(seed)
                model=PolicyValue(architecture).to(args.device)
                optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                initial=loss_metrics(model,validation,args.device)
                rng=random.Random(seed)
                start=time.perf_counter()
                for _ in range(args.updates):
                    model.train()
                    update(model,optimizer,training,128,rng,args.device)
                elapsed=time.perf_counter()-start
                row=dict(architecture=architecture,seed=seed,mode=mode,
                         parameters=sum(p.numel() for p in model.parameters()),updates=args.updates,
                         fit_seconds=elapsed,initial_validation=initial,
                         training=loss_metrics(model,training,args.device),
                         validation=loss_metrics(model,validation,args.device),
                         inference_ms=latency(model,args.device))
                results.append(row); save_json(args.out/'frozen-results.json',results)
                print('FROZEN',architecture,seed,mode,round(elapsed,2),row['validation'],flush=True)
    return results


@torch.inference_mode()
def evaluate(model, cfg, device, suite, games=40, opponent_model=None):
    bridge=DirectBridge(workers=4,mps_inference='hybrid')
    rng=random.Random(917)
    active=[]; completed=[]
    try:
        for index in range(games):
            if suite == 'race-1':
                state=bridge.ask(position=gate_opening({**cfg,'seed':917,'walls':2},0,index,1))
                history={key(state['position']):1}
            else:
                walls=2 if suite == 'standard-2' else 10
                state,history=evaluation_opening(bridge,{**cfg,'seed':917,'walls':walls,'varied_eval':True},index,'heldout')
            active.append(dict(id=str(index),state=state,history=history,ply=0,learner=index%2))
        while active:
            remaining=[]
            for g in active:
                winner=g['state']['winner']
                if winner != -1 or g['history'][key(g['state']['position'])]>=3 or g['ply']>=100:
                    completed.append(dict(score=.5 if winner == -1 else float(winner == g['learner']),
                                          plies=g['ply'],reason='goal' if winner != -1 else 'cutoff'))
                    bridge.ask(op='drop',ids=[g['id']])
                else: remaining.append(g)
            active=remaining
            neural=[g for g in active if g['state']['position']['turn']==g['learner']]
            policies=search_batch(model,bridge,neural,100,64,rng,device) if neural else {}
            opponents=[g for g in active if g['id'] not in policies]
            if opponent_model is not None:
                opposing_policies=search_batch(opponent_model,bridge,opponents,100,64,rng,device) if opponents else {}
                other={identifier:choose(policy,rng,False) for identifier,policy in opposing_policies.items()}
            else:
                responses=bridge.ask(op='negamax_batch',positions=[g['state']['position'] for g in opponents],nodes=2000)['results'] if opponents else []
                other={g['id']:r['move'] for g,r in zip(opponents,responses)}
            moves=[dict(id=g['id'],position=g['state']['position'],move=choose(policies[g['id']],rng,False)
                        if g['id'] in policies else other[g['id']]) for g in active]
            states=bridge.ask(op='advance',moves=moves)['states'] if moves else []
            for g,state in zip(active,states):
                g['state']=state; g['ply']+=1
                k=key(state['position']); g['history'][k]=g['history'].get(k,0)+1
            if opponent_model is not None:
                # Different networks must not inherit each other's cached priors
                # or values. Both sides get a fresh 64-simulation tree per move.
                bridge.ask(op='clear')
        scores=[r['score'] for r in completed]
        return dict(suite=suite,games=games,wins=scores.count(1.),cutoffs=scores.count(.5),losses=scores.count(0.),
                    score=sum(scores)/games,records=completed)
    finally: bridge.close()


def online(args):
    results=[]
    for seed_index,seed in enumerate(args.seeds):
        architectures=list(args.architectures)
        if seed_index%2: architectures.reverse()
        for architecture in architectures:
            directory=args.out/f'online-{architecture}-{seed}'
            command=[sys.executable,str(ROOT/'training/train.py'),'--out',str(directory),
                     '--architecture',architecture,'--seed',str(seed),'--device',args.device,'--plain',
                     '--engine','direct','--workers','4','--mps-inference','hybrid',
                     '--iterations','100000','--games','32','--parallel-games','32',
                     '--simulations','64','--updates','32','--batch-size','128',
                     '--walls','2','--curriculum','none','--eval-games','2','--heldout-games','0',
                     '--eval-every','100000','--checkpoint-seconds','60']
            print('ONLINE',architecture,seed,f'{args.seconds}s budget',flush=True)
            start=time.perf_counter()
            with (args.out/f'online-{architecture}-{seed}.log').open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=log,cwd=ROOT)
                try:
                    try:
                        code=process.wait(timeout=args.seconds)
                        raise RuntimeError(f'Trainer exited early ({code}); inspect {directory}')
                    except subprocess.TimeoutExpired:
                        process.send_signal(signal.SIGINT)
                        code=process.wait(timeout=120)
                        if code: raise RuntimeError(f'Trainer failed ({code})')
                finally:
                    if process.poll() is None:
                        process.terminate(); process.wait(timeout=120)
            wall=time.perf_counter()-start
            saved=torch.load(directory/'latest.pt',map_location='cpu',weights_only=True)
            row=dict(architecture=architecture,seed=seed,command=command,budget_seconds=args.seconds,
                     process_wall_seconds=wall,iteration=saved['iteration'],phase=saved['phase'],
                     performance=json.loads((directory/'performance.json').read_text()),arms={})
            for mode in ('selfplay','negamax'):
                model=PolicyValue(architecture).to(args.device)
                model.load_state_dict(saved['arms'][mode]['model']); model.eval()
                metrics=[r for r in saved['records'] if r['mode']==mode and r['phase']=='train']
                row['arms'][mode]=dict(updates=sum(r['mode']==mode for r in saved.get('loss_history',[])),
                    training_games=len(metrics),training_goals=sum(r['termination']=='goal' for r in metrics),
                    evaluation=[evaluate(model,saved['config'],args.device,suite,args.eval_games)
                                for suite in ('race-1','standard-2','standard-10')])
                print('EVAL',architecture,seed,mode,[(r['suite'],r['wins'],r['cutoffs']) for r in row['arms'][mode]['evaluation']],flush=True)
            del saved
            results.append(row); save_json(args.out/'online-results.json',results)
    return results


def matches(args):
    if args.trained_runs is None:
        raise ValueError('--trained-runs is required for matches')
    results=[]
    for seed in args.seeds:
        for mode in ('selfplay','negamax'):
            saved=torch.load(args.trained_runs/f'online-small-{seed}'/f'{mode}.pt',map_location='cpu',weights_only=True)
            reference=PolicyValue('small').to(args.device)
            reference.load_state_dict(saved['model']); reference.eval()
            for architecture in (a for a in args.architectures if a != 'small'):
                saved=torch.load(args.trained_runs/f'online-{architecture}-{seed}'/f'{mode}.pt',map_location='cpu',weights_only=True)
                model=PolicyValue(architecture).to(args.device)
                model.load_state_dict(saved['model']); model.eval()
                row=dict(architecture=architecture,seed=seed,mode=mode,opponent='small',
                         result=evaluate(model,saved['config'],args.device,'standard-2',args.eval_games,reference))
                results.append(row); save_json(args.out/'matches.json',results)
                print('MATCH',architecture,seed,mode,row['result']['wins'],row['result']['cutoffs'],row['result']['losses'],flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--architectures', nargs='+', choices=ARCHITECTURES,
                        default=['small', 'residual32', 'residual64'])
    parser.add_argument('--source',type=Path,default=ROOT/'experiments/training/both-values-full-curriculum/latest.pt')
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='mps')
    parser.add_argument('--seeds',type=int,nargs='+',default=[1,2])
    parser.add_argument('--updates',type=int,default=500)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--eval-games',type=int,default=40)
    parser.add_argument('--phase',choices=['frozen','online','both','matches'],default='both')
    parser.add_argument('--trained-runs',type=Path)
    args=parser.parse_args()
    if args.seconds<=0 or args.updates<1 or args.eval_games<2 or args.eval_games%2:
        parser.error('Positive budgets and even evaluation games are required')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    if args.phase in ('frozen','both'): frozen(args)
    if args.phase in ('online','both'): online(args)
    if args.phase == 'matches': matches(args)


if __name__ == '__main__': main()
