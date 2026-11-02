"""Bounded policy/value distillation from a frozen model onto packed replay positions.

Teacher predictions are raw network predictions, not fresh MCTS targets. A hybrid
term retains available replay policies/outcomes. Playing strength needs a separate
search-budget sweep; imitation loss alone cannot establish it.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from az import Engine, Net, Replay, TransformerNet, load_model, save_model


def distillation_loss(logits, value, teacher_logits, teacher_value, temperature=2.0):
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    policy = F.kl_div(F.log_softmax(logits.float() / temperature, dim=-1),
                      F.softmax(teacher_logits.detach().float() / temperature, dim=-1),
                      reduction='batchmean') * temperature ** 2
    regression = (value.float() - teacher_value.detach().float()).square().mean()
    return policy, regression


def replay_split(positions):
    # Duplicate exact board states stay in the same split, before augmentation.
    with np.errstate(over='ignore'):
        key = positions[:, 0] ^ (positions[:, 1] * np.uint64(0x9E3779B97F4A7C15))
        key ^= positions[:, 2] * np.uint64(0xBF58476D1CE4E5B9)
        key ^= key >> np.uint64(30)
    heldout = key % np.uint64(10) == 0
    return np.flatnonzero(~heldout), np.flatnonzero(heldout)


def read_replay(path):
    if str(path).endswith('.npz'):
        return Replay.load(path)
    with (gzip.open if str(path).endswith('.gz') else open)(path, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    if 'replay' not in state:
        raise ValueError('Input must be packed replay .npz or a full RL state with replay')
    return Replay.restore(state['replay'])


def fingerprint(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--replay', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--student', choices=['transformer', 'resnet'], required=True)
    parser.add_argument('--minutes', type=float, default=20)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=2)
    parser.add_argument('--hybrid-weight', type=float, default=.25)
    parser.add_argument('--teacher-weight', type=float, default=1,
                        help='Set 0 with hybrid-weight 1 for a replay-only control')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if (min(args.minutes, args.batch, args.temperature, args.lr, args.workers) <= 0
            or min(args.hybrid_weight, args.teacher_weight) < 0 or args.hybrid_weight + args.teacher_weight == 0):
        parser.error('Budgets, temperature, learning rate and workers must be positive; hybrid weight nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot requires CUDA')
    overall_start = time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.workers)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'model.pt').exists():
        raise RuntimeError('Use a fresh output directory to avoid overwriting an existing pilot')
    teacher, _ = load_model(args.teacher, 'cuda')
    teacher.requires_grad_(False)
    teacher.eval()
    replay = read_replay(args.replay)
    train, heldout = replay_split(replay.position[:replay.size])
    if not len(train) or not len(heldout):
        raise ValueError('Replay must contain both training and heldout state groups')
    student = (TransformerNet(width=128, blocks=2, heads=2) if args.student == 'transformer'
               else Net(channels=64, blocks=4)).cuda().to(memory_format=torch.channels_last)
    engine = Engine(args.workers)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    config = dict(args=vars(args), teacher_sha256=fingerprint(args.teacher),
                  replay_sha256=fingerprint(args.replay), config=student.config,
                  parameters=sum(p.numel() for p in student.parameters()),
                  train_rows=len(train), heldout_rows=len(heldout),
                  split='exact packed board-state groups, before random mirror; symmetry equivalents may cross splits',
                  targets='raw teacher logits/value plus available replay policies/outcomes; no new teacher search')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    def inputs(indices, mirror):
        features = engine.features(replay.position[indices], replay.repetitions[indices],
                                   replay.remaining[indices], mirror)
        return torch.from_numpy(features).cuda()

    @torch.no_grad()
    def validate():
        student.eval()
        indices = np.random.default_rng(731).choice(heldout, min(4096, len(heldout)), replace=False)
        totals = np.zeros(4)
        for start in range(0, len(indices), args.batch):
            idx = indices[start:start + args.batch]
            x = inputs(idx, np.zeros(len(idx), np.uint8))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tp, tv = teacher(x)
                sp, sv = student(x)
            p, v = distillation_loss(sp, sv, tp, tv, args.temperature)
            agreement = (sp.argmax(1) == tp.argmax(1)).float().mean()
            totals += np.array([p.item(), v.item(), agreement.item(), len(idx)]) * np.array([len(idx)] * 3 + [1])
        student.train()
        return dict(policy_kl_temperature_scaled=totals[0]/totals[3], value_mse=totals[1]/totals[3],
                    raw_policy_top1_agreement=totals[2]/totals[3], examples=int(totals[3]))

    try:
        initial = validate()
        setup_seconds = time.monotonic() - overall_start
        start, last_save, step = time.monotonic(), time.monotonic(), 0
        log = []
        while time.monotonic() - start < args.minutes * 60:
            step += 1
            elapsed = time.monotonic() - start
            fraction = min(1, elapsed / (args.minutes * 60))
            lr = args.lr * min(1, step / 200) * (.05 + .95 * .5 * (1 + math.cos(math.pi * fraction)))
            for group in optimizer.param_groups:
                group['lr'] = lr
            idx = rng.choice(train, args.batch, replace=True)
            mirror = rng.integers(0, 2, args.batch).astype(np.uint8)
            x = inputs(idx, mirror)
            target = replay.policy[idx].astype(np.float32)
            flipped = mirror.astype(bool)
            target[flipped] = target[flipped][:, engine.canonical[0, 1]]
            target = torch.from_numpy(target).cuda()
            mask = torch.from_numpy(replay.has_policy[idx]).cuda()
            outcome = torch.from_numpy(replay.value[idx]).cuda()
            if args.teacher_weight:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    tp, tv = teacher(x)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                sp, sv = student(x)
            p, v = (distillation_loss(sp, sv, tp, tv, args.temperature) if args.teacher_weight
                    else (sp.new_zeros(()), sv.new_zeros(())))
            supervised = (-(target * sp.float().log_softmax(1)).sum(1) * mask).sum() / mask.sum().clamp(min=1)
            supervised = supervised + (sv.float() - outcome).square().mean()
            loss = args.teacher_weight * (p + v) + args.hybrid_weight * supervised
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite distillation loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            if step % 100 == 0:
                row = dict(step=step, seconds=time.monotonic()-start, policy=float(p.detach()),
                           value=float(v.detach()), hybrid=float(supervised.detach()), lr=lr,
                           gradient_norm=float(norm))
                log.append(row)
                print(json.dumps(row), flush=True)
            if time.monotonic() - last_save >= 300:
                save_model(student, out / 'model.pt', optimizer=optimizer.state_dict(), step=step,
                           distilled_from=config['teacher_sha256'])
                last_save = time.monotonic()
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        validation_start = time.monotonic()
        final = validate()
        validation_seconds = time.monotonic() - validation_start
        save_model(student, out / 'model.pt', optimizer=optimizer.state_dict(), step=step,
                   distilled_from=config['teacher_sha256'])
        result = dict(config=config, training_seconds=seconds, steps=step, examples=step*args.batch,
                      setup_seconds=setup_seconds, final_validation_seconds=validation_seconds,
                      total_seconds=time.monotonic()-overall_start, initial=initial, final=final, log=log)
        (out / 'distillation.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(steps=step, training_seconds=seconds, final=final)), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
