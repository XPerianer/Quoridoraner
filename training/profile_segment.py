"""Profile one additional iteration from a completed checkpoint without modifying it."""
import argparse
import cProfile
import json
from pathlib import Path
import pstats
import random
import time

import torch
import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    if state['phase'] != 'done':
        parser.error('Use a completed checkpoint; in-flight games must not be discarded')
    args.out.mkdir(parents=True, exist_ok=False)
    train.migrate(state)
    state.update(phase='train', mode='selfplay', iteration=state['iteration'] + 1,
                 game_index=0, active_games=[], game=None, trees=[], performance={})
    state['config']['iterations'] = state['iteration']
    models, optimizers = {}, {}
    for mode in train.MODES:
        models[mode], optimizers[mode] = train.new_arm(state['arms'][mode]['model'], args.device, state['config'])
        optimizers[mode].load_state_dict(state['arms'][mode]['optimizer'])
    rng = random.Random()
    rng.setstate(state['rng'])
    torch.set_rng_state(state['torch_rng'])
    if args.device == 'mps' and state.get('mps_rng') is not None:
        torch.mps.set_rng_state(state['mps_rng'])
    if args.device == 'cuda' and state.get('cuda_rng') is not None:
        torch.cuda.set_rng_state(state['cuda_rng'])
    state['selected_device'] = args.device
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.runcall(train.run, state, models, optimizers, rng, args.out, args.device, True)
    elapsed = time.perf_counter() - start
    profile.dump_stats(str(args.out / 'profile.pstats'))
    with (args.out / 'profile.txt').open('w') as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(65)
    (args.out / 'wall.json').write_text(json.dumps(dict(wall_seconds=elapsed,
        source=str(args.checkpoint.resolve()), device=args.device), indent=2) + '\n')


if __name__ == '__main__':
    main()
