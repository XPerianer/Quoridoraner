"""Use an exported neural bot from the terminal: play, choose a move, or evaluate."""
import argparse
import json
from pathlib import Path
import random

import torch

from device import select_device
from bridge import create_bridge
from core import model_from_config, choose, key, search
from curriculum import evaluation_opening


def board_text(position):
    down, right = set(), set()
    for anchor in position['hWalls']:
        y, x = divmod(anchor, 8)
        down.update([(y, x), (y, x + 1)])
    for anchor in position['vWalls']:
        y, x = divmod(anchor, 8)
        right.update([(y, x), (y + 1, x)])
    lines = ['    a b c d e f g h i']
    for y in range(9):
        row = f'{9-y}   '
        for x in range(9):
            cell = y * 9 + x
            pawn = 'W' if cell == position['pawns'][0] else 'B' if cell == position['pawns'][1] else '·'
            row += pawn + ('┃' if (y, x) in right else ' ')
        lines.append(row)
        if y < 8:
            lines.append('    ' + ''.join('━━' if (y, x) in down else '  ' for x in range(9)))
    lines.append(f"Walls: W {position['wallsLeft'][0]} / B {position['wallsLeft'][1]}")
    return '\n'.join(lines)


def notation(move):
    if move < 81:
        y, x = divmod(move, 9)
        return f'{chr(97+x)}{9-y}'
    y, x = divmod(move - (145 if move >= 145 else 81), 8)
    return f"{chr(97+x)}{9-y}{'v' if move >= 145 else 'h'}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['play', 'move', 'evaluate'])
    parser.add_argument('checkpoint', type=Path, help='Exported selfplay.pt or negamax.pt')
    parser.add_argument('--position', type=Path, help='Position JSON for move command; defaults to initial board')
    parser.add_argument('--games', type=int, default=4)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--suite', choices=['monitor', 'heldout'], default='heldout',
                        help='Evaluation opening namespace; use a new seed for independent follow-up tests')
    parser.add_argument('--negamax-nodes', type=int, default=2000)
    parser.add_argument('--max-plies', type=int, default=200)
    parser.add_argument('--engine', choices=['typescript', 'native', 'native-negamax', 'direct'], default='typescript')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--human', choices=['white', 'black'], default='white')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    if min(args.games, args.simulations, args.negamax_nodes, args.max_plies) < 1:
        parser.error('Budgets must be positive')
    if args.command == 'evaluate' and args.games % 2:
        parser.error('--games must be even for paired colors')
    torch.set_num_threads(1)
    # Keep move-command stdout machine-readable while reporting device timings.
    import contextlib
    import sys
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    with contextlib.redirect_stdout(sys.stderr):
        device, _ = select_device(args.device, 1, model_config=saved.get('config'))
    model = model_from_config(saved.get('config')).to(device)
    model.load_state_dict(saved['model'])
    model.eval()
    rng = random.Random(args.seed)
    bridge = create_bridge(args.engine, mps_inference=saved.get('config', {}).get('mps_inference', 'bucketed'))
    try:
        if args.command == 'move':
            state = bridge.ask(**({'position': json.loads(args.position.read_text())} if args.position else {}))
            policy = search(model, bridge, state, {key(state['position']): 1}, 0, args.max_plies,
                            args.simulations, rng, device)
            move = choose(policy, rng, False)
            print(json.dumps({'move': move, 'notation': notation(move), 'policy': policy.tolist()}))
            return
        scores = []
        for game_index in range(args.games if args.command == 'evaluate' else 1):
            bridge.ask(op='clear')
            state = bridge.ask()
            counts = {key(state['position']): 1}
            bot_color = game_index % 2 if args.command == 'evaluate' else (1 if args.human == 'white' else 0)
            if args.command == 'evaluate':
                config = {**saved.get('config', {}), 'seed': args.seed, 'varied_eval': True}
                state, counts = evaluation_opening(bridge, config, game_index, args.suite)
            reason = 'maxPlies'
            for ply in range(args.max_plies):
                if state['winner'] != -1:
                    reason = 'goal'
                    break
                if counts[key(state['position'])] >= 3:
                    reason = 'repetition'
                    break
                if args.command == 'play':
                    print(board_text(state['position']))
                if state['position']['turn'] == bot_color:
                    policy = search(model, bridge, state, counts, ply, args.max_plies, args.simulations, rng, device, tree_id='bot')
                    move = choose(policy, rng, False)
                    if args.command == 'play':
                        print(f'Bot: {notation(move)}')
                elif args.command == 'evaluate':
                    move = bridge.ask(op='negamax', position=state['position'], nodes=args.negamax_nodes)['move']
                else:
                    legal = {notation(m): m for m in state['legal']}
                    while True:
                        answer = input('Move (e2, d5h, d5v; ? legal moves; q quit): ').strip().lower()
                        if answer == 'q':
                            return
                        if answer == '?':
                            print(' '.join(legal))
                        elif answer in legal:
                            move = legal[answer]
                            break
                        else:
                            print('That move is not legal.')
                state = bridge.ask(op='advance', moves=[{'id': 'bot', 'position': state['position'], 'move': move}])['states'][0]
                k = key(state['position'])
                counts[k] = counts.get(k, 0) + 1
            winner = state['winner']
            if winner != -1:
                reason = 'goal'
            score = 0.5 if winner == -1 else float(winner == bot_color)
            scores.append(score)
            print(f"Game {game_index+1}: {'artificial cutoff' if winner == -1 else 'bot wins' if score else 'opponent wins'} ({reason})")
        if args.command == 'evaluate':
            print(f'Bot: {scores.count(1.0)}W / {scores.count(0.5)} cutoffs / {scores.count(0.0)}L; score {sum(scores)/len(scores):.1%}')
    finally:
        bridge.close()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nStopped.')
