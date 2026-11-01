"""Play the Colab model against pinned, independent public Quoridor bots.

Dependencies live in experiments/training/external-bots; see EXTERNAL_BOTS.md for provenance.
No fallback moves: encoding/legality disagreements abort the match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import torch
from az import Engine, START, load_model, turn, winner

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / 'experiments/training'
BOTS = RUNS / 'external-bots'


class Goris:
    def __init__(self, sims, seed):
        self.sims = sims
        self.proc = subprocess.Popen(['node', str(HERE / 'external_goris.cjs'),
                                      str(BOTS / 'quoridor-ai')], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ask(op='reset', seed=seed)

    def ask(self, **q):
        self.proc.stdin.write(json.dumps(q) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('Gorisanson bridge exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def legal(self):
        return set(self.ask(op='legal')['legal'])

    def choose(self):
        return self.ask(op='choose', simulations=self.sims)['move']

    def advance(self, move):
        self.ask(op='apply', move=move)

    def close(self):
        self.proc.terminate()
        self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()


class Sigma:
    def __init__(self, sims, seed):
        sys.path.insert(0, str(BOTS / 'SigmaQuoridor'))
        from game import State, PawnAction
        from mcts import MCTSAgent
        from dual_network import make_nn_evaluator
        self.PawnAction = PawnAction
        self.state = State(boardsize=9, walls_p1=10, walls_p2=10, walls_initial=10)
        self.agent = MCTSAgent(evaluator=make_nn_evaluator(
            str(BOTS / 'SigmaQuoridor/runs/models_9x9_scratch/best.pt'),
            device=torch.device('cpu')), num_simulations=sims,
            training=False, temperature=0.0)

    def encode(self, action):
        if isinstance(action, self.PawnAction):
            x, y = self.state._pawn_dest(action)
            return (8 - y) * 9 + x
        return (81 if action.orientation == 'h' else 145) + (7 - action.y) * 8 + action.x

    def actions(self):
        return {self.encode(a): a for a in self.state.get_legal_actions()}

    def legal(self):
        # Sigma prunes third repetitions as a search rule. For geometry parity
        # only, inspect a copy with repetition counts reset; actual search keeps history.
        history = self.state.position_history
        cache = self.state._legal_actions_cache
        try:
            self.state.position_history = {}
            self.state._legal_actions_cache = None
            return set(self.actions())
        finally:
            self.state.position_history = history
            self.state._legal_actions_cache = cache

    def choose(self):
        return self.encode(self.agent.select_action(self.state))

    def advance(self, move):
        self.state = self.state.next(self.actions()[move])

    def close(self):
        pass


class Borowski(Goris):
    """The page's hard-coded top-side AI, reflected vertically when playing first."""
    def __init__(self, sims, seed):
        self.sims = sims
        self.pawns, self.walls, self.turn = [76, 4], [10, 10], 0
        self.wall_history = []
        self.proc = subprocess.Popen(['node', str(HERE / 'external_borowski.cjs'),
                                      str(BOTS / 'borowski')], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)
        self.ask(op='reset', seed=seed)

    def transform(self, move):
        if self.turn == 1:
            return move
        if move < 81:
            return (8 - move // 9) * 9 + move % 9
        base = 81 if move < 145 else 145
        i = move - base
        return base + (7 - i // 8) * 8 + i % 8

    def payload(self):
        return dict(p0=self.transform(self.pawns[1-self.turn]),
                    p1=self.transform(self.pawns[self.turn]),
                    w0=self.walls[1-self.turn], w1=self.walls[self.turn],
                    wallHistory=[self.transform(a) for a in self.wall_history])

    def legal(self):
        # This older bot's UI maintains its own candidate list, including its
        # original splice behavior; it is not a complete standard-rules generator.
        return {self.transform(a) for a in self.ask(op='legal', state=self.payload())['legal']}

    def choose(self):
        return self.transform(self.ask(op='choose', state=self.payload())['move'])

    def advance(self, move):
        if move < 81:
            self.pawns[self.turn] = move
        else:
            self.walls[self.turn] -= 1
            self.wall_history.append(move)
        self.turn ^= 1


def notation(a):
    if a < 81:
        return f'{chr(97 + a % 9)}{9 - a // 9}'
    i = a - (81 if a < 145 else 145)
    return f'{chr(97 + i % 8)}{9 - i // 8}{"h" if a < 145 else "v"}'


def check(engine, pos, opponent):
    if isinstance(opponent, Borowski):
        packed = pos[2]
        assert opponent.pawns == [packed & 127, (packed >> 7) & 127]
        assert opponent.walls == [(packed >> 14) & 15, (packed >> 18) & 15]
        assert opponent.turn == turn(pos)
        h = sum(1 << (a-81) for a in opponent.wall_history if a < 145)
        v = sum(1 << (a-145) for a in opponent.wall_history if a >= 145)
        assert (h, v) == pos[:2]
        return
    ours, theirs = set(engine.legal(pos)), opponent.legal()
    if ours != theirs:
        raise AssertionError(f'Legal sets differ: ours-only={ours-theirs}, theirs-only={theirs-ours}, pos={pos}')


def make_opponent(name, args, seed):
    if name == 'gorisanson':
        return Goris(args.goris_sims, seed)
    if name == 'sigma':
        return Sigma(args.sigma_sims, seed)
    if name == 'borowski':
        return Borowski(0, seed)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opponents', nargs='+', choices=['gorisanson', 'sigma', 'borowski'], default=['gorisanson', 'sigma'])
    p.add_argument('--model', type=Path, default=RUNS / 'az/colab/rl-model.pt')
    p.add_argument('--simulations', type=int, default=400)
    p.add_argument('--goris-sims', type=int, default=60000)
    p.add_argument('--sigma-sims', type=int, default=400)
    p.add_argument('--pairs', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--max-plies', type=int, default=200)
    p.add_argument('--validate', action='store_true')
    p.add_argument('--resume', action='store_true', help='Skip completed saved games with matching configuration')
    p.add_argument('--out', type=Path, default=RUNS / 'external-matches')
    args = p.parse_args()
    torch.set_num_threads(1)
    engine = Engine(1)
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.validate:
        for name in args.opponents:
            count = 0
            for game in range(8):
                opponent = make_opponent(name, args, args.seed + game)
                pos = START
                try:
                    for ply in range(35):
                        check(engine, pos, opponent)
                        legal = list(engine.legal(pos))
                        # Sigma excludes repeated positions internally; use its action map here.
                        if name == 'sigma':
                            legal = list(opponent.actions())
                        move = rng.choice(legal)
                        pos = engine.advance(pos, move)
                        opponent.advance(move)
                        count += 1
                        if winner(pos) >= 0:
                            break
                finally:
                    opponent.close()
            print(json.dumps({'validated': name, 'positions': count}), flush=True)
        return
    model, _ = load_model(args.model, torch.device('cpu'))
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta['model_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    source_names = {'gorisanson': 'quoridor-ai', 'sigma': 'SigmaQuoridor', 'borowski': 'borowski'}
    meta['sources'] = {source_names[n]: json.loads((BOTS / source_names[n] / 'SOURCE.json').read_text())
                       for n in args.opponents}
    config_path = args.out / 'config.json'
    if args.resume and config_path.exists():
        old = json.loads(config_path.read_text())
        for key in meta:
            if key != 'resume' and old.get(key) != meta[key]:
                raise ValueError(f'Resume configuration differs: {key}')
    config_path.write_text(json.dumps(meta, indent=2))
    results = []
    for name in args.opponents:
        for pair in range(args.pairs):
            opening = []
            if pair:
                pos = START
                for _ in range(4):
                    move = rng.choice(engine.legal(pos))
                    opening.append(move)
                    pos = engine.advance(pos, move)
            for color in (0, 1):
                game_id = f'{name}-pair{pair}-ours{color}'
                saved_path = args.out / f'{game_id}.json'
                if args.resume and saved_path.exists():
                    saved = json.loads(saved_path.read_text())
                    if 'outcome' in saved:
                        summary = {k: v for k, v in saved.items() if k != 'moves'}
                        summary['plies'] = len(saved['moves']) + len(saved['opening'])
                        results.append(summary)
                        continue
                opponent = make_opponent(name, args, args.seed + pair)
                engine.clear()
                pos, history, moves = START, {START: 1}, []
                # The native engine only advances trees with a registered history.
                engine.history(0, history)
                try:
                    for move in opening:
                        opponent.advance(move)
                        pos = engine.advance(pos, move, 0)
                        history[pos] = history.get(pos, 0) + 1
                    result = dict(id=game_id, opponent=name, our_color=color, opening=opening, moves=moves)
                    start = time.perf_counter()
                    for ply in range(len(opening), args.max_plies):
                        check(engine, pos, opponent)
                        t = time.perf_counter()
                        if turn(pos) == color:
                            pi, value = engine.search(model, [dict(tree=0, position=pos, history=history,
                                ply=ply, max_plies=args.max_plies)], args.simulations, torch.device('cpu'))
                            move = int(pi[0].argmax())
                            value = float(value[0])
                        else:
                            move, value = opponent.choose(), None
                        elapsed = time.perf_counter() - t
                        if move not in engine.legal(pos):
                            raise AssertionError(f'{name}: illegal chosen move {move}')
                        side = turn(pos)
                        pos = engine.advance(pos, move, 0)
                        history[pos] = history.get(pos, 0) + 1
                        moves.append(dict(ply=ply + 1, player=side, move=move, notation=notation(move),
                                          seconds=elapsed, value=value, position=pos))
                        print(f'{game_id} ply={ply+1} {"ours" if side==color else name} '
                              f'{notation(move)} {elapsed:.2f}s value={value}', flush=True)
                        w = winner(pos)
                        if w >= 0:
                            result.update(winner=w, outcome='win' if w == color else 'loss', reason='goal')
                        elif history[pos] >= 3:
                            result.update(winner=-1, outcome='cutoff', reason='threefold')
                        elif ply + 1 >= args.max_plies:
                            result.update(winner=-1, outcome='cutoff', reason='plies')
                        (args.out / f'{game_id}.json').write_text(json.dumps(result, indent=2))
                        if 'outcome' in result:
                            break
                        opponent.advance(move)
                    result['seconds'] = time.perf_counter() - start
                    (args.out / f'{game_id}.json').write_text(json.dumps(result, indent=2))
                    summary = {k: v for k, v in result.items() if k != 'moves'}
                    summary['plies'] = len(moves) + len(opening)
                    results.append(summary)
                    (args.out / 'summary.json').write_text(json.dumps(results, indent=2))
                    print('RESULT ' + json.dumps(summary), flush=True)
                finally:
                    opponent.close()


if __name__ == '__main__':
    main()
