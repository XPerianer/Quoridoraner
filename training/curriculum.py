"""Deterministic training starts and independent paired evaluation openings."""
import random


def training_position(config, iteration, index, stage=None):
    walls = config.get('walls', 10)
    position = dict(pawns=[76, 4], wallsLeft=[walls, walls], hWalls=[], vWalls=[], turn=0)
    if config.get('curriculum', 'none') == 'none':
        return position, 'standard'
    if stage is None:
        stage = (0 if config.get('curriculum') == 'gated' else
                 min(3, max(0, iteration - 1) // config.get('curriculum_interval', 25)))
    # One pair in four always practices the target game. Earlier tasks remain
    # in the mixture after promotion, rather than disappearing from replay.
    rng = random.Random(f"train:{config['seed']}:{iteration}:{index // 2}")
    if index // 2 % 4 == 3 and (config.get('curriculum') != 'gated' or stage == 3):
        return position, 'standard'
    level = rng.randrange(stage + 1)
    budget = min(walls, (0, 2, 5, 10)[level])
    distance = (2, 4, 6, 8)[level]
    a, b = rng.randint(1, distance), rng.randint(1, distance)
    pawns = [9 * a + rng.randrange(9), 9 * (8 - b) + rng.randrange(9)]
    if pawns[0] == pawns[1]:
        pawns[1] = 9 * (8 - b) + (pawns[1] % 9 + 1) % 9
    position.update(pawns=pawns, wallsLeft=[budget, budget], turn=rng.randrange(2))
    return position, f'race-{level}'


def evaluation_target(state):
    cfg = state['config']
    promotion = cfg.get('promotion_games', 40) if cfg.get('adaptive_negamax') and state['mode'] == 'negamax' else 0
    gate = cfg.get('curriculum_games', 40) if cfg.get('curriculum') == 'gated' else 0
    return gate + promotion + cfg['eval_games'] + (cfg.get('heldout_games', 0)
                               if state['iteration'] == cfg['iterations'] else 0)


def curriculum_state(state):
    return state.setdefault('curriculum_progress', {}).setdefault(
        state['mode'], dict(stage=0, streak=0, last_iteration=-1, history=[]))


def gate_opening(config, iteration, index, stage):
    """Independent paired starts at exactly the current race difficulty."""
    rng = random.Random(f"gate:{config['seed']}:{stage}:{iteration}:{index // 2}")
    distance = (2, 4, 6, 8)[stage]
    a, b = rng.randint(1, distance), rng.randint(1, distance)
    pawns = [9 * a + rng.randrange(9), 9 * (8 - b) + rng.randrange(9)]
    if pawns[0] == pawns[1]:
        pawns[1] = 9 * (8 - b) + (pawns[1] % 9 + 1) % 9
    walls = min(config.get('walls', 10), (0, 2, 5, 10)[stage])
    return dict(pawns=pawns, wallsLeft=[walls, walls], hWalls=[], vWalls=[], turn=rng.randrange(2))


def curriculum_check(state):
    cfg = state['config']
    if cfg.get('curriculum') != 'gated':
        return
    current = curriculum_state(state)
    iteration = state['iteration']
    if current['last_iteration'] >= iteration:
        return
    rows = [r for r in state['records'] if r['mode'] == state['mode'] and r['phase'] == 'eval'
            and r['iteration'] == iteration and r.get('suite') == 'curriculum']
    if len(rows) != cfg['curriculum_games']:
        return
    stage = current['stage']
    if any(r['task'] != f'race-{stage}' for r in rows):
        raise ValueError('Curriculum evaluation has inconsistent difficulty')
    wins = sum(r['score'] == 1 and r['termination'] == 'goal' for r in rows)
    rate = wins / len(rows)
    # Baseline measures the initial model but cannot advance training.
    current['streak'] = (current['streak'] + 1 if iteration > 0
                         and rate >= cfg['curriculum_win_rate'] else 0)
    if current['streak'] >= cfg['curriculum_streak'] and stage < 3:
        current['stage'] += 1
        current['streak'] = 0
    current['last_iteration'] = iteration
    current['history'].append(dict(iteration=iteration, stage=stage, wins=wins, games=len(rows),
                                   win_rate=rate, next_stage=current['stage'], streak=current['streak']))
    print(f"Curriculum {state['mode']}: race-{stage} {wins}/{len(rows)} goal wins; "
          f"streak {current['streak']}/{cfg['curriculum_streak']}; next race-{current['stage']}", flush=True)


def evaluation_opening(bridge, config, index, suite='monitor'):
    """Paired colors share an opening; held-out seeds use a separate namespace."""
    legacy = not config.get('varied_eval', False)
    rng = random.Random(config['seed'] + 10000 + index // 2 if legacy else
                        f"eval:{suite}:{config['seed']}:{index // 2}")
    walls = config.get('walls', 10)
    current = bridge.ask(position=dict(pawns=[76, 4], wallsLeft=[walls, walls],
                                       hWalls=[], vWalls=[], turn=0))
    from core import key
    history = {key(current['position']): 1}
    lengths = (10, 12, 14, 16, 18) if suite == 'heldout' else (0, 2, 4, 6, 8)
    for _ in range(4 if legacy else lengths[index // 2 % 5]):
        # Include legal wall openings, but retain mostly pawn moves. No third
        # repetition or terminal starting positions can enter the suite.
        pawn_moves = [m for m in current['legal'] if m < 81]
        candidates = list(current['legal'] if not legacy and rng.random() < .25 else pawn_moves)
        if not legacy:
            rng.shuffle(candidates)
        if legacy:
            candidates = [rng.choice(pawn_moves)]
        for move in candidates:
            child = bridge.ask(position=current['position'], move=move)
            k = key(child['position'])
            if child['winner'] == -1 and history.get(k, 0) < 2:
                current = child
                history[k] = history.get(k, 0) + 1
                break
    return current, history


def summarize(records):
    """Report actual goals separately from artificial zero-value cutoffs."""
    groups = {}
    for row in records:
        group = (row['mode'], row['phase'], row['iteration'], row.get('suite', 'monitor'))
        groups.setdefault(group, []).append(row)
    result = []
    for (mode, phase, iteration, suite), rows in sorted(groups.items()):
        n = len(rows)
        result.append(dict(mode=mode, phase=phase, iteration=iteration, suite=suite,
                           games=n, wins=sum(r['score'] == 1 for r in rows),
                           losses=sum(r['score'] == 0 for r in rows),
                           cutoffs=sum(r['termination'] != 'goal' for r in rows),
                           repetition_rate=sum(r['termination'] == 'repetition' for r in rows) / n,
                           goal_rate=sum(r['termination'] == 'goal' for r in rows) / n,
                           mean_plies=sum(r['plies'] for r in rows) / n,
                           score=sum(r['score'] for r in rows) / n))
    return result
