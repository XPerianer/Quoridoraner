"""Checkpoint-owned opponent curriculum; fixed evaluations remain independent."""


def progression(state):
    cfg = state['config']
    return state.setdefault('adaptive', dict(nodes=cfg['negamax_start_nodes'], streak=0,
                                             last_iteration=-1, history=[]))


def opponent_nodes(state, suite):
    cfg = state['config']
    if cfg.get('adaptive_negamax') and state['mode'] == 'negamax' and (
            state['phase'] == 'train' or suite == 'promotion'):
        return progression(state)['nodes']
    return cfg.get('negamax_nodes', 2000)


def promotion_check(state):
    cfg = state['config']
    if not cfg.get('adaptive_negamax'):
        return
    current = progression(state)
    iteration = state['iteration']
    if current['last_iteration'] >= iteration:
        return
    rows = [r for r in state['records'] if r['mode'] == 'negamax' and r['phase'] == 'eval'
            and r['iteration'] == iteration and r.get('suite') == 'promotion']
    if len(rows) != cfg['promotion_games']:
        return
    if any(r['opponent_nodes'] != current['nodes'] for r in rows):
        raise ValueError('Promotion evaluation has inconsistent opponent strength')
    wins = sum(r['score'] == 1 and r['termination'] == 'goal' for r in rows)
    rate = wins / len(rows)
    current['streak'] = current['streak'] + 1 if rate >= cfg['promotion_win_rate'] else 0
    before = current['nodes']
    if current['streak'] >= cfg['promotion_streak']:
        current['nodes'] = min(cfg['negamax_nodes'], before * 2)
        current['streak'] = 0
    current['last_iteration'] = iteration
    current['history'].append(dict(iteration=iteration, nodes=before, wins=wins, games=len(rows),
                                   win_rate=rate, next_nodes=current['nodes'], streak=current['streak']))
    print(f"Adaptive negamax: {wins}/{len(rows)} wins at {before} nodes; "
          f"streak {current['streak']}/{cfg['promotion_streak']}; next {current['nodes']} nodes.", flush=True)
