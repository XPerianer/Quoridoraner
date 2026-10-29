"""Task-balanced retention and minibatches; legacy three-field rows stay valid."""
from collections import defaultdict


def task(row):
    return row[3] if len(row) > 3 else 'standard'


def retain(replay, capacity, balanced=False):
    if not balanced:
        del replay[:-capacity]
        return
    groups = defaultdict(list)
    for index, row in enumerate(replay):
        groups[task(row)].append(index)
    if not groups:
        return
    # Reserve equal capacity per observed task. Spare capacity is deliberately
    # not filled by long games, which would displace rare tasks later.
    keep = set()
    for i, name in enumerate(sorted(groups)):
        quota = capacity // len(groups) + (i < capacity % len(groups))
        if quota:
            keep.update(groups[name][-quota:])
    replay[:] = [row for i, row in enumerate(replay) if i in keep]


def sample(replay, batch_size, rng, balanced=False):
    if not balanced:
        return rng.sample(replay, min(batch_size, len(replay)))
    groups = defaultdict(list)
    for row in replay:
        groups[task(row)].append(row)
    names = sorted(groups)
    rng.shuffle(names)
    batch = []
    for i, name in enumerate(names):
        count = batch_size // len(names) + (i < batch_size % len(names))
        pool = groups[name]
        batch.extend(rng.sample(pool, count) if count <= len(pool)
                     else rng.choices(pool, k=count))
    rng.shuffle(batch)
    return batch
