"""Dependency-free terminal loss plots, derived from checkpoint history."""
import math


def loss_curves(history, width=76, height=5):
    """Policy/value panels use independent y scales and a shared update axis.

    The most recent loss for each arm is printed in the panel title, so the
    chart is readable at a glance even when the curves are still flat.
    """
    series = {mode: [row for row in history if row['mode'] == mode]
              for mode in ('selfplay', 'negamax')}
    count = max(map(len, series.values()), default=0)
    if not count:
        return ['  awaiting gradient updates']
    stacked = width < 60
    panel_width = width if stacked else (width - 3) // 2
    plot_width = max(2, panel_width - 9)
    height = 2 if stacked else max(3, height)
    markers = (('selfplay', '●'), ('negamax', '◆'))
    panels = []
    for metric in ('policy', 'value'):
        values = [row[metric] for rows in series.values() for row in rows if math.isfinite(row[metric])]
        if not values:
            panels.append([f'{metric.upper()} LOSS'.ljust(panel_width)] +
                          [' ' * panel_width] * (height + 1))
            continue
        low, high = min(values), max(values)
        if high == low:
            padding = max(abs(high) * 0.05, 0.01)
            low, high = max(0, low - padding), high + padding
        grid = [[' ' for _ in range(plot_width)] for _ in range(height)]
        for mode, marker in markers:
            previous = None
            for index, row in enumerate(series[mode]):
                value = row[metric]
                if not math.isfinite(value):
                    previous = None
                    continue
                x = round(index * (plot_width - 1) / max(1, count - 1))
                y = round((high - value) * (height - 1) / (high - low))
                y = max(0, min(height - 1, y))
                start_x, start_y = previous if previous is not None else (x, y)
                steps = max(abs(x - start_x), abs(y - start_y), 1)
                for step in range(steps + 1):
                    px = round(start_x + (x - start_x) * step / steps)
                    py = round(start_y + (y - start_y) * step / steps)
                    old = grid[py][px]
                    grid[py][px] = marker if old in (' ', marker) else '◎'
                previous = x, y
        latest = []
        for mode, marker in markers:
            row = next((row for row in reversed(series[mode]) if math.isfinite(row[metric])), None)
            if row is not None:
                latest.append(f'{marker}{row[metric]:.3g}')
        title = f'{metric.upper()} LOSS'
        if latest:
            title += '  ' + ' '.join(latest)
        panel = [title[:panel_width].ljust(panel_width)]
        for y, cells in enumerate(grid):
            label = f'{high if y == 0 else low:.3g}' if y in (0, height - 1) else ''
            panel.append(f'{label:>7} │' + ''.join(cells))
        panel.append('        └' + '─' * plot_width)
        panels.append(panel)
    lines = panels[0] + panels[1] if stacked else [left + '   ' + right for left, right in zip(*panels)]
    lines.append(f'● selfplay  ◆ negamax  ◎ overlap   updates 1–{count}')
    return [line[:width] for line in lines]
