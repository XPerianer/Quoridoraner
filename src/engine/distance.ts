import type { Board } from './board'
import type { Player } from './types'

export type DistanceKernel = (board: Board, player: Player) => number

/** Independent queue BFS reference and candidate. Scratch belongs to one board. */
export function queueDistance(): DistanceKernel {
  const queue = new Uint8Array(81)
  const dist = new Int8Array(81)
  return (b, p) => {
    dist.fill(-1)
    const start = b.pawn[p], goal = p === 0 ? 0 : 8
    queue[0] = start
    dist[start] = 0
    let head = 0, tail = 1
    while (head < tail) {
      const c = queue[head++], y = (c / 9) | 0, x = c % 9
      if (y === goal) return dist[c]
      for (let d = 0; d < 4; d++) {
        if (!b.canStep(x, y, d)) continue
        const n = c + [-9, 1, 9, -1][d]
        if (dist[n] !== -1) continue
        dist[n] = dist[c] + 1
        queue[tail++] = n
      }
    }
    return -1
  }
}

/** One loop per wave: horizontal/vertical expansion and convergence together. */
export function fusedDistance(): DistanceKernel {
  let a = new Uint16Array(11), z = new Uint16Array(11)
  return (b, p) => {
    const c = b.pawn[p], row = ((c / 9) | 0) + 1, goal = p === 0 ? 1 : 9
    if (row === goal) return 0
    a.fill(0)
    a[row] = 1 << (c % 9)
    for (let steps = 1; ; steps++) {
      let changed = 0
      for (let y = 0; y < 9; y++) {
        const r = a[y + 1], right = b.blockRight[y]
        const next = r | (((r & ~right) << 1) & 511) | ((r >> 1) & ~right)
          | (a[y] & ~(y === 0 ? 511 : b.blockDown[y - 1]))
          | (a[y + 2] & ~b.blockDown[y])
        z[y + 1] = next
        changed |= next ^ r
      }
      if (z[goal]) return steps
      if (!changed) return -1
      const t = a; a = z; z = t
    }
  }
}

/** Restrict the fused wave to rows reachable within this many steps. */
export function boundedDistance(): DistanceKernel {
  let a = new Uint16Array(11), z = new Uint16Array(11)
  return (b, p) => {
    const c = b.pawn[p], row = ((c / 9) | 0) + 1, goal = p === 0 ? 1 : 9
    if (row === goal) return 0
    a.fill(0); z.fill(0)
    a[row] = 1 << (c % 9)
    let lo = row, hi = row
    for (let steps = 1; ; steps++) {
      lo = Math.max(1, lo - 1); hi = Math.min(9, hi + 1)
      let changed = 0
      for (let i = lo; i <= hi; i++) {
        const y = i - 1, r = a[i], right = b.blockRight[y]
        const next = r | (((r & ~right) << 1) & 511) | ((r >> 1) & ~right)
          | (a[i - 1] & ~(y === 0 ? 511 : b.blockDown[y - 1]))
          | (a[i + 1] & ~b.blockDown[y])
        z[i] = next
        changed |= next ^ r
      }
      if (z[goal]) return steps
      if (!changed) return -1
      const t = a; a = z; z = t
    }
  }
}

/** Scalar row bitboards: avoid typed-array indexing in the wave loop. */
export const scalarDistance: DistanceKernel = (b, p) => {
  const c = b.pawn[p], y = (c / 9) | 0, bit = 1 << (c % 9)
  if (y === (p === 0 ? 0 : 8)) return 0
  let r0 = y === 0 ? bit : 0
  let r1 = y === 1 ? bit : 0
  let r2 = y === 2 ? bit : 0
  let r3 = y === 3 ? bit : 0
  let r4 = y === 4 ? bit : 0
  let r5 = y === 5 ? bit : 0
  let r6 = y === 6 ? bit : 0
  let r7 = y === 7 ? bit : 0
  let r8 = y === 8 ? bit : 0
  const h0 = ~b.blockRight[0]
  const h1 = ~b.blockRight[1]
  const h2 = ~b.blockRight[2]
  const h3 = ~b.blockRight[3]
  const h4 = ~b.blockRight[4]
  const h5 = ~b.blockRight[5]
  const h6 = ~b.blockRight[6]
  const h7 = ~b.blockRight[7]
  const h8 = ~b.blockRight[8]
  const v0 = ~b.blockDown[0]
  const v1 = ~b.blockDown[1]
  const v2 = ~b.blockDown[2]
  const v3 = ~b.blockDown[3]
  const v4 = ~b.blockDown[4]
  const v5 = ~b.blockDown[5]
  const v6 = ~b.blockDown[6]
  const v7 = ~b.blockDown[7]
  for (let steps = 1; ; steps++) {
    const n0 = r0 | (((r0 & h0) << 1) & 511) | ((r0 >> 1) & h0) | (r1 & v0)
    const n1 = r1 | (((r1 & h1) << 1) & 511) | ((r1 >> 1) & h1) | (r0 & v0) | (r2 & v1)
    const n2 = r2 | (((r2 & h2) << 1) & 511) | ((r2 >> 1) & h2) | (r1 & v1) | (r3 & v2)
    const n3 = r3 | (((r3 & h3) << 1) & 511) | ((r3 >> 1) & h3) | (r2 & v2) | (r4 & v3)
    const n4 = r4 | (((r4 & h4) << 1) & 511) | ((r4 >> 1) & h4) | (r3 & v3) | (r5 & v4)
    const n5 = r5 | (((r5 & h5) << 1) & 511) | ((r5 >> 1) & h5) | (r4 & v4) | (r6 & v5)
    const n6 = r6 | (((r6 & h6) << 1) & 511) | ((r6 >> 1) & h6) | (r5 & v5) | (r7 & v6)
    const n7 = r7 | (((r7 & h7) << 1) & 511) | ((r7 >> 1) & h7) | (r6 & v6) | (r8 & v7)
    const n8 = r8 | (((r8 & h8) << 1) & 511) | ((r8 >> 1) & h8) | (r7 & v7)
    if (p === 0 ? n0 : n8) return steps
    if (((n0 ^ r0) | (n1 ^ r1) | (n2 ^ r2) | (n3 ^ r3) | (n4 ^ r4) | (n5 ^ r5) | (n6 ^ r6) | (n7 ^ r7) | (n8 ^ r8)) === 0) return -1
    r0 = n0
    r1 = n1
    r2 = n2
    r3 = n3
    r4 = n4
    r5 = n5
    r6 = n6
    r7 = n7
    r8 = n8
  }
}
