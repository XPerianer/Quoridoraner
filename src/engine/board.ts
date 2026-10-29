import { scalarDistance } from './distance'
import type { Move, Position, Player } from './types'

export const SIZE = 9
export const CELLS = SIZE * SIZE
export const WALL_SIZE = SIZE - 1
export const WALL_SLOTS = WALL_SIZE * WALL_SIZE
export const H_BASE = CELLS
export const V_BASE = CELLS + WALL_SLOTS
export const MOVE_SPACE = V_BASE + WALL_SLOTS
export const MAX_MOVES = 8 + 2 * WALL_SLOTS
export const START_WALLS = 10
export const GOAL_ROW: readonly number[] = [0, SIZE - 1]

const ROW_MASK = (1 << SIZE) - 1
const VERTICES = (SIZE + 1) * (SIZE + 1)
const DX = [0, 1, 0, -1]
const DY = [-1, 0, 1, 0]

const ZOBRIST_PAWN = 0
const ZOBRIST_H = 2 * CELLS
const ZOBRIST_V = ZOBRIST_H + WALL_SLOTS
const ZOBRIST_WALLS_LEFT = ZOBRIST_V + WALL_SLOTS
const ZOBRIST_TURN = ZOBRIST_WALLS_LEFT + 2 * (START_WALLS + 1)
const ZOBRIST_LEN = ZOBRIST_TURN + 1

function zobristTable(seed: number): Int32Array {
  const table = new Int32Array(ZOBRIST_LEN)
  let s = seed
  for (let i = 0; i < ZOBRIST_LEN; i++) {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    table[i] = s
  }
  return table
}

const Z1 = zobristTable(0x9e3779b9)
const Z2 = zobristTable(0x85ebca6b)

export const isPawnMove = (m: Move) => m < H_BASE
export const isVerticalWall = (m: Move) => m >= V_BASE
export const wallIndex = (m: Move) => (m >= V_BASE ? m - V_BASE : m - H_BASE)
export const wallMove = (vertical: boolean, x: number, y: number): Move =>
  (vertical ? V_BASE : H_BASE) + y * WALL_SIZE + x
export const cellX = (c: number) => c % SIZE
export const cellY = (c: number) => (c / SIZE) | 0

/** Reflect a move across the vertical centerline; negative sentinels remain unchanged. */
export function mirrorMove(m: Move): Move {
  if (m < 0) return m
  if (m < H_BASE) return m + SIZE - 1 - 2 * cellX(m)
  const i = wallIndex(m)
  return m + WALL_SIZE - 1 - 2 * (i % WALL_SIZE)
}

function mirroredTable(z: Int32Array): Int32Array {
  const result = z.slice()
  for (let p = 0; p < 2; p++) {
    for (let c = 0; c < CELLS; c++) result[p * CELLS + c] = z[p * CELLS + mirrorMove(c)]
  }
  for (let i = 0; i < WALL_SLOTS; i++) {
    result[ZOBRIST_H + i] = z[ZOBRIST_H + mirrorMove(H_BASE + i) - H_BASE]
    result[ZOBRIST_V + i] = z[ZOBRIST_V + mirrorMove(V_BASE + i) - V_BASE]
  }
  return result
}

const M1 = mirroredTable(Z1)
const M2 = mirroredTable(Z2)

export function moveToString(m: Move): string {
  if (isPawnMove(m)) return String.fromCharCode(97 + cellX(m)) + (SIZE - cellY(m))
  const i = wallIndex(m)
  const x = i % WALL_SIZE
  const y = (i / WALL_SIZE) | 0
  return String.fromCharCode(97 + x) + (SIZE - y) + (isVerticalWall(m) ? 'v' : 'h')
}

export class Board {
  readonly pawn = new Int8Array(2)
  readonly wallsLeft = new Int8Array(2)
  readonly hWall = new Uint8Array(WALL_SLOTS)
  readonly vWall = new Uint8Array(WALL_SLOTS)
  /** 64 wall slots per orientation, split into native 32-bit limbs for JS bit operations. */
  readonly hBits = new Uint32Array(2)
  readonly vBits = new Uint32Array(2)
  /** bit x of row y: stepping (x,y) -> (x,y+1) is blocked */
  readonly blockDown = new Uint16Array(SIZE)
  /** bit x of row y: stepping (x,y) -> (x+1,y) is blocked */
  readonly blockRight = new Uint16Array(SIZE)
  turn: Player = 0
  hash1 = 0
  hash2 = 0
  mirrorHash1 = 0
  mirrorHash2 = 0

  private readonly wallTouch = new Uint8Array(VERTICES)
  private readonly fromStack = new Int8Array(1024)
  private sp = 0
  private floodA = new Uint16Array(SIZE)
  private floodB = new Uint16Array(SIZE)
  private readonly queue = new Int16Array(CELLS)
  private readonly candidates = new Uint8Array(2 * WALL_SLOTS)

  static initial(): Board {
    return Board.from({
      pawns: [CELLS - 1 - 4, 4],
      wallsLeft: [START_WALLS, START_WALLS],
      hWalls: [],
      vWalls: [],
      turn: 0,
    })
  }

  static from(p: Position): Board {
    const b = new Board()
    b.pawn[0] = p.pawns[0]
    b.pawn[1] = p.pawns[1]
    b.wallsLeft[0] = p.wallsLeft[0]
    b.wallsLeft[1] = p.wallsLeft[1]
    b.turn = p.turn
    for (const i of p.hWalls) b.setWall(false, i)
    for (const i of p.vWalls) b.setWall(true, i)
    b.hash1 = b.computeHash(Z1)
    b.hash2 = b.computeHash(Z2)
    b.mirrorHash1 = b.computeHash(M1)
    b.mirrorHash2 = b.computeHash(M2)
    return b
  }

  toPosition(): Position {
    const walls = (w: Uint8Array) => [...w.keys()].filter((i) => w[i])
    return {
      pawns: [this.pawn[0], this.pawn[1]],
      wallsLeft: [this.wallsLeft[0], this.wallsLeft[1]],
      hWalls: walls(this.hWall),
      vWalls: walls(this.vWall),
      turn: this.turn,
    }
  }

  private computeHash(z: Int32Array): number {
    let h = z[ZOBRIST_PAWN + this.pawn[0]] ^ z[ZOBRIST_PAWN + CELLS + this.pawn[1]]
    for (let i = 0; i < WALL_SLOTS; i++) {
      if (this.hWall[i]) h ^= z[ZOBRIST_H + i]
      if (this.vWall[i]) h ^= z[ZOBRIST_V + i]
    }
    h ^= z[ZOBRIST_WALLS_LEFT + this.wallsLeft[0]]
    h ^= z[ZOBRIST_WALLS_LEFT + START_WALLS + 1 + this.wallsLeft[1]]
    if (this.turn) h ^= z[ZOBRIST_TURN]
    return h
  }

  hasReachedGoal(p: Player): boolean {
    return cellY(this.pawn[p]) === GOAL_ROW[p]
  }

  winner(): Player | -1 {
    if (this.hasReachedGoal(0)) return 0
    if (this.hasReachedGoal(1)) return 1
    return -1
  }

  canStep(x: number, y: number, dir: number): boolean {
    switch (dir) {
      case 0:
        return y > 0 && !((this.blockDown[y - 1] >> x) & 1)
      case 1:
        return x < SIZE - 1 && !((this.blockRight[y] >> x) & 1)
      case 2:
        return y < SIZE - 1 && !((this.blockDown[y] >> x) & 1)
      default:
        return x > 0 && !((this.blockRight[y] >> (x - 1)) & 1)
    }
  }

  pawnMoves(out: Int16Array, n = 0): number {
    const opp = this.pawn[this.turn ^ 1]
    const x = cellX(this.pawn[this.turn])
    const y = cellY(this.pawn[this.turn])
    for (let d = 0; d < 4; d++) {
      if (!this.canStep(x, y, d)) continue
      const nx = x + DX[d]
      const ny = y + DY[d]
      const nc = ny * SIZE + nx
      if (nc !== opp) {
        out[n++] = nc
      } else if (this.canStep(nx, ny, d)) {
        out[n++] = nc + DY[d] * SIZE + DX[d]
      } else {
        for (const side of [(d + 1) & 3, (d + 3) & 3]) {
          if (this.canStep(nx, ny, side)) out[n++] = (ny + DY[side]) * SIZE + nx + DX[side]
        }
      }
    }
    return n
  }

  wallFits(vertical: boolean, x: number, y: number): boolean {
    if (x < 0 || y < 0 || x >= WALL_SIZE || y >= WALL_SIZE) return false
    const i = y * WALL_SIZE + x
    if (this.hWall[i] || this.vWall[i]) return false
    if (vertical) {
      return (y === 0 || !this.vWall[i - WALL_SIZE]) && (y === WALL_SIZE - 1 || !this.vWall[i + WALL_SIZE])
    }
    return (x === 0 || !this.hWall[i - 1]) && (x === WALL_SIZE - 1 || !this.hWall[i + 1])
  }

  private touches(vx: number, vy: number): number {
    return vx === 0 || vy === 0 || vx === SIZE || vy === SIZE || this.wallTouch[vy * (SIZE + 1) + vx] > 0 ? 1 : 0
  }

  /** A wall can only cut off a region if it closes a loop, i.e. touches existing obstacles at two or more points. */
  private mayDisconnect(vertical: boolean, x: number, y: number): boolean {
    const contacts = vertical
      ? this.touches(x + 1, y) + this.touches(x + 1, y + 1) + this.touches(x + 1, y + 2)
      : this.touches(x, y + 1) + this.touches(x + 1, y + 1) + this.touches(x + 2, y + 1)
    return contacts >= 2
  }

  wallLegal(vertical: boolean, x: number, y: number): boolean {
    if (this.wallsLeft[this.turn] <= 0 || !this.wallFits(vertical, x, y)) return false
    return this.wallReachable(vertical, x, y)
  }

  /** Reachability only: caller has already rejected overlap and crossing. */
  private wallReachable(vertical: boolean, x: number, y: number): boolean {
    if (!this.mayDisconnect(vertical, x, y)) return true
    const i = y * WALL_SIZE + x
    this.setWall(vertical, i)
    const ok = this.distance(0) >= 0 && this.distance(1) >= 0
    this.clearWall(vertical, i)
    return ok
  }

  private setWall(vertical: boolean, i: number): void {
    const bits = vertical ? this.vBits : this.hBits
    bits[i >>> 5] |= 1 << (i & 31)
    const x = i % WALL_SIZE
    const y = (i / WALL_SIZE) | 0
    const stride = SIZE + 1
    if (vertical) {
      this.vWall[i] = 1
      this.blockRight[y] |= 1 << x
      this.blockRight[y + 1] |= 1 << x
      const v = y * stride + x + 1
      this.wallTouch[v]++
      this.wallTouch[v + stride]++
      this.wallTouch[v + 2 * stride]++
    } else {
      this.hWall[i] = 1
      this.blockDown[y] |= 3 << x
      const v = (y + 1) * stride + x
      this.wallTouch[v]++
      this.wallTouch[v + 1]++
      this.wallTouch[v + 2]++
    }
  }

  private clearWall(vertical: boolean, i: number): void {
    const bits = vertical ? this.vBits : this.hBits
    bits[i >>> 5] &= ~(1 << (i & 31))
    const x = i % WALL_SIZE
    const y = (i / WALL_SIZE) | 0
    const stride = SIZE + 1
    if (vertical) {
      this.vWall[i] = 0
      this.blockRight[y] &= ~(1 << x)
      this.blockRight[y + 1] &= ~(1 << x)
      const v = y * stride + x + 1
      this.wallTouch[v]--
      this.wallTouch[v + stride]--
      this.wallTouch[v + 2 * stride]--
    } else {
      this.hWall[i] = 0
      this.blockDown[y] &= ~(3 << x)
      const v = (y + 1) * stride + x
      this.wallTouch[v]--
      this.wallTouch[v + 1]--
      this.wallTouch[v + 2]--
    }
  }

  /** Shortest path length to the goal row ignoring pawns, -1 if unreachable. Bit-parallel flood fill over rows. */
  distance(p: Player): number {
    return scalarDistance(this, p)
  }

  /** Original row-array implementation, retained as a benchmark reference. */
  distanceReference(p: Player): number {
    const goal = GOAL_ROW[p]
    const c = this.pawn[p]
    if (cellY(c) === goal) return 0
    let f = this.floodA
    let next = this.floodB
    f.fill(0)
    f[cellY(c)] = 1 << cellX(c)
    for (let steps = 1; ; steps++) {
      let changed = false
      for (let i = 0; i < SIZE; i++) {
        const row = f[i]
        next[i] = row | (((row & ~this.blockRight[i]) << 1) & ROW_MASK) | ((row >> 1) & ~this.blockRight[i])
      }
      for (let i = 0; i < SIZE; i++) {
        const row = f[i]
        if (i > 0) next[i - 1] |= row & ~this.blockDown[i - 1]
        if (i < SIZE - 1) next[i + 1] |= row & ~this.blockDown[i]
      }
      if (next[goal]) {
        this.floodA = f
        this.floodB = next
        return steps
      }
      for (let i = 0; i < SIZE; i++) if (next[i] !== f[i]) changed = true
      if (!changed) {
        this.floodA = f
        this.floodB = next
        return -1
      }
      const t = f
      f = next
      next = t
    }
  }

  /** Distance to the goal row of `p` from every cell (-1 if unreachable). */
  distanceMap(p: Player, out: Int8Array): void {
    out.fill(-1)
    let head = 0
    let tail = 0
    const goal = GOAL_ROW[p]
    for (let x = 0; x < SIZE; x++) {
      out[goal * SIZE + x] = 0
      this.queue[tail++] = goal * SIZE + x
    }
    while (head < tail) {
      const c = this.queue[head++]
      const x = cellX(c)
      const y = cellY(c)
      for (let d = 0; d < 4; d++) {
        if (!this.canStep(x, y, d)) continue
        const n = c + DY[d] * SIZE + DX[d]
        if (out[n] >= 0) continue
        out[n] = out[c] + 1
        this.queue[tail++] = n
      }
    }
  }

  /** Deferred mode only omits reachability checks; overlap, crossing and inventory checks remain strict. */
  generate(out: Int16Array, likelyOnly: boolean, deferLegality = false): number {
    let n = this.pawnMoves(out)
    if (this.wallsLeft[this.turn] <= 0) return n
    const cand = likelyOnly ? this.markLikelyWalls() : null
    for (let limb = 0; limb < 2; limb++) {
      const h = this.hBits[limb]
      const v = this.vBits[limb]
      // Horizontal shifts cannot cross an 8-slot row. Vertical shifts carry between limbs.
      const hs = ~(h | v | ((h << 1) & 0xfefefefe) | ((h >>> 1) & 0x7f7f7f7f))
      const above = (v << 8) | (limb === 1 ? this.vBits[0] >>> 24 : 0)
      const below = (v >>> 8) | (limb === 0 ? this.vBits[1] << 24 : 0)
      const vs = ~(h | v | above | below)
      let available = hs | vs
      while (available !== 0) {
        const bit = available & -available
        const i = limb * 32 + 31 - Math.clz32(bit)
        const x = i % WALL_SIZE
        const y = i >>> 3
        if ((hs & bit) && (!cand || cand[i]) && (deferLegality || this.wallReachable(false, x, y))) out[n++] = H_BASE + i
        if ((vs & bit) && (!cand || cand[WALL_SLOTS + i]) && (deferLegality || this.wallReachable(true, x, y))) out[n++] = V_BASE + i
        available &= available - 1
      }
    }
    return n
  }

  /** Walls a human would almost always consider: next to a pawn, or extending/abutting an existing wall. */
  private markLikelyWalls(): Uint8Array {
    const cand = this.candidates
    cand.fill(0)
    const mark = (vertical: boolean, x: number, y: number) => {
      if (x >= 0 && y >= 0 && x < WALL_SIZE && y < WALL_SIZE) cand[(vertical ? WALL_SLOTS : 0) + y * WALL_SIZE + x] = 1
    }
    for (let p = 0; p < 2; p++) {
      const x = cellX(this.pawn[p])
      const y = cellY(this.pawn[p])
      for (const [dx, dy] of NEAR_PAWN) {
        mark(false, x + dx, y + dy)
        mark(true, x + dx, y + dy)
      }
    }
    for (let i = 0; i < WALL_SLOTS; i++) {
      const x = i % WALL_SIZE
      const y = (i / WALL_SIZE) | 0
      if (this.hWall[i]) for (const [dx, dy, v] of NEAR_WALL) mark(v, x + dx, y + dy)
      if (this.vWall[i]) for (const [dy, dx, v] of NEAR_WALL) mark(!v, x + dx, y + dy)
    }
    return cand
  }

  legalMoves(): Move[] {
    const out = new Int16Array(MAX_MOVES)
    return [...out.subarray(0, this.generate(out, false))]
  }

  make(m: Move): void {
    const t = this.turn
    if (m < H_BASE) {
      const from = this.pawn[t]
      this.fromStack[this.sp++] = from
      this.hash1 ^= Z1[ZOBRIST_PAWN + t * CELLS + from] ^ Z1[ZOBRIST_PAWN + t * CELLS + m]
      this.mirrorHash1 ^= M1[ZOBRIST_PAWN + t * CELLS + from] ^ M1[ZOBRIST_PAWN + t * CELLS + m]
      this.hash2 ^= Z2[ZOBRIST_PAWN + t * CELLS + from] ^ Z2[ZOBRIST_PAWN + t * CELLS + m]
      this.mirrorHash2 ^= M2[ZOBRIST_PAWN + t * CELLS + from] ^ M2[ZOBRIST_PAWN + t * CELLS + m]
      this.pawn[t] = m
    } else {
      const vertical = m >= V_BASE
      const i = vertical ? m - V_BASE : m - H_BASE
      const zw = (vertical ? ZOBRIST_V : ZOBRIST_H) + i
      const zl = ZOBRIST_WALLS_LEFT + t * (START_WALLS + 1) + this.wallsLeft[t]
      this.hash1 ^= Z1[zw] ^ Z1[zl] ^ Z1[zl - 1]
      this.mirrorHash1 ^= M1[zw] ^ M1[zl] ^ M1[zl - 1]
      this.hash2 ^= Z2[zw] ^ Z2[zl] ^ Z2[zl - 1]
      this.mirrorHash2 ^= M2[zw] ^ M2[zl] ^ M2[zl - 1]
      this.setWall(vertical, i)
      this.wallsLeft[t]--
    }
    this.turn = (t ^ 1) as Player
    this.hash1 ^= Z1[ZOBRIST_TURN]
    this.mirrorHash1 ^= M1[ZOBRIST_TURN]
    this.hash2 ^= Z2[ZOBRIST_TURN]
    this.mirrorHash2 ^= M2[ZOBRIST_TURN]
  }

  unmake(m: Move): void {
    const t = (this.turn ^ 1) as Player
    this.turn = t
    this.hash1 ^= Z1[ZOBRIST_TURN]
    this.mirrorHash1 ^= M1[ZOBRIST_TURN]
    this.hash2 ^= Z2[ZOBRIST_TURN]
    this.mirrorHash2 ^= M2[ZOBRIST_TURN]
    if (m < H_BASE) {
      const from = this.fromStack[--this.sp]
      this.hash1 ^= Z1[ZOBRIST_PAWN + t * CELLS + from] ^ Z1[ZOBRIST_PAWN + t * CELLS + m]
      this.mirrorHash1 ^= M1[ZOBRIST_PAWN + t * CELLS + from] ^ M1[ZOBRIST_PAWN + t * CELLS + m]
      this.hash2 ^= Z2[ZOBRIST_PAWN + t * CELLS + from] ^ Z2[ZOBRIST_PAWN + t * CELLS + m]
      this.mirrorHash2 ^= M2[ZOBRIST_PAWN + t * CELLS + from] ^ M2[ZOBRIST_PAWN + t * CELLS + m]
      this.pawn[t] = from
    } else {
      const vertical = m >= V_BASE
      const i = vertical ? m - V_BASE : m - H_BASE
      this.wallsLeft[t]++
      this.clearWall(vertical, i)
      const zw = (vertical ? ZOBRIST_V : ZOBRIST_H) + i
      const zl = ZOBRIST_WALLS_LEFT + t * (START_WALLS + 1) + this.wallsLeft[t]
      this.hash1 ^= Z1[zw] ^ Z1[zl] ^ Z1[zl - 1]
      this.mirrorHash1 ^= M1[zw] ^ M1[zl] ^ M1[zl - 1]
      this.hash2 ^= Z2[zw] ^ Z2[zl] ^ Z2[zl - 1]
      this.mirrorHash2 ^= M2[zw] ^ M2[zl] ^ M2[zl - 1]
    }
  }

  checkHash(): boolean {
    return this.hash1 === this.computeHash(Z1) && this.hash2 === this.computeHash(Z2)
      && this.mirrorHash1 === this.computeHash(M1) && this.mirrorHash2 === this.computeHash(M2)
  }
}

const NEAR_PAWN: readonly (readonly [number, number])[] = [
  [-1, 0],
  [0, 0],
  [-1, -1],
  [0, -1],
]

/** Offsets around a horizontal wall [dx, dy, vertical]; transposed for vertical walls. */
const NEAR_WALL: readonly (readonly [number, number, boolean])[] = [
  [-2, 0, false],
  [2, 0, false],
  [-1, -1, true],
  [0, -1, true],
  [1, -1, true],
  [-1, 1, true],
  [0, 1, true],
  [1, 1, true],
]
