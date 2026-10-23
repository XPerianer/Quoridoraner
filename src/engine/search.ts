import { Board, CELLS, H_BASE, MAX_MOVES, MOVE_SPACE, V_BASE, WALL_SLOTS } from './board'
import type {
  Bound,
  EngineBackend,
  IterationStat,
  Move,
  Position,
  ReplyStat,
  RootMoveStat,
  SearchOptions,
  SearchSnapshot,
} from './types'

export const WIN = 10000
export const STEP = 10
const INF = 32000
const MAX_PLY = 64
const MATE_BOUND = WIN - MAX_PLY
const TT_BITS = 20
const TT_MASK = (1 << TT_BITS) - 1
const TT_EXACT = 0
const TT_LOWER = 1
const TT_UPPER = 2
const MAX_REPLIES_REPORTED = 32

export const DEFAULT_OPTIONS: SearchOptions = {
  maxDepth: 12,
  timeMs: 2000,
  maxNodes: Number.MAX_SAFE_INTEGER,
  moveGen: 'all',
  exactRoot: false,
  wallWeight: 10,
  tempo: 0,
  reportEveryMs: 80,
}

export const isWinScore = (s: number) => Math.abs(s) >= MATE_BOUND

class Aborted extends Error {}

export class TsSearcher implements EngineBackend {
  readonly name = 'typescript'

  private readonly boardFactory: (position: Position) => Board

  constructor(boardFactory: (position: Position) => Board = Board.from) {
    this.boardFactory = boardFactory
  }

  private board = Board.initial()
  private opts!: SearchOptions
  private readonly moveBuf = Array.from({ length: MAX_PLY }, () => new Int16Array(MAX_MOVES))
  private readonly orderBuf = Array.from({ length: MAX_PLY }, () => new Int32Array(MAX_MOVES))
  private readonly distMap = new Int8Array(CELLS)
  private readonly history = new Int32Array(MOVE_SPACE)

  private readonly ttKey = new Int32Array(1 << TT_BITS)
  private readonly ttDepth = new Int8Array(1 << TT_BITS).fill(-1)
  private readonly ttFlag = new Uint8Array(1 << TT_BITS)
  private readonly ttScore = new Int16Array(1 << TT_BITS)
  private readonly ttMove = new Int16Array(1 << TT_BITS)

  private start = 0
  private lastReport = 0
  private canAbort = false
  private nodes = 0
  private ttProbes = 0
  private ttHits = 0
  private plyNodes: number[] = []
  private plyCutoffs: number[] = []
  private pawnHeat: [number[], number[]] = [[], []]
  private wallHeat: [number[], number[]] = [[], []]
  private root: RootMoveStat[] = []
  private iterations: IterationStat[] = []
  private replies: ReplyStat[] = []
  private replyCount = 0
  private repliesSearched = 0
  private bestMove: Move = -1
  private bestScore = 0
  private pv: Move[] = []
  private depth = 0
  private currentIndex = -1
  private onProgress: (s: SearchSnapshot) => void = () => {}

  search(position: Position, options: SearchOptions, onProgress: (s: SearchSnapshot) => void): SearchSnapshot {
    this.board = this.boardFactory(position)
    if (this.board.winner() !== -1) throw new Error('search called on a finished game')
    this.opts = options
    this.onProgress = onProgress
    this.start = performance.now()
    this.lastReport = this.start
    this.nodes = this.ttProbes = this.ttHits = 0
    this.plyNodes = new Array(MAX_PLY).fill(0)
    this.plyCutoffs = new Array(MAX_PLY).fill(0)
    this.pawnHeat = [new Array(CELLS).fill(0), new Array(CELLS).fill(0)]
    this.wallHeat = [new Array(WALL_SLOTS).fill(0), new Array(WALL_SLOTS).fill(0)]
    this.iterations = []
    this.history.fill(0)
    this.pv = []

    const n = this.orderMoves(0, this.probeMove())
    if (n === 0) throw new Error('no legal moves')
    this.root = [...this.moveBuf[0].subarray(0, n)].map((move) => ({
      move,
      score: 0,
      bound: 'exact' as Bound,
      nodes: 0,
      depth: 0,
      replies: [],
      replyCount: 0,
      repliesSearched: 0,
    }))
    this.bestMove = this.root[0].move
    this.bestScore = 0

    for (let depth = 1; depth <= options.maxDepth; depth++) {
      this.depth = depth
      this.canAbort = depth > 1
      const iterStart = performance.now()
      const nodesBefore = this.nodes
      try {
        this.searchRoot(depth)
      } catch (e) {
        if (!(e instanceof Aborted)) throw e
        break
      }
      this.pv = this.extractPv(depth)
      this.iterations.push({
        depth,
        nodes: this.nodes - nodesBefore,
        timeMs: performance.now() - iterStart,
        bestMove: this.bestMove,
        score: this.bestScore,
        pv: this.pv,
      })
      this.root.sort((a, b) => rankScore(b) - rankScore(a))
      if (isWinScore(this.bestScore) || performance.now() - this.start > options.timeMs || this.nodes >= options.maxNodes) break
    }
    this.currentIndex = -1
    return this.snapshot(true)
  }

  private searchRoot(depth: number): void {
    const b = this.board
    let alpha = -INF
    let iterBest: Move = -1
    let iterScore = -INF
    for (let i = 0; i < this.root.length; i++) {
      const st = this.root[i]
      this.currentIndex = i
      this.replies = []
      this.replyCount = this.repliesSearched = 0
      const before = this.nodes
      b.make(st.move)
      this.heat(st.move)
      const full = this.opts.exactRoot || i === 0
      const s = -this.negamax(depth - 1, -INF, full ? INF : -alpha, 1)
      b.unmake(st.move)
      st.score = s
      st.bound = full || s > alpha ? 'exact' : 'upper'
      st.nodes = this.nodes - before
      st.depth = depth
      st.replies = this.replies.sort((a, c) => c.nodes - a.nodes)
      st.replyCount = this.replyCount
      st.repliesSearched = this.repliesSearched
      if (s > iterScore) {
        iterScore = s
        iterBest = st.move
        this.bestMove = iterBest
        this.bestScore = iterScore
      }
      if (s > alpha) alpha = s
    }
  }

  private negamax(depth: number, alpha: number, beta: number, ply: number): number {
    const b = this.board
    this.nodes++
    this.plyNodes[ply]++
    if ((this.nodes & 1023) === 0) this.tick()
    if (b.hasReachedGoal((b.turn ^ 1) as 0 | 1)) return -(WIN - ply)
    if (depth === 0 || ply >= MAX_PLY - 1) return this.evaluate()

    const idx = b.hash1 & TT_MASK
    let ttMove = -1
    this.ttProbes++
    if (this.ttKey[idx] === b.hash2 && this.ttDepth[idx] >= 0) {
      this.ttHits++
      ttMove = this.ttMove[idx]
      if (this.ttDepth[idx] >= depth && ply > 1) {
        const s = fromTt(this.ttScore[idx], ply)
        const flag = this.ttFlag[idx]
        if (flag === TT_EXACT || (flag === TT_LOWER && s >= beta) || (flag === TT_UPPER && s <= alpha)) return s
      }
    }

    const n = this.orderMoves(ply, ttMove)
    if (n === 0) return this.evaluate()
    const moves = this.moveBuf[ply]
    const alpha0 = alpha
    const record = ply === 1
    let best = -INF
    let bestMove = moves[0]
    let i = 0
    for (; i < n; i++) {
      const m = moves[i]
      const before = this.nodes
      b.make(m)
      this.heat(m)
      const s = -this.negamax(depth - 1, -beta, -alpha, ply + 1)
      b.unmake(m)
      if (record) {
        const bound: Bound = s <= alpha ? 'lower' : s >= beta ? 'upper' : 'exact'
        this.replies.push({ move: m, score: -s, bound, nodes: this.nodes - before })
      }
      if (s > best) {
        best = s
        bestMove = m
      }
      if (s > alpha) alpha = s
      if (alpha >= beta) {
        this.plyCutoffs[ply]++
        if (m >= H_BASE) this.history[m] += depth * depth
        i++
        break
      }
    }
    if (record) {
      this.replyCount = n
      this.repliesSearched = Math.min(i, n)
    }

    this.ttKey[idx] = b.hash2
    this.ttDepth[idx] = depth
    this.ttFlag[idx] = best <= alpha0 ? TT_UPPER : best >= beta ? TT_LOWER : TT_EXACT
    this.ttScore[idx] = toTt(best, ply)
    this.ttMove[idx] = bestMove
    return best
  }

  private evaluate(): number {
    const b = this.board
    const me = b.turn
    const opp = (me ^ 1) as 0 | 1
    return STEP * (b.distance(opp) - b.distance(me)) + this.opts.wallWeight * (b.wallsLeft[me] - b.wallsLeft[opp]) + this.opts.tempo
  }

  private orderMoves(ply: number, ttMove: Move): number {
    const b = this.board
    const moves = this.moveBuf[ply]
    const keys = this.orderBuf[ply]
    const n = b.generate(moves, this.opts.moveGen === 'likely')
    b.distanceMap(b.turn, this.distMap)
    for (let i = 0; i < n; i++) {
      const m = moves[i]
      keys[i] = m === ttMove ? 1 << 30 : m < H_BASE ? (1 << 29) - this.distMap[m] : this.history[m]
    }
    for (let i = 1; i < n; i++) {
      const m = moves[i]
      const k = keys[i]
      let j = i - 1
      while (j >= 0 && keys[j] < k) {
        moves[j + 1] = moves[j]
        keys[j + 1] = keys[j]
        j--
      }
      moves[j + 1] = m
      keys[j + 1] = k
    }
    return n
  }

  private probeMove(): Move {
    const idx = this.board.hash1 & TT_MASK
    return this.ttKey[idx] === this.board.hash2 && this.ttDepth[idx] >= 0 ? this.ttMove[idx] : -1
  }

  private extractPv(depth: number): Move[] {
    const b = this.board
    const pv: Move[] = [this.bestMove]
    b.make(this.bestMove)
    while (pv.length < depth && b.winner() === -1) {
      const m = this.probeMove()
      if (m < 0 || !b.legalMoves().includes(m)) break
      pv.push(m)
      b.make(m)
    }
    for (let i = pv.length - 1; i >= 0; i--) b.unmake(pv[i])
    return pv
  }

  private heat(m: Move): void {
    if (m < H_BASE) this.pawnHeat[this.board.turn ^ 1][m]++
    else if (m < V_BASE) this.wallHeat[0][m - H_BASE]++
    else this.wallHeat[1][m - V_BASE]++
  }

  private tick(): void {
    const now = performance.now()
    if (this.canAbort && (now - this.start > this.opts.timeMs || this.nodes >= this.opts.maxNodes)) throw new Aborted()
    if (now - this.lastReport >= this.opts.reportEveryMs) {
      this.lastReport = now
      this.onProgress(this.snapshot(false))
    }
  }

  private snapshot(done: boolean): SearchSnapshot {
    const current = this.root[this.currentIndex]
    return {
      done,
      backend: this.name,
      depth: this.depth,
      nodes: this.nodes,
      elapsedMs: performance.now() - this.start,
      bestMove: this.bestMove,
      score: this.bestScore,
      pv: this.pv,
      currentMove: current ? current.move : -1,
      currentIndex: this.currentIndex,
      root: this.root.map((r) => ({ ...r, replies: r.replies.slice(0, MAX_REPLIES_REPORTED) })),
      iterations: this.iterations.slice(),
      plyNodes: trimZeros(this.plyNodes),
      plyCutoffs: this.plyCutoffs.slice(0, trimZeros(this.plyNodes).length),
      ttProbes: this.ttProbes,
      ttHits: this.ttHits,
      pawnHeat: [this.pawnHeat[0].slice(), this.pawnHeat[1].slice()],
      wallHeat: [this.wallHeat[0].slice(), this.wallHeat[1].slice()],
    }
  }
}

function rankScore(r: RootMoveStat): number {
  return r.bound === 'exact' ? r.score * 2 + 1 : r.score * 2
}

function toTt(s: number, ply: number): number {
  return s >= MATE_BOUND ? s + ply : s <= -MATE_BOUND ? s - ply : s
}

function fromTt(s: number, ply: number): number {
  return s >= MATE_BOUND ? s - ply : s <= -MATE_BOUND ? s + ply : s
}

function trimZeros(a: number[]): number[] {
  let end = a.length
  while (end > 0 && a[end - 1] === 0) end--
  return a.slice(0, end)
}
