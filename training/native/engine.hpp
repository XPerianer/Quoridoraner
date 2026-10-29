#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace quoridor {
constexpr int DX[4] = {0, 1, 0, -1};
constexpr int DY[4] = {-1, 0, 1, 0};
constexpr int ZLEN = 313;
inline std::array<uint32_t, ZLEN> zobrist(uint32_t seed) {
    std::array<uint32_t, ZLEN> result{};
    for (auto& value : result) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; value = seed; }
    return result;
}
inline const auto Z1 = zobrist(0x9e3779b9u);
inline const auto Z2 = zobrist(0x85ebca6bu);
using Bits = unsigned __int128;
constexpr Bits rightEdges() {
    Bits mask = 0;
    for (int y = 0; y < 9; y++) mask |= Bits(255) << (9*y);
    return mask;
}
inline int mirrorMove(int move) {
    if (move < 0) return move;
    if (move < 81) return move / 9 * 9 + 8 - move % 9;
    int base = move >= 145 ? 145 : 81, i = move-base;
    return base + i / 8 * 8 + 7 - i % 8;
}
struct Board {
    int pawn[2] = {76, 4}, walls[2] = {10, 10}, turn = 0;
    std::array<uint8_t, 64> h{}, v{};
    std::array<uint16_t, 9> down{}, right{};
    std::array<uint8_t, 100> touch{};
    uint32_t hash1 = 0, hash2 = 0, mirrorHash1 = 0, mirrorHash2 = 0;
    uint64_t hBits = 0, vBits = 0;
    Bits openRight = rightEdges(), openDown = (Bits(1) << 72)-1;
    int from[128]{}, sp = 0;

    bool goal(int p) const { return pawn[p] / 9 == (p == 0 ? 0 : 8); }
    int winner() const { return goal(0) ? 0 : goal(1) ? 1 : -1; }
    bool canStep(int x, int y, int d) const {
        switch (d) {
            case 0: return y > 0 && !(down[y-1] & (1 << x));
            case 1: return x < 8 && !(right[y] & (1 << x));
            case 2: return y < 8 && !(down[y] & (1 << x));
            default: return x > 0 && !(right[y] & (1 << (x-1)));
        }
    }
    void setWall(bool vertical, int i, bool add) {
        const int x = i % 8, y = i / 8, delta = add ? 1 : -1;
        if (vertical) {
            v[i] = add;
            if (add) vBits |= uint64_t(1) << i; else vBits &= ~(uint64_t(1) << i);
            const Bits edges = (Bits(1) << (y*9+x)) | (Bits(1) << ((y+1)*9+x));
            if (add) openRight &= ~edges; else openRight |= edges;
            if (add) { right[y] |= 1 << x; right[y+1] |= 1 << x; }
            else { right[y] &= ~(1 << x); right[y+1] &= ~(1 << x); }
            for (int k = 0; k < 3; k++) touch[(y+k)*10+x+1] += delta;
        } else {
            h[i] = add;
            if (add) hBits |= uint64_t(1) << i; else hBits &= ~(uint64_t(1) << i);
            const Bits edges = Bits(3) << (y*9+x);
            if (add) openDown &= ~edges; else openDown |= edges;
            if (add) down[y] |= 3 << x; else down[y] &= ~(3 << x);
            for (int k = 0; k < 3; k++) touch[(y+1)*10+x+k] += delta;
        }
    }
    bool fits(bool vertical, int x, int y) const {
        int i = y * 8 + x;
        if (h[i] || v[i]) return false;
        return vertical ? (y == 0 || !v[i-8]) && (y == 7 || !v[i+8])
                        : (x == 0 || !h[i-1]) && (x == 7 || !h[i+1]);
    }
    bool touches(int x, int y) const { return x == 0 || y == 0 || x == 9 || y == 9 || touch[y*10+x]; }
    int distance(int p) const {
        // All 81 cells fit in a native two-register bitboard: no per-row wave loop.
        const Bits goalMask = Bits(511) << (p == 0 ? 0 : 72);
        Bits reached = Bits(1) << pawn[p];
        if (reached & goalMask) return 0;
        for (int steps = 1; steps <= 81; steps++) {
            Bits next = reached | ((reached & openRight) << 1) | ((reached >> 1) & openRight)
                                | ((reached & openDown) << 9) | ((reached >> 9) & openDown);
            if (next & goalMask) return steps;
            if (next == reached) return -1;
            reached = next;
        }
        return -1;
    }
    bool legalWall(bool vertical, int i) {
        const int x = i % 8, y = i / 8;
        if (walls[turn] <= 0 || !fits(vertical, x, y)) return false;
        int contacts = vertical ? touches(x+1,y)+touches(x+1,y+1)+touches(x+1,y+2)
                                : touches(x,y+1)+touches(x+1,y+1)+touches(x+2,y+1);
        if (contacts < 2) return true;
        setWall(vertical, i, true);
        bool ok = distance(0) >= 0 && distance(1) >= 0;
        setWall(vertical, i, false);
        return ok;
    }
    int generate(int* out, bool deferLegality = false) {
        int n = 0, x = pawn[turn] % 9, y = pawn[turn] / 9;
        for (int d = 0; d < 4; d++) {
            if (!canStep(x,y,d)) continue;
            int nx = x+DX[d], ny = y+DY[d], cell = ny*9+nx;
            if (cell != pawn[turn^1]) out[n++] = cell;
            else if (canStep(nx,ny,d)) out[n++] = cell+DY[d]*9+DX[d];
            else for (int side : {(d+1)&3, (d+3)&3})
                if (canStep(nx,ny,side)) out[n++] = (ny+DY[side])*9+nx+DX[side];
        }
        if (walls[turn]) {
            const uint64_t occupied = hBits | vBits;
            const uint64_t hCandidates = ~(occupied | ((hBits & 0x7f7f7f7f7f7f7f7full) << 1)
                                                    | ((hBits & 0xfefefefefefefefeull) >> 1));
            const uint64_t vCandidates = ~(occupied | (vBits << 8) | (vBits >> 8));
            uint64_t candidates = hCandidates | vCandidates;
            while (candidates) {
                int i = __builtin_ctzll(candidates); uint64_t bit = uint64_t(1) << i;
                candidates &= candidates-1;
                if ((hCandidates & bit) && (deferLegality || legalWall(false,i))) out[n++] = 81+i;
                if ((vCandidates & bit) && (deferLegality || legalWall(true,i))) out[n++] = 145+i;
            }
        }
        return n;
    }
    void distanceMap(int p, int* dist) const {
        std::fill(dist, dist+81, -1);
        int queue[81], head = 0, tail = 0, row = p == 0 ? 0 : 8;
        for (int x = 0; x < 9; x++) { dist[row*9+x] = 0; queue[tail++] = row*9+x; }
        while (head < tail) {
            int c = queue[head++], x = c%9, y = c/9;
            for (int d = 0; d < 4; d++) {
                if (!canStep(x,y,d)) continue;
                int next = c + DY[d]*9 + DX[d];
                if (dist[next] >= 0) continue;
                dist[next] = dist[c]+1;
                queue[tail++] = next;
            }
        }
    }
    uint32_t hash(const std::array<uint32_t, ZLEN>& z, bool mirror = false) const {
        uint32_t value = z[mirror ? mirrorMove(pawn[0]) : pawn[0]] ^ z[81+(mirror ? mirrorMove(pawn[1]) : pawn[1])] ^ z[290+walls[0]] ^ z[301+walls[1]];
        for (int i = 0; i < 64; i++) { int j = mirror ? i / 8 * 8 + 7 - i % 8 : i; if (h[i]) value ^= z[162+j]; if (v[i]) value ^= z[226+j]; }
        if (turn) value ^= z[312];
        return value;
    }
    void initHash() { hash1 = hash(Z1); hash2 = hash(Z2); mirrorHash1 = hash(Z1,true); mirrorHash2 = hash(Z2,true); }
    void mirrorUpdate(int move, int old, int wallCount) {
        if (move < 81) {
            mirrorHash1 ^= Z1[turn*81+mirrorMove(old)] ^ Z1[turn*81+mirrorMove(move)];
            mirrorHash2 ^= Z2[turn*81+mirrorMove(old)] ^ Z2[turn*81+mirrorMove(move)];
        } else {
            int zw = 162 + mirrorMove(move)-81, zl = 290+turn*11+wallCount;
            mirrorHash1 ^= Z1[zw] ^ Z1[zl] ^ Z1[zl-1];
            mirrorHash2 ^= Z2[zw] ^ Z2[zl] ^ Z2[zl-1];
        }
        mirrorHash1 ^= Z1[312]; mirrorHash2 ^= Z2[312];
    }
    void make(int move) {
        mirrorUpdate(move,pawn[turn],walls[turn]);
        if (move < 81) {
            int old = pawn[turn]; from[sp++] = old;
            hash1 ^= Z1[turn*81+old] ^ Z1[turn*81+move];
            hash2 ^= Z2[turn*81+old] ^ Z2[turn*81+move];
            pawn[turn] = move;
        } else {
            bool vertical = move >= 145; int i = move-(vertical ? 145 : 81);
            int zw = (vertical ? 226 : 162)+i, zl = 290+turn*11+walls[turn];
            hash1 ^= Z1[zw] ^ Z1[zl] ^ Z1[zl-1]; hash2 ^= Z2[zw] ^ Z2[zl] ^ Z2[zl-1];
            setWall(vertical,i,true); walls[turn]--;
        }
        turn ^= 1; hash1 ^= Z1[312]; hash2 ^= Z2[312];
    }
    void unmake(int move) {
        turn ^= 1; hash1 ^= Z1[312]; hash2 ^= Z2[312];
        if (move < 81) {
            int old = from[--sp];
            mirrorUpdate(move,old,walls[turn]);
            hash1 ^= Z1[turn*81+old] ^ Z1[turn*81+move];
            hash2 ^= Z2[turn*81+old] ^ Z2[turn*81+move]; pawn[turn] = old;
        } else {
            bool vertical = move >= 145; int i = move-(vertical ? 145 : 81);
            walls[turn]++; mirrorUpdate(move,0,walls[turn]); setWall(vertical,i,false);
            int zw = (vertical ? 226 : 162)+i, zl = 290+turn*11+walls[turn];
            hash1 ^= Z1[zw] ^ Z1[zl] ^ Z1[zl-1]; hash2 ^= Z2[zw] ^ Z2[zl] ^ Z2[zl-1];
        }
    }
};

struct SearchResult { int move, score; int64_t nodes; };
class Searcher {
    static constexpr int INVALID = 32001, INF = 32000, WIN = 10000, MATE = 9936, MASK = (1<<20)-1;
    struct Entry { uint32_t key = 0, key1 = 0, generation = 0; int16_t score = 0, move = 0; int8_t depth = -1; uint8_t flag = 0; };
    struct Root { int move, score = 0; bool exact = true; };
    struct Abort {};
    std::vector<Entry> tt = std::vector<Entry>(1<<20);
    Board b;
    uint32_t generation = 0;
    int history[209]{}, moves[64][136]{}, keys[64][136]{}, dist[81]{}, opponentDist[81]{}, pathWalls[2][209]{};
    int64_t nodes = 0, budget = 0;
    bool canAbort = false;
    int bestMove = -1, bestScore = 0;
    void markPathWalls(int cell, const int* distances, int* out) {
        std::fill(out,out+209,0);
        while (distances[cell] > 0) {
            int x = cell%9, y = cell/9, next = -1;
            for (int d = 0; d < 4; d++) {
                if (!b.canStep(x,y,d)) continue;
                int candidate = cell+DY[d]*9+DX[d];
                if (distances[candidate] == distances[cell]-1) { next = candidate; break; }
            }
            if (next < 0) break;
            int nx = next%9, ny = next/9;
            if (x == nx) {
                int row = std::min(y,ny);
                if (x < 8) out[81+row*8+x]++;
                if (x > 0) out[81+row*8+x-1]++;
            } else {
                int col = std::min(x,nx);
                if (y < 8) out[145+y*8+col]++;
                if (y > 0) out[145+(y-1)*8+col]++;
            }
            cell = next;
        }
    }
    int order(int ply, int ttMove, int depth) {
        int n = b.generate(moves[ply],ply > 1); b.distanceMap(b.turn, dist);
        bool orderWalls = depth > 1 && b.walls[b.turn] > 0;
        if (orderWalls) {
            b.distanceMap(b.turn^1,opponentDist);
            markPathWalls(b.pawn[b.turn],dist,pathWalls[0]);
            markPathWalls(b.pawn[b.turn^1],opponentDist,pathWalls[1]);
        }
        for (int i = 0; i < n; i++) {
            int m = moves[ply][i]; keys[ply][i] = m == ttMove ? 1<<30 : m < 81 ? (1<<29)-dist[m] : history[m]+(orderWalls ? 64*(pathWalls[1][m]-pathWalls[0][m]) : 0);
        }
        for (int i = 1; i < n; i++) {
            int m = moves[ply][i], k = keys[ply][i], j = i-1;
            while (j >= 0 && keys[ply][j] < k) { moves[ply][j+1] = moves[ply][j]; keys[ply][j+1] = keys[ply][j]; j--; }
            moves[ply][j+1] = m; keys[ply][j+1] = k;
        }
        return n;
    }
    int evaluate() {
        int mine = b.distance(b.turn), theirs = b.distance(b.turn^1);
        return mine < 0 || theirs < 0 ? INVALID : 10*(theirs-mine) + 10*(b.walls[b.turn]-b.walls[b.turn^1]);
    }
    int negamax(int depth, int alpha, int beta, int ply) {
        nodes++;
        if ((nodes & 1023) == 0 && canAbort && nodes >= budget) throw Abort{};
        if (b.goal(b.turn^1)) return b.distance(0) < 0 || b.distance(1) < 0 ? INVALID : -(WIN-ply);
        if (depth == 0 || ply >= 63) return evaluate();
        bool mirrored = b.mirrorHash1 < b.hash1 || (b.mirrorHash1 == b.hash1 && b.mirrorHash2 < b.hash2);
        uint32_t hash1 = mirrored ? b.mirrorHash1 : b.hash1, hash2 = mirrored ? b.mirrorHash2 : b.hash2;
        auto& entry = tt[hash1 & MASK];
        int ttMove = -1;
        if (entry.generation == generation && entry.key1 == hash1 && entry.key == hash2 && entry.depth >= 0) {
            ttMove = mirrored ? mirrorMove(entry.move) : entry.move;
            if (entry.depth >= depth && ply > 1) {
                int s = entry.score >= MATE ? entry.score-ply : entry.score <= -MATE ? entry.score+ply : entry.score;
                if (entry.flag == 0 || (entry.flag == 1 && s >= beta) || (entry.flag == 2 && s <= alpha)) return s;
            }
        }
        int n = order(ply,ttMove,depth);
        if (!n) return evaluate();
        int alpha0 = alpha, best = -INF, chosen = moves[ply][0];
        bool parentChecked = false;
        for (int i = 0; i < n; i++) {
            int m = moves[ply][i]; b.make(m);
            int child;
            try { child = negamax(depth-1,-beta,-alpha,ply+1); }
            catch (...) { b.unmake(m); throw; }
            b.unmake(m);
            if (child == INVALID) {
                if (!parentChecked) {
                    if (b.distance(0) < 0 || b.distance(1) < 0) return INVALID;
                    parentChecked = true;
                }
                continue;
            }
            parentChecked = true;
            int s = -child;
            if (s > best) { best = s; chosen = m; }
            alpha = std::max(alpha,s);
            if (alpha >= beta) { if (m >= 81) history[m] += depth*depth; break; }
        }
        if (best == -INF) return evaluate();
        entry.generation = generation; entry.key1 = hash1; entry.key = hash2; entry.depth = depth; entry.move = mirrored ? mirrorMove(chosen) : chosen;
        entry.flag = best <= alpha0 ? 2 : best >= beta ? 1 : 0;
        entry.score = best >= MATE ? best+ply : best <= -MATE ? best-ply : best;
        return best;
    }
public:
    SearchResult search(Board position, int64_t nodeBudget, int maxDepth = 64) {
        b = position; b.initHash(); b.sp = 0; budget = nodeBudget; nodes = 0;
        // A new generation is logically an empty TT, without touching every cache line.
        if (++generation == 0) {
            for (auto& entry : tt) entry.generation = 0;
            generation = 1;
        }
        std::fill(std::begin(history),std::end(history),0);
        if (b.winner() != -1) throw std::runtime_error("Search on finished game");
        int n = order(0,-1,maxDepth); std::vector<Root> root; root.reserve(n);
        for (int i = 0; i < n; i++) root.push_back({moves[0][i]});
        if (root.empty()) throw std::runtime_error("No legal moves");
        bestMove = root[0].move; bestScore = 0;
        for (int depth = 1; depth <= maxDepth; depth++) {
            canAbort = depth > 1; int alpha = -INF, iterScore = -INF;
            try {
                for (int i = 0; i < n; i++) {
                    auto& st = root[i]; b.make(st.move);
                    int child;
                    try { child = negamax(depth-1,-INF,i == 0 ? INF : -alpha,1); }
                    catch (...) { b.unmake(st.move); throw; }
                    b.unmake(st.move);
                    if (child == INVALID) throw std::runtime_error("Illegal root position");
                    int score = -child;
                    st.score = score; st.exact = i == 0 || score > alpha;
                    if (score > iterScore) { iterScore = score; bestMove = st.move; bestScore = score; }
                    alpha = std::max(alpha,score);
                }
            } catch (const Abort&) { break; }
            std::stable_sort(root.begin(),root.end(),[](const Root& a,const Root& c) {
                return a.score*2+int(a.exact) > c.score*2+int(c.exact);
            });
            if (std::abs(bestScore) >= MATE || nodes >= budget) break;
        }
        return {bestMove,bestScore,nodes};
    }
};
}
