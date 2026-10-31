#pragma once

// Exact zero-wall Quoridor pawn-race tablebase, enabled by the tablebase search mode.
// The graph is the product of two pawn squares and side to move for one fixed
// wall geometry. Prior twice-seen states are draw sinks; decisive DTM strategies
// strictly decrease distance and cannot create new threefold repetitions.
#include "engine.hpp"
#include <array>
#include <cstdint>
#include <algorithm>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace quoridor::race_tablebase {

struct Position {
    uint8_t pawn0=0, pawn1=0, turn=0;
    bool operator==(const Position& o) const {
        return pawn0==o.pawn0 && pawn1==o.pawn1 && turn==o.turn;
    }
};

struct HistoryEntry {
    Position position;
    uint8_t count=1;
    uint64_t hBits=0, vBits=0;
};
struct Result { int outcome=0; int distance=0; int states=0; int workStates=0; bool usedHistoryTable=false; };

inline uint16_t indexOf(Position p) {
    return uint16_t((uint16_t(p.pawn0)*81u + p.pawn1)*2u + p.turn);
}
inline Position decode(uint16_t i) {
    Position p; p.turn=i&1; i>>=1; p.pawn1=i%81; p.pawn0=i/81; return p;
}

inline bool canPawnStep(const Board& layout,int cell,int direction) {
    int x=cell%9,y=cell/9;
    switch(direction) {
        case 0: return y>0 && ((layout.openDown>>(cell-9))&1);
        case 1: return x<8 && ((layout.openRight>>cell)&1);
        case 2: return y<8 && ((layout.openDown>>cell)&1);
        default: return x>0 && ((layout.openRight>>(cell-1))&1);
    }
}

// The race graph has no wall-placement actions. Use wall-edge bitboards directly
// and mirror Board::generate's straight-jump and diagonal-fallback rules.
inline int generatePawnMoves(const Board& layout,Position p,int* out) {
    constexpr int dx[4]={0,1,0,-1}, dy[4]={-1,0,1,0};
    int n=0, from=p.turn==0?p.pawn0:p.pawn1, opponent=p.turn==0?p.pawn1:p.pawn0;
    for(int d=0;d<4;d++) {
        if(!canPawnStep(layout,from,d)) continue;
        int adjacent=from+dy[d]*9+dx[d];
        if(adjacent!=opponent) out[n++]=adjacent;
        else if(canPawnStep(layout,opponent,d)) out[n++]=opponent+dy[d]*9+dx[d];
        else for(int side:{(d+1)&3,(d+3)&3}) if(canPawnStep(layout,opponent,side))
            out[n++]=opponent+dy[side]*9+dx[side];
    }
    return n;
}

class Graph {
public:
    using Edges=std::vector<uint16_t>;
    Board layout;
    std::vector<uint8_t> valid;
    std::vector<Edges> next, previous;
    mutable std::mutex baseMutex;
    struct Labels { std::vector<int8_t> outcome; std::vector<uint16_t> dtm; };
    mutable std::shared_ptr<Labels> baseLabels;
    struct HistoryLabelEntry { std::vector<uint16_t> forbidden; std::shared_ptr<Labels> labels; };
    mutable std::list<HistoryLabelEntry> historyLabels;
    mutable std::mutex historyMutex;

    explicit Graph(Board b):layout(std::move(b)),valid(81*81*2),next(81*81*2),previous(81*81*2) {
        if(layout.walls[0] || layout.walls[1])
            throw std::invalid_argument("Race tablebase requires zero wall reserves");
        layout.sp=0;
        int moves[8];
        for(int p0=0;p0<81;p0++) for(int p1=0;p1<81;p1++) {
            if(p0==p1) continue;
            for(int turn=0;turn<2;turn++) {
                Position p{uint8_t(p0),uint8_t(p1),uint8_t(turn)};
                uint16_t from=indexOf(p); valid[from]=1;
                if(terminal(from)) continue;
                int n=generatePawnMoves(layout,p,moves);
                next[from].reserve(n);
                for(int i=0;i<n;i++) {
                    if(moves[i]>=81) throw std::logic_error("Zero-wall move generator returned a wall move");
                    Position q=p;
                    if(turn==0) q.pawn0=moves[i]; else q.pawn1=moves[i];
                    q.turn^=1;
                    uint16_t to=indexOf(q);
                    next[from].push_back(to); previous[to].push_back(from);
                }
            }
        }
    }

    bool terminal(uint16_t i) const {
        Position p=decode(i);
        return p.pawn0/9==0 || p.pawn1/9==8;
    }

    int terminalValue(uint16_t i) const {
        Position p=decode(i);
        int winner=p.pawn0/9==0?0:p.pawn1/9==8?1:-1;
        return winner<0?0:winner==p.turn?1:-1;
    }

    int noHistoryValue(Position p,int remainingPlies) const {
        uint16_t i=indexOf(p);
        if(i>=valid.size() || !valid[i]) throw std::invalid_argument("Invalid race position");
        if(terminal(i)) return terminalValue(i);
        if(remainingPlies<=0) return 0;
        ensureBase();
        int value=baseLabels->outcome[i];
        if((value==1 || value==-1) && baseLabels->dtm[i]>remainingPlies) return 0;
        return value;
    }

    void buildLabels(const std::vector<uint8_t>* forbidden,
                     std::vector<int8_t>& outcome,std::vector<uint16_t>& dtm) const {
        constexpr int8_t UNKNOWN=2, DRAW=0, WIN=1, LOSS=-1;
        const size_t count=valid.size();
        outcome.assign(count,UNKNOWN); dtm.assign(count,0);
        std::vector<uint16_t> unresolved(count,0), longest(count,0);
        std::vector<uint8_t> hasDraw(count,0);
        using Item=std::pair<uint16_t,uint16_t>; // DTM, state index
        std::priority_queue<Item,std::vector<Item>,std::greater<Item>> ready;
        for(uint32_t i=0;i<count;i++) {
            if(!valid[i]) continue;
            if(terminal(uint16_t(i))) {
                outcome[i]=int8_t(terminalValue(uint16_t(i))); dtm[i]=0; ready.emplace(0,uint16_t(i));
            } else if(forbidden && (*forbidden)[i]) {
                outcome[i]=DRAW;
            }
        }
        for(uint32_t i=0;i<count;i++) {
            if(!valid[i] || outcome[i]!=UNKNOWN) continue;
            for(uint16_t child:next[i]) {
                if(outcome[child]==DRAW) hasDraw[i]=1;
                else unresolved[i]++;
            }
            if(unresolved[i]==0 && !hasDraw[i]) {
                // A nonterminal state without moves is not expected on a valid board,
                // but preserving the draw rule is safer than manufacturing a loss.
                outcome[i]=DRAW;
            }
        }
        while(!ready.empty()) {
            auto [distance,child]=ready.top(); ready.pop();
            if(outcome[child]!=WIN && outcome[child]!=LOSS) continue;
            for(uint16_t parent:previous[child]) {
                if(outcome[parent]!=UNKNOWN) continue;
                if(outcome[child]==LOSS) {
                    outcome[parent]=WIN;
                    dtm[parent]=uint16_t(distance+1);
                    ready.emplace(dtm[parent],parent);
                } else {
                    if(unresolved[parent]) --unresolved[parent];
                    longest[parent]=std::max<uint16_t>(longest[parent],uint16_t(distance+1));
                    if(unresolved[parent]==0 && !hasDraw[parent]) {
                        outcome[parent]=LOSS; dtm[parent]=longest[parent];
                        ready.emplace(dtm[parent],parent);
                    }
                }
            }
        }
        for(size_t i=0;i<count;i++) if(valid[i] && outcome[i]==UNKNOWN) outcome[i]=DRAW;
    }

    bool ensureBase() const {
        std::lock_guard<std::mutex> lock(baseMutex);
        if(baseLabels) return false;
        auto labels=std::make_shared<Labels>();
        buildLabels(nullptr,labels->outcome,labels->dtm); baseLabels=std::move(labels); return true;
    }

    size_t memoryBytes() const {
        size_t bytes=sizeof(*this)+valid.capacity()*sizeof(uint8_t)+
            (next.capacity()+previous.capacity())*sizeof(Edges);
        for(const auto& edges:next) bytes+=edges.capacity()*sizeof(uint16_t);
        for(const auto& edges:previous) bytes+=edges.capacity()*sizeof(uint16_t);
        if(baseLabels) bytes+=baseLabels->outcome.capacity()*sizeof(int8_t)+baseLabels->dtm.capacity()*sizeof(uint16_t);
        for(const auto& entry:historyLabels)
            bytes+=entry.forbidden.capacity()*sizeof(uint16_t)+sizeof(Labels)+
                entry.labels->outcome.capacity()*sizeof(int8_t)+entry.labels->dtm.capacity()*sizeof(uint16_t);
        return bytes;
    }

    std::shared_ptr<const Labels> getHistoryLabels(const std::vector<uint8_t>& forbidden,bool& built) const {
        built=false;
        std::vector<uint16_t> key;
        for(uint32_t i=0;i<forbidden.size();i++) if(forbidden[i]) key.push_back(uint16_t(i));
        {
            std::lock_guard<std::mutex> lock(historyMutex);
            for(auto it=historyLabels.begin();it!=historyLabels.end();++it) if(it->forbidden==key) {
                auto labels=it->labels; historyLabels.splice(historyLabels.begin(),historyLabels,it); return labels;
            }
        }
        auto labels=std::make_shared<Labels>();
        buildLabels(&forbidden,labels->outcome,labels->dtm);
        std::lock_guard<std::mutex> lock(historyMutex);
        for(auto it=historyLabels.begin();it!=historyLabels.end();++it) if(it->forbidden==key) {
            auto existing=it->labels; historyLabels.splice(historyLabels.begin(),historyLabels,it); return existing;
        }
        historyLabels.push_front({std::move(key),labels});
        while(historyLabels.size()>8) historyLabels.pop_back();
        built=true;
        return labels;
    }
};

class GraphCache {
    using Geometry=std::array<uint64_t,2>;
    struct Entry { Geometry geometry; std::shared_ptr<Graph> graph; };
    std::mutex mutex;
    std::list<Entry> lru;
    size_t limit=128;
public:
    static GraphCache& instance() { static GraphCache cache; return cache; }
    std::shared_ptr<Graph> get(const Board& b) {
        Geometry key{b.hBits,b.vBits};
        {
            std::lock_guard<std::mutex> lock(mutex);
            for(auto it=lru.begin();it!=lru.end();++it) if(it->geometry==key) {
                auto graph=it->graph; lru.splice(lru.begin(),lru,it); return graph;
            }
        }
        auto graph=std::make_shared<Graph>(b);
        std::lock_guard<std::mutex> lock(mutex);
        for(auto it=lru.begin();it!=lru.end();++it) if(it->geometry==key) {
            auto existing=it->graph; lru.splice(lru.begin(),lru,it); return existing;
        }
        lru.push_front({key,graph});
        while(lru.size()>limit) lru.pop_back();
        return graph;
    }
};

class Tablebase {
    std::shared_ptr<Graph> graph;
public:
    explicit Tablebase(const Board& zeroWallLayout):graph(GraphCache::instance().get(zeroWallLayout)) {}
    size_t graphMemoryBytes() const { return graph->memoryBytes(); }

    Result solve(Position root,int remainingPlies,const std::vector<HistoryEntry>& history={}) const {
        constexpr int8_t DRAW=0, LOSS=-1, WIN=1;
        Result result;
        uint16_t rootIndex=indexOf(root);
        if(root.pawn0>=81 || root.pawn1>=81 || root.pawn0==root.pawn1 || root.turn>1)
            throw std::invalid_argument("Invalid race position");
        if(!graph->valid[rootIndex]) throw std::invalid_argument("Invalid race position");
        auto goalWinner=[&](Position p) {
            if(p.pawn0/9==0) return 0;
            if(p.pawn1/9==8) return 1;
            return -1;
        };
        int winner=goalWinner(root);
        if(winner!=-1) { result.outcome=winner==root.turn?WIN:LOSS; return result; }
        if(remainingPlies<=0) { result.outcome=DRAW; return result; }

        // A state seen twice in the prior game is a draw sink on any future visit.
        // The current root itself is not a third occurrence yet, so it is evaluated
        // by the one-ply wrapper below rather than by looking up its tablebase label.
        std::vector<uint8_t> forbidden(graph->valid.size(),0);
        std::vector<uint8_t> counts(graph->valid.size(),0);
        for(const auto& entry:history) {
            if(entry.count<1 || entry.count>3) throw std::invalid_argument("Invalid repetition count");
            if(entry.hBits!=graph->layout.hBits || entry.vBits!=graph->layout.vBits) continue;
            if(entry.position.pawn0>=81 || entry.position.pawn1>=81 ||
               entry.position.pawn0==entry.position.pawn1 || entry.position.turn>1)
                throw std::invalid_argument("Invalid history position");
            uint16_t i=indexOf(entry.position);
            if(i>=forbidden.size() || !graph->valid[i]) continue;
            counts[i]=uint8_t(std::min(3,int(counts[i])+entry.count));
        }
        uint16_t rootCount=counts[rootIndex];
        if(rootCount>=3) { result.outcome=DRAW; return result; }
        for(size_t i=0;i<counts.size();i++)
            if(counts[i]>=2 && !graph->terminal(uint16_t(i))) forbidden[i]=1;
        bool hasForbidden=false;
        for(uint8_t f:forbidden) if(f) { hasForbidden=true; break; }

        std::shared_ptr<const Graph::Labels> labels;
        if(hasForbidden) {
            bool built=false;
            labels=graph->getHistoryLabels(forbidden,built);
            result.usedHistoryTable=true;
            result.workStates=built?int(graph->valid.size()):0;
        } else {
            bool built=graph->ensureBase(); labels=graph->baseLabels;
            result.workStates=built?int(graph->valid.size()):0;
        }
        result.states=int(std::count(graph->valid.begin(),graph->valid.end(),uint8_t(1)));

        // In a decisive retrograde line DTM decreases at every ply: the winner picks
        // the shortest successor, while every move from a losing state has smaller DTM.
        // Thus the proof line cannot repeat a full state. Prior count-1 states cannot
        // become third occurrences on that line; count-2 states were removed above.
        int moves[8], n=generatePawnMoves(graph->layout,root,moves), best=-2, bestDtm=0;
        for(int j=0;j<n;j++) {
            Position child=root;
            if(root.turn==0) child.pawn0=moves[j]; else child.pawn1=moves[j];
            child.turn^=1;
            uint16_t ci=indexOf(child);
            int childValue;
            int childDistance=0;
            int w=goalWinner(child);
            if(w!=-1) childValue=w==child.turn?WIN:LOSS;
            else if(forbidden[ci]) childValue=DRAW;
            else {
                childValue=labels->outcome[ci]; childDistance=labels->dtm[ci];
                int left=remainingPlies-1;
                // DTM is the shortest forced win for W and longest resistance for L.
                // If it exceeds the remaining cutoff, W cannot finish in time and L
                // can select its delaying line, so the actual game result is a draw.
                if((childValue==WIN || childValue==LOSS) && childDistance>left) childValue=DRAW;
            }
            int value=-childValue;
            int distance=(childValue==DRAW)?0:childDistance+1;
            if(value>best || (value==best && value==WIN && distance<bestDtm) ||
               (value==best && value==LOSS && distance>bestDtm)) {
                best=value; bestDtm=distance;
            }
        }
        result.outcome=best==-2?DRAW:best;
        result.distance=result.outcome==DRAW?0:bestDtm;
        return result;
    }
};

} // namespace quoridor::race_tablebase
