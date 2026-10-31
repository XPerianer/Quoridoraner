#include "race_tablebase.hpp"
#include <cassert>
#include <iostream>
#include <random>
#include <unordered_map>

using namespace quoridor;
using namespace quoridor::race_tablebase;

static uint16_t ix(int p0,int p1,int turn) { return indexOf(Position{uint8_t(p0),uint8_t(p1),uint8_t(turn)}); }

static int brute(const Board& layout,Position p,int left,std::unordered_map<uint16_t,int>& seen) {
    Board b=layout; b.pawn[0]=p.pawn0; b.pawn[1]=p.pawn1; b.turn=p.turn; b.sp=0;
    int winner=b.winner();
    if(winner!=-1) return winner==p.turn?1:-1;
    uint16_t key=indexOf(p);
    if(seen[key]>=3 || left<=0) return 0;
    int moves[136],n=b.generate(moves),best=-1;
    for(int i=0;i<n;i++) {
        Position child=p;
        if(p.turn==0) child.pawn0=moves[i]; else child.pawn1=moves[i];
        child.turn^=1;
        uint16_t k=indexOf(child); ++seen[k];
        best=std::max(best,-brute(layout,child,left-1,seen));
        if(--seen[k]==0) seen.erase(k);
        if(best==1) break;
    }
    return best;
}

static std::vector<HistoryEntry> historyOf(const Board& layout,const std::unordered_map<uint16_t,int>& history) {
    std::vector<HistoryEntry> out;
    for(auto [i,count]:history) out.push_back({decode(i),uint8_t(count),layout.hBits,layout.vBits});
    return out;
}

static Board noWalls() {
    Board b; b.walls[0]=b.walls[1]=0; b.initHash(); return b;
}

static void compareMoveGenerator(const Board& layout) {
    for(int p0=0;p0<81;p0++) for(int p1=0;p1<81;p1++) if(p0!=p1) for(int turn=0;turn<2;turn++) {
        Position p{uint8_t(p0),uint8_t(p1),uint8_t(turn)};
        int direct[8],referenceMoves[136];
        int n=generatePawnMoves(layout,p,direct);
        Board b=layout; b.pawn[0]=p0; b.pawn[1]=p1; b.turn=turn; b.sp=0;
        int m=b.generate(referenceMoves);
        std::vector<int> a(direct,direct+n),r(referenceMoves,referenceMoves+m);
        std::sort(a.begin(),a.end()); std::sort(r.begin(),r.end());
        if(a!=r) {
            std::cerr<<"move mismatch p0="<<p0<<" p1="<<p1<<" turn="<<turn<<" direct:";
            for(int v:a) std::cerr<<' '<<v;
            std::cerr<<" reference:"; for(int v:r) std::cerr<<' '<<v; std::cerr<<"\n";
            std::cerr<<"openRight="<<uint64_t(layout.openRight)<<" stepEast="<<canPawnStep(layout,0,1)<<"\n";
            std::abort();
        }
    }
}

static void compareAllFiniteHorizons(const Board& layout) {
    Graph graph(layout);
    const size_t count=graph.valid.size();
    std::vector<int8_t> previous(count,0),next(count,0);
    for(uint32_t i=0;i<count;i++) if(graph.valid[i] && graph.terminal(uint16_t(i)))
        previous[i]=int8_t(graph.terminalValue(uint16_t(i)));
    for(int horizon=0;horizon<=100;horizon++) {
        for(uint32_t i=0;i<count;i++) if(graph.valid[i]) {
            int expected=previous[i];
            Position p=decode(uint16_t(i));
            int actual=graph.noHistoryValue(p,horizon);
            if(actual!=expected) {
                std::cerr<<"Finite-horizon mismatch state="<<i<<" horizon="<<horizon
                         <<" tablebase="<<actual<<" oracle="<<expected<<"\n";
                std::abort();
            }
        }
        if(horizon==100) break;
        for(uint32_t i=0;i<count;i++) if(graph.valid[i]) {
            if(graph.terminal(uint16_t(i))) next[i]=int8_t(graph.terminalValue(uint16_t(i)));
            else {
                int best=-1;
                for(uint16_t child:graph.next[i]) best=std::max(best,-int(previous[child]));
                next[i]=int8_t(best);
            }
        }
        previous.swap(next);
    }
}

static void compare(const Board& layout,Position p,int rem,std::unordered_map<uint16_t,int> history) {
    Tablebase table(layout);
    auto actual=table.solve(p,rem,historyOf(layout,history));
    int expected=brute(layout,p,rem,history);
    if(actual.outcome!=expected) {
        std::cerr<<"Mismatch p0="<<int(p.pawn0)<<" p1="<<int(p.pawn1)<<" turn="<<int(p.turn)
                 <<" rem="<<rem<<" tb="<<actual.outcome<<" brute="<<expected<<"\n";
        std::abort();
    }
}

int main() {
    Board empty=noWalls();
    compareMoveGenerator(empty);
    compareAllFiniteHorizons(empty);
    // Exhaustive finite-horizon comparison over a broad boundary slice, all turns,
    // and cutoffs 0..4. The independent reference uses recursive minimax and history counts.
    for(int p0=0;p0<27;p0++) for(int p1=54;p1<81;p1++) if(p0!=p1)
        for(int turn=0;turn<2;turn++) for(int rem=0;rem<=4;rem++) {
            Position p{uint8_t(p0),uint8_t(p1),uint8_t(turn)};
            compare(empty,p,rem,{{ix(p0,p1,turn),1}});
        }

    // One-step child already present twice must be a draw on its next visit.
    Position root{40,4,0};
    Board rb=empty; rb.pawn[0]=root.pawn0; rb.pawn[1]=root.pawn1; rb.turn=root.turn;
    int moves[136],n=rb.generate(moves); assert(n>0);
    Position child=root; child.pawn0=moves[0]; child.turn=1;
    compare(empty,root,6,{{ix(40,4,0),1},{indexOf(child),2}});
    compare(empty,root,6,{{ix(40,4,0),2}}); // root itself is handled by the one-ply wrapper

    // Real walls with no reserves: straight jump blocked and only the west diagonal is open.
    Board blocked=noWalls();
    blocked.setWall(false,20,true); blocked.setWall(true,28,true); blocked.initHash();
    compareMoveGenerator(blocked);
    compareAllFiniteHorizons(blocked);
    Position jump{40,31,0};
    blocked.pawn[0]=40; blocked.pawn[1]=31; blocked.turn=0;
    n=blocked.generate(moves);
    bool west=false,straight=false,east=false;
    for(int i=0;i<n;i++) { west|=moves[i]==30; straight|=moves[i]==22; east|=moves[i]==32; }
    assert(west && !straight && !east);
    for(int rem=0;rem<=5;rem++) compare(blocked,jump,rem,{{indexOf(jump),1}});
    compare(blocked,jump,5,{{indexOf(jump),1},{indexOf(Position{30,31,1}),2}});

    // Deeper cutoffs and count-1/count-2 historical states from a deterministic
    // random sample exercise finite DTM conversion and history-filtered retrograde.
    std::mt19937 rng(20261030);
    for(int sample=0;sample<48;sample++) {
        Position p{uint8_t(rng()%81),uint8_t(rng()%81),uint8_t(rng()%2)};
        while(p.pawn0==p.pawn1) p.pawn1=uint8_t(rng()%81);
        int rem=5+int(rng()%4);
        std::unordered_map<uint16_t,int> history{{indexOf(p),sample%3==0?2:1}};
        for(int h=0;h<2;h++) {
            Position old{uint8_t(rng()%81),uint8_t(rng()%81),uint8_t(rng()%2)};
            while(old.pawn0==old.pawn1) old.pawn1=uint8_t(rng()%81);
            history[indexOf(old)]=2;
        }
        compare(empty,p,rem,history);
    }

    // Goal takes precedence over cutoff.
    Position goal{0,40,1};
    compare(empty,goal,0,{{indexOf(goal),1}});
    compare(empty,goal,0,{{indexOf(goal),3}});
    std::cout<<"race tablebase exhaustive comparisons passed\n";
}
