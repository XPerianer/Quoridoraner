#include "race_tablebase.hpp"
#include <chrono>
#include <iostream>
#include <cstdlib>

using namespace quoridor;
using namespace quoridor::race_tablebase;

int main(int argc,char** argv) {
    int perLayout=argc>1?std::max(1,std::atoi(argv[1])):128;
    int layouts=argc>2?std::max(1,std::atoi(argv[2])):32;
    auto t0=std::chrono::steady_clock::now();
    int resolved=0, historyResolved=0, repeatedHistoryResolved=0, checksum=0;
    size_t graphBytes=0;
    auto coldStart=std::chrono::steady_clock::now();
    double noHistorySeconds=0, historySeconds=0, repeatedHistorySeconds=0;
    for(int l=0;l<layouts;l++) {
        Board layout; layout.walls[0]=layout.walls[1]=0;
        for(int j=0;j<1+(l%3);j++) {
            bool vertical=((l+j)&1)!=0;
            int anchor=(l*17+j*13)%64;
            if(layout.fits(vertical,anchor%8,anchor/8)) layout.setWall(vertical,anchor,true);
        }
        layout.initHash();
        Tablebase table(layout);
        graphBytes=std::max(graphBytes,table.graphMemoryBytes());
        std::vector<HistoryEntry> none;
        auto noHistoryStart=std::chrono::steady_clock::now();
        for(int i=0;i<perLayout;i++) {
            Position p{uint8_t(27+(i%27)),uint8_t(54+((i/27)%27)),uint8_t(i&1)};
            Result r=table.solve(p,100,none);
            checksum+=r.outcome+2*r.distance; resolved+=r.outcome!=2;
        }
        noHistorySeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-noHistoryStart).count();
        auto historyStart=std::chrono::steady_clock::now();
        for(int i=0;i<std::min(perLayout,32);i++) {
            Position p{uint8_t(27+(i%27)),uint8_t(54+((i/27)%27)),uint8_t(i&1)};
            Position old{uint8_t((p.pawn0+7)%81),uint8_t((p.pawn1+11)%81),uint8_t((p.turn+1)&1)};
            if(old.pawn0==old.pawn1) old.pawn1=(old.pawn1+1)%81;
            std::vector<HistoryEntry> seen{{p,1,layout.hBits,layout.vBits},{old,2,layout.hBits,layout.vBits}};
            Result r=table.solve(p,100,seen);
            checksum+=r.outcome+2*r.distance; historyResolved+=r.outcome!=2;
        }
        historySeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-historyStart).count();
        auto repeatedStart=std::chrono::steady_clock::now();
        Position old{3,70,1};
        std::vector<HistoryEntry> sameSeen{{old,2,layout.hBits,layout.vBits}};
        for(int i=0;i<perLayout;i++) {
            Position p{uint8_t(27+(i%27)),uint8_t(54+((i/27)%27)),uint8_t(i&1)};
            sameSeen.push_back({p,1,layout.hBits,layout.vBits});
            Result r=table.solve(p,100,sameSeen);
            sameSeen.pop_back();
            checksum+=r.outcome+2*r.distance; repeatedHistoryResolved+=r.outcome!=2;
        }
        repeatedHistorySeconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-repeatedStart).count();
        graphBytes=std::max(graphBytes,table.graphMemoryBytes());
    }
    auto coldSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-coldStart).count();
    auto totalSeconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    int queries=layouts*perLayout;
    int totalQueries=resolved+historyResolved+repeatedHistoryResolved;
    std::cout<<"layouts="<<layouts<<" queries_per_layout="<<perLayout<<" queries="<<queries
             <<" answered="<<resolved<<" build_and_solve_seconds="<<coldSeconds
             <<" no_history_qps="<<resolved/noHistorySeconds<<" estimated_graph_MiB_each="
             <<double(graphBytes)/(1024*1024)<<" retained_MiB_at_cache_limit="
             <<double(graphBytes*std::min(layouts,128))/(1024*1024)
             <<" distinct_history_queries="<<historyResolved<<" history_seconds="<<historySeconds
             <<" history_queries_per_second="<<historyResolved/historySeconds<<" total_seconds="<<totalSeconds
             <<" cached_history_queries="<<repeatedHistoryResolved<<" cached_history_seconds="<<repeatedHistorySeconds
             <<" cached_history_queries_per_second="<<repeatedHistoryResolved/repeatedHistorySeconds
             <<" mixed_qps="<<totalQueries/totalSeconds
             <<" checksum="<<checksum<<"\n";
}
