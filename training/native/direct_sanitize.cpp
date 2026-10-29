// Standalone memory-safety stress test for threaded MCTS, rules and snapshots.
#include "direct.cpp"
#include <iostream>
#include <random>

int main() {
    auto require=[](int result) { if(result<0) throw std::runtime_error(q_error()); return result; };
    std::unique_ptr<Pool> pool(static_cast<Pool*>(q_create(4)));
    if(!pool) throw std::runtime_error(q_error());
    constexpr int games=16;
    std::array<Key,games> positions;
    std::array<std::vector<Key>,games> histories;
    std::array<int,games> plies{};
    Board initial; initial.initHash();
    for(int i=0;i<games;i++) { positions[i]=packed(initial); histories[i].push_back(positions[i]); }
    std::vector<float> featuresBuffer(games*729), policies(games*136), values(games), moves(games*209);
    std::vector<int32_t> legal(games*136), counts(games), indices(games);
    std::mt19937 rng(29);
    int64_t reused=0;
    for(int turn=0;turn<150;turn++) {
        require(q_start(pool.get()));
        for(int i=0;i<games;i++) {
            // Keep full repetition history, combining duplicate positions.
            std::map<Key,int> seen; for(const auto& k:histories[i]) seen[k]++;
            if(unpack(positions[i].data()).winner()!=-1 || plies[i]>=100 || seen[positions[i]]>=3) {
                require(q_drop(pool.get(),i)); positions[i]=packed(initial); histories[i]={positions[i]}; plies[i]=0;
                seen.clear(); seen[positions[i]]=1;
            }
            std::vector<Key> keys; std::vector<int32_t> amounts;
            for(const auto& pair:seen) { keys.push_back(pair.first); amounts.push_back(pair.second); }
            double unused=0;
            require(q_begin(pool.get(),i,positions[i].data(),keys[0].data(),amounts.data(),keys.size(),plies[i],100,32,&unused,0));
        }
        while(true) {
            int n=require(q_wave(pool.get(),featuresBuffer.data(),legal.data(),counts.data(),indices.data()));
            if(!n) break;
            for(int i=0;i<n;i++) {
                if(counts[i]<1 || counts[i]>136) throw std::runtime_error("Invalid move count");
                for(int k=0;k<counts[i];k++) policies[i*136+k]=1+float(rng()%100);
                values[i]=float(int(rng()%201)-100)/100;
            }
            require(q_evaluate(pool.get(),policies.data(),values.data(),indices.data(),n));
        }
        require(q_policy(pool.get(),moves.data(),&reused));
        for(int i=0;i<games;i++) {
            int move=std::max_element(moves.begin()+i*209,moves.begin()+(i+1)*209)-(moves.begin()+i*209);
            Key next; require(q_advance(pool.get(),i,positions[i].data(),move,next.data()));
            positions[i]=next; histories[i].push_back(next); plies[i]++;
        }
    }
    require(q_clear(pool.get()));
    std::cout << "ASAN/UBSAN: 2,400 game turns, 76,800 new MCTS simulations passed\n";
}
