#include "engine.hpp"
#include <cassert>
#include <iostream>

int main() {
    uint32_t seed = 173;
    quoridor::Searcher search;
    for (int game = 0; game < 100; game++) {
        quoridor::Board board;
        board.initHash();
        const auto initial1 = board.hash1, initial2 = board.hash2;
        std::vector<int> played;
        for (int ply = 0; ply < 64 && board.winner() < 0; ply++) {
            int moves[136]; int n = board.generate(moves);
            assert(n > 0 && n <= 136);
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            int move = moves[seed % n];
            board.make(move); played.push_back(move);
            assert(board.hash1 == board.hash(quoridor::Z1));
            assert(board.hash2 == board.hash(quoridor::Z2));
            assert(board.mirrorHash1 == board.hash(quoridor::Z1,true));
            assert(board.mirrorHash2 == board.hash(quoridor::Z2,true));
            for (int i = 0; i < 64; i++) {
                assert(bool(board.hBits & (uint64_t(1) << i)) == bool(board.h[i]));
                assert(bool(board.vBits & (uint64_t(1) << i)) == bool(board.v[i]));
            }
            assert(board.distance(0) >= 0 && board.distance(1) >= 0);
            if (ply % 16 == 0 && board.winner() < 0) {
                auto result = search.search(board,2000);
                int legal[136]; int count = board.generate(legal);
                assert(std::find(legal,legal+count,result.move) != legal+count);
            }
        }
        for (auto it = played.rbegin(); it != played.rend(); ++it) board.unmake(*it);
        assert(board.hash1 == initial1 && board.hash2 == initial2);
        assert(board.mirrorHash1 == board.hash(quoridor::Z1,true));
        assert(board.mirrorHash2 == board.hash(quoridor::Z2,true));
        assert(board.hBits == 0 && board.vBits == 0);
        assert(board.pawn[0] == 76 && board.pawn[1] == 4);
    }
    std::cout << "100 games: make/unmake, hashes, reachability and search passed\n";
}
