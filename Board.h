#pragma once
#include "Player.h"
#include "Move.h"
#include "Barrier.h"
#include "Move.h"
#include <chrono>
#include <queue>
#include <bitset>

using namespace std;

class Board
{
private:
	bool playerMove = true; // true = Player 1 | false = Player 2
public:
	Board();
	~Board();

	Board(Board& b);
	bool possible(short x, short y, short direction);
	bool barrierPossible(Barrier b);

	bool checkWin(Player p);
	void switchPlayer();
	int floodFill(Player p);
	int evaluate();
	void makePlay(Move p);
	void undoPlay(Move p);

	void placeBarrier(Barrier b);
	void removeBarrier(Barrier b);

	queue<Move> possibleMoves();
    queue<Move> likelyMoves();//Not all moves, but the ones a human player will Move almost always


	Player* activePlayer();
	Player* unactivePlayer();

	bitset<9> vBarriers[9] = {}; // 9*9 0 == you can move from there ; 1 == theres a barrier, can't pass //9th place unused for real data but helpful for ORing
	bitset<9> hBarriers[8] = {}; // 9*8 Numbers are counted from top left to bot right

	vector<Barrier> barriers;

	Player one, two;

	bool operator==(Board c);

};

