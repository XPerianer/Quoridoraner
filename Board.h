#pragma once
#include "Player.h"
#include "play.h"
#include <chrono>
#include <queue>

using namespace std;

class Board
{
private:
	bool playerMove = true; // true = Player 1 | false = Player 2
public:
	Board();
	~Board();

	Board(Board& b);
	bool checkBarrier(int horizontal, int x, int y);
	bool possible(short x, short y, short direction);
	bool barrierPossible(int x, int y, bool vert);
	bool barrierConnected(int x, int y, bool vert);
	bool checkWin(Player p);
	void switchPlayer();
	int floodFill(Player p);
	int evaluate();
	void makePlay(play p);
	void undoPlay(play p);

	queue<play> possibleMoves();

	string stringify();
	Player* activePlayer();
	Player* unactivePlayer();

	bool barriers[2][8][8] = { false }; // [0=horizontally;1=vertical][x][y]
	Player one, two;

	bool operator==(Board c);

};

