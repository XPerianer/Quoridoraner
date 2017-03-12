#pragma once
#include "Player.h"
#include "Board.h"
#include "play.h"
#include <wx/wx.h>
#include <thread>
#include <future>
#include <queue>
#include <chrono>
#include <mutex>

using namespace std;



class Game
{
private:
	Board gameBoard;
	int searchDepth = 0;
	long evaluated;
	long pruned;
	long transpositions;
public:
	int length;
	Game();
	~Game();

	string msg="Welcome at the Quoridoraner";
	void drawGame(wxPaintDC* dc);
	void tryMove(short direction);
	void tryBarrier(int x, int y, bool vert);
	int negamax(Board* b, int depth, int alpha, int beta);
	void computerMove();


};


