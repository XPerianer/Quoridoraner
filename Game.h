#pragma once

#include "Player.h"
#include "Board.h"
#include "play.h"
#include "evaluation.h"
#include <wx/wx.h>
#include <unordered_map>
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
	long storedTranspositions = 0;
	mutex transpositionTableLock;
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
	

	unordered_map<string, evaluation> transpositionTable;

};


