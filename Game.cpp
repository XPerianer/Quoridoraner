#include "Game.h"
Game::Game()
{
}

Game::~Game()
{
}

void Game::tryMove(short direction) {
	
	if (gameBoard.possible(gameBoard.activePlayer()->x, gameBoard.activePlayer()->y, direction)) {
		
			if (direction == 0) {
				gameBoard.activePlayer()->y--;
			}
			else if (direction == 1) {
				gameBoard.activePlayer()->x++;
			}
			else if (direction == 2) {
				gameBoard.activePlayer()->y++;
			}
			else if (direction == 3) {
				gameBoard.activePlayer()->x--;
			}
			gameBoard.switchPlayer();
			//computerMove();
			msg += to_string(gameBoard.evaluate()) + " FloodFill: " + to_string(gameBoard.floodFill(*gameBoard.activePlayer()));
		}

	

}


void Game::tryBarrier(int x, int y, bool vert)
{
	if (gameBoard.barrierPossible(x, y, vert)) {
		gameBoard.placeBarrier(Barrier(vert, x, y));
		gameBoard.activePlayer()->barriers--;
		gameBoard.switchPlayer();
	}
	else {
		return;
	}
	//computerMove();
	msg += to_string(gameBoard.evaluate()) + " FloodFill: " + to_string(gameBoard.floodFill(*gameBoard.activePlayer()));
}

void Game::drawGame(wxPaintDC * dc) {

	for (int i = 0; i < 9; i++) { // Draw Grid
		for (int j = 0; j < 9; j++) {
			dc->DrawRoundedRectangle(wxRect(20 + i*(50 + 20), 20 + j*(50 + 20), 50, 50), 5);
		}
	}
	//Draw Players
	dc->SetBrush(wxBrush(gameBoard.one.color));
	dc->DrawCircle(20 + gameBoard.one.x * 70 + 25, 20 + gameBoard.one.y * 70 + 25, 20);
	dc->SetBrush(wxBrush(gameBoard.two.color));
	dc->DrawCircle(20 + gameBoard.two.x * 70 + 25, 20 + gameBoard.two.y * 70 + 25, 20);

	//Draw Active Player
	dc->SetBrush(wxBrush(gameBoard.activePlayer()->color));
	dc->DrawCircle(820, 160, 20);


	//Draw Barriers
	dc->SetBrush(wxBrush(wxColor("grey")));



	for(auto br = gameBoard.barriers.begin();br != gameBoard.barriers.end();br++){ //Remove Barrier from the list of barriers
			if(br->vert) {
				dc->DrawRectangle(70 + br->x * 70 + 3, 20 + br->y * 70, 14, 120);
			} else {
				dc->DrawRectangle(20 + br->x * 70, 70 + br->y * 70 + 3, 120, 14);

		}

	}


	dc->SetBrush(wxBrush(wxColor("black")));
	//Show Barriers Left
	dc->DrawText(wxT("Barriers(Red): " + to_string(gameBoard.one.barriers)), 1000, 20);
	dc->DrawText(wxT("Barriers(Green): " + to_string(gameBoard.two.barriers)), 1000, 40);

}


int Game::negamax(Board *b, int depth, int alpha, int beta)
{
	if (b->checkWin(*b->activePlayer()) || b->checkWin(*b->unactivePlayer())) {
		//msg += to_string(b->evaluate());
		return b->evaluate();
	}
	if (depth == 0) {
		evaluated++;
		return b->evaluate();
	}

	queue<play> moves = b->possibleMoves();
	int score = -10000;
	while (!moves.empty()) {
		b->makePlay(moves.front()); //Try next Move
		score = -negamax(b,depth - 1, -beta, -alpha);
		b->undoPlay(moves.front());
		
		//msg += to_string(score) + "|";

		if (score > alpha) {
			alpha = score; // alpha acts like max in MiniMax
		}
		if (alpha >= beta) {
			pruned++;
			break;  //   beta-cutoff
		}
		moves.pop();
	}

	
	return alpha;
}


void Game::computerMove() {
	auto start = chrono::high_resolution_clock::now();
	msg = "Information:\n";
	play bestPlay;
	
	int alpha = -1000, beta = 1000;

	evaluated = 0; pruned = 0; transpositions = 0;
	searchDepth = 0;
	queue<play> q = gameBoard.possibleMoves();

	vector<future<int>> score;
	vector<play> plays;
	while (!q.empty()) {
		Board g = gameBoard;
		g.makePlay(q.front());
		plays.push_back(q.front());
		score.push_back(async(launch::async,&Game::negamax, this , new Board(g) , 2, alpha, beta)); //Parralel Execution
		q.pop();
	}

	int help = -1;
	int bestScore = -100000;
	for (unsigned int i = 0; i < score.size(); i++) {
		int s = score.at(i).get();
		if(s >= help)
			msg += "S: " + to_string(s) + "P: " + to_string(plays.at(i).direction) + "D " + to_string(plays.at(i).x) + "x " + to_string(plays.at(i).y) + "y \n";
			
		help = s;
		
		if (s >= bestScore) {
				bestPlay = plays.at(i);
				bestScore = s;
			}
	}
	
	
	
	
	msg += "Best Score: " + to_string(bestScore) + "\n";
	msg += "Best Move: " + to_string(bestPlay.direction) + "D " + to_string(bestPlay.x) + "x " + to_string(bestPlay.y) + "y \n";
	msg += "Evaluated|Pruned|Transpositions: " + to_string(evaluated) + " | " + to_string(pruned) + +" | " + to_string(transpositions) + "\n";

	gameBoard.makePlay(bestPlay);

	auto end = chrono::high_resolution_clock::now();
	auto diff = end - start;
	msg += "Total Computation Time: " + to_string(diff.count()) + "\n";

}

