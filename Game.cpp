#include "Game.h"
Game::Game()
{
}

Game::~Game()
{
}


void Game::tryMove(Move m) {//TODO: Save move in history
    if(m.playerMove) {
		tryMoving(m.direction);
	} else{
		tryBarrier(m.barrier);
	}
}

void Game::tryMoving(short direction) {
	
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


void Game::tryBarrier(Barrier b)
{
	if (gameBoard.barrierPossible(b)) {
		gameBoard.placeBarrier(b);
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
	for(auto br : gameBoard.barriers){ //Remove Barrier from the list of barriers
			if(br.vert) {
				dc->DrawRectangle(70 + br.x * 70 + 3, 20 + br.y * 70, 14, 120);
			} else {
				dc->DrawRectangle(20 + br.x * 70, 70 + br.y * 70 + 3, 120, 14);

		}

	}
	dc->SetBrush(wxBrush(wxColor("black")));
	//Show Barriers Left
	dc->DrawText(wxT("Barriers(Red): " + to_string(gameBoard.one.barriers)), 1000, 20);
	dc->DrawText(wxT("Barriers(Green): " + to_string(gameBoard.two.barriers)), 1000, 40);

}


int Game::negamax(Board *b, int depth, int alpha, int beta)
{
	if (depth == 0 || b->checkWin(*b->activePlayer()) || b->checkWin(*b->unactivePlayer())) {
		//msg += to_string(b->evaluate());
		return b->evaluate();
	}

	queue<Move> moves = b->likelyMoves();
	while (!moves.empty()) {
		b->makePlay(moves.front()); //Try next Move
		int score = -negamax(b,depth - 1, -beta, -alpha);
		b->undoPlay(moves.front());
        moves.pop();
		//msg += to_string(score) + "|";
		if (score > alpha) {
			alpha = score; // alpha acts like max in MiniMax
		}
		if (score >= beta) {
			pruned++;
			return beta; //beta-cutoff
		}
	}
	return alpha;
}


void Game::computerMove() {
	auto start = chrono::high_resolution_clock::now();
	msg = "Information:\n";
	Move bestPlay;
	
	int alpha = -1001, beta = 1001;

	evaluated = 0; pruned = 0;
	searchDepth = 0;
    queue<Move> q = gameBoard.likelyMoves();
    int bestScore=-1001;
    while(!q.empty()){
        gameBoard.makePlay(q.front());
        int score = -negamax(&this->gameBoard,4,alpha,beta);
        if(score > bestScore) {
            bestScore = score;
            bestPlay = q.front();
        }
        gameBoard.undoPlay(q.front());
        q.pop();
    }

	msg += "Best Score: " + to_string(bestScore) + "\n";
	msg += "Evaluated|Pruned: " + to_string(evaluated) + " | " + to_string(pruned) + "\n";

	gameBoard.makePlay(bestPlay);

	auto end = chrono::high_resolution_clock::now();
	auto diff = end - start;
	msg += "Total Computation Time: " + to_string(diff.count()/1000000) + "\n";

}


