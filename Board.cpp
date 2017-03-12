#include "Board.h"



Board::Board()
{
	one.playerOne = true;
	two.playerOne = false;
	one.x = 4; one.y = 0;
	two.x = 4; two.y = 8;
	one.color = wxColor(196, 0, 50);
	two.color = wxColor(135, 196, 0);
}


Board::~Board()
{
}

Board::Board(Board& b) {
	one = b.one;
	two = b.two;
	playerMove = b.playerMove;

	for (int ix = 0; ix < 8; ix++) {
		for (int iy = 0; iy < 8; iy++) {
			barriers[0][ix][iy] = b.barriers[0][ix][iy];
			barriers[1][ix][iy] = b.barriers[1][ix][iy];
			
		}
	}
	
}

bool Board::checkBarrier(int horizontal, int x, int y) {
	if (x < 0 || x > 7 || y < 0 || y > 7) { //Wheres no game there can't be barriers
		return false;
	}
	else {
		return barriers[horizontal][x][y];
	}
}

bool Board::possible(short x, short y, short direction)
{
	if (direction == 2 && y == 8) { //bottom corner
		return false;
	}
	else if (direction == 0 && y == 0) {
		return false;
	}
	else if (direction == 1 && x == 8) {
		return false;
	}
	else if (direction == 3 && x == 0) {
		return false;
	}
	else if (direction == 0) {
		if (!checkBarrier(0, x - 1, y - 1) && !checkBarrier(0, x, y - 1)) {
			return true;
		}
	}
	else if (direction == 2) { // go to bottom
		if (!checkBarrier(0, x - 1, y) && !checkBarrier(0, x, y)) {
			return true;
		}
	}
	else if (direction == 1) {
		if (!checkBarrier(1, x, y - 1) && !checkBarrier(1, x, y)) {
			return true;
		}
	}
	else if (direction == 3) {
		if (!checkBarrier(1, x - 1, y - 1) && !checkBarrier(1, x - 1, y)) {
			return true;
		}
	}
	return false;
}

bool Board::barrierPossible(int x, int y, bool vert) {

	if (x < 0 || x > 7 || y < 0 || y > 7) {
		return false;
	}
	if (activePlayer()->barriers <= 0) {
		return false;
	}
	if (checkBarrier(0, x, y) || checkBarrier(1, x, y)) { //When barrier blocking each other
		return false;
	}
	if ((vert && !checkBarrier(1, x, y + 1) && !checkBarrier(1, x, y - 1)) || (!vert && !checkBarrier(0, x - 1, y) && !checkBarrier(0, x + 1, y))) {
		play p(false, -1, x, y, vert);
		makePlay(p);
		bool possible = true;
		if (floodFill(one) == -1 || floodFill(two) == -1) { //Play blocks one Player
			possible = false;
			//msg += "fF(one) " + to_string(floodFill(one)) + " fF(two) " + to_string(floodFill(two)) + "\n";
		}
		undoPlay(p);
		if (possible) {
			return true;
		}
	}
	return false;
}

bool Board::barrierConnected(int x, int y, bool vert) {

	if (!vert) {
		if (checkBarrier(true, x - 2, y) || checkBarrier(true, x + 2, y)) {
			return true;
		}
		else if (checkBarrier(false, x - 1, y) || checkBarrier(false, x + 1, y) || checkBarrier(false, x - 1, y - 1) || checkBarrier(false, x, y - 1) || checkBarrier(false, x + 1, y - 1)) {
			return true;
		}
		else if (checkBarrier(false, x - 1, y + 1) || checkBarrier(false, x, y + 1) || checkBarrier(false, x + 1, y + 1)) {
			return true;
		}
	}
	else {
		if (checkBarrier(true, x, y + 2) || checkBarrier(true, x, y - 2)) {
			return true;
		}
		else if (checkBarrier(false, x - 1, y - 1) || checkBarrier(false, x - 1, y) || checkBarrier(false, x - 1, y + 1)) {
			return true;
		}
		else if (checkBarrier(false, x + 1, y - 1) || checkBarrier(false, x + 1, y) || checkBarrier(false, x + 1, y + 1)) {
			return true;
		}
	}
	return false;

}

int Board::evaluate() { // positive is really good //evaluate light...

	if (checkWin(*activePlayer())) {
		return -1000;
	}
	if (checkWin(*unactivePlayer())) {
		return 1000;
	}
	
	int barrierDiff = -activePlayer()->barriers + unactivePlayer()->barriers;
	int moveDiff = floodFill(*activePlayer()) - floodFill(*unactivePlayer()); //-1 for the extra step the unactive Player has to go
	//msg += "Evaluated: " + to_string(moveDiff * 10 + barrierDiff * 1);

	return moveDiff * 10 + barrierDiff * 2;
}

int Board::floodFill(Player p) {

	int target = 0;
	if (p.playerOne) {
		target = 8;
	}
	int f[9][9] = {};
	struct node
	{
		int x, y;
	};
	queue<node> q;

	node start;
	start.x = p.x;
	start.y = p.y;
	q.push(start);

	f[p.x][p.y] = 1;

	while (q.front().y != target) {
		int xPos = q.front().x, yPos = q.front().y;

		if (yPos > 0 && f[xPos][yPos - 1] == 0 && possible(xPos, yPos, 0)) {
			f[xPos][yPos - 1] = f[xPos][yPos] + 1;
			node n;
			n.x = xPos; n.y = yPos - 1;
			q.push(n);
		}
		if (yPos < 8 && f[xPos][yPos + 1] == 0 && possible(xPos, yPos, 2)) {
			f[xPos][yPos + 1] = f[xPos][yPos] + 1;
			node n;
			n.x = xPos; n.y = yPos + 1;
			q.push(n);
		}
		if (xPos < 8 && f[xPos + 1][yPos] == 0 && possible(xPos, yPos, 1)) {
			f[xPos + 1][yPos] = f[xPos][yPos] + 1;
			node n;
			n.x = xPos + 1; n.y = yPos;
			q.push(n);
		}
		if (xPos > 0 && f[xPos - 1][yPos] == 0 && possible(xPos, yPos, 3)) {
			f[xPos - 1][yPos] = f[xPos][yPos] + 1;
			node n;
			n.x = xPos - 1; n.y = yPos;
			q.push(n);
		}
		q.pop();
		if (q.empty()) {
			return -1; // unreachable
		}
	}


	return f[q.front().x][q.front().y] - 1;

}

void Board::makePlay(play p) {
	if (p.playerMove == true) {
		if (p.direction == 0) {
			activePlayer()->y--;
		}
		else if (p.direction == 1) {
			activePlayer()->x++;
		}
		else if (p.direction == 2) {
			activePlayer()->y++;
		}
		else if (p.direction == 3) {
			activePlayer()->x--;
		}
	}
	else {
		activePlayer()->barriers--;
		if (p.vert) {
			barriers[1][p.x][p.y] = true;
		}
		else {
			barriers[0][p.x][p.y] = true;
		}
	}
	playerMove = !playerMove;
}

void Board::undoPlay(play p) {
	playerMove = !playerMove;
	if (p.playerMove == true) {
		if (p.direction == 0) {
			activePlayer()->y++;
		}
		else if (p.direction == 1) {

			activePlayer()->x--;
		}
		else if (p.direction == 2) {
			activePlayer()->y--;
		}
		else if (p.direction == 3) {
			activePlayer()->x++;
		}
	}
	else {
		activePlayer()->barriers++;
		if (p.vert) {
			barriers[1][p.x][p.y] = false;
		}
		else {
			barriers[0][p.x][p.y] = false;
		}
	}

}

Player *Board::activePlayer()
{
	if (playerMove) {
		return &one;
	}
	else {
		return &two;
	}
}

Player *Board::unactivePlayer() {
	if (!playerMove) {
		return &one;
	}
	else {
		return &two;
	}
}

bool Board::checkWin(Player p) {
	if (p.playerOne && p.y == 8) {
		return true;
	}
	else if (!p.playerOne && p.y == 0) {
		return true;
	}
	return false;
}

void Board::switchPlayer()
{
	playerMove = !playerMove;
}


bool Board::operator==(Board c) {
	if (playerMove != c.playerMove)
		return false;
	if (one.x != c.one.x || one.y != c.one.y)
		return false;
	if (two.x != c.two.x || two.y != c.two.y)
		return false;
	if (one.barriers != c.one.barriers || two.barriers != c.two.barriers)
		return false;
	
	for (int ix = 0; ix < 8; ix++) {
		for (int iy = 0; iy < 8; iy++) {
			if (barriers[0][ix][iy] != c.barriers[0][ix][iy])
				return false;
			if (barriers[1][ix][iy] != c.barriers[1][ix][iy])
				return false;
		}
	}

	return true;
}


queue<play> Board::possibleMoves()
{
	queue <play> q;
	queue <play> pq; //Priority queue, this moves are more likely to be good
	int x = activePlayer()->x; int y = activePlayer()->y;
	if (possible(x, y, 0)) {
		pq.push(play(true, 0, x, y - 1, false));
	}
	if (possible(x, y, 1)) {
		pq.push(play(true, 1, x + 1, y, false));
	}
	if (possible(x, y, 2)) {
		pq.push(play(true, 2, x, y + 1, false));
	}
	if (possible(x, y, 3)) {
		pq.push(play(true, 3, x - 1, y, false));
	}

	for (int ix = 0; ix < 8; ix++) {
		for (int iy = 0; iy < 8; iy++) {
			if (barrierPossible(ix, iy, false)) {
				if (barrierConnected(ix, iy, false)) {
					pq.push(play(false, -1, ix, iy, false));
				}
				else {
					q.push(play(false, -1, ix, iy, false));
				}
			}
			if (barrierPossible(ix, iy, true)) {
				if (barrierConnected(ix, iy, true)) {
					pq.push(play(false, -1, ix, iy, true));
				}
				else {
					q.push(play(false, -1, ix, iy, true));
				}
			}
		}
	}

	while (!q.empty()) { //All moves are stored in the priority queue
		pq.push(q.front());
		q.pop();
	}
	return pq;
}