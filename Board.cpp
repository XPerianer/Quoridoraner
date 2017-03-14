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
			return !hBarriers[y-1][x];
	}
	else if (direction == 2) { // go to bottom
			return !hBarriers[y][x];

	}
	else if (direction == 1) {
			return !vBarriers[y][x];

	}
	else if (direction == 3) {
			return !vBarriers[y][x-1];

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

    if(vert){ //If this happens it is impossible to place the Barrier, but it can still be the case that barriers are blocking each other
        if(!possible(x,y,1) || !possible(x,y+1,1))
            return false;
    } else{
        if(!possible(x,y,0) || !possible(x+1,y,0))
            return false;
    }

    for(auto b = barriers.begin();b != barriers.end();b++){ //Barriers are blocking each other "in a cross"
        if(b->vert != vert && x == b->x && y == b->y)
            return false;
    }

    //Barrier is theoretical possible to place, now check if both players can win if it is placed
    play p(false, -1, x, y, vert);
    makePlay(p);
    bool possible = true;
    if (floodFill(one) == -1 || floodFill(two) == -1) { //Play blocks one Player
        possible = false;
			//msg += "fF(one) " + to_string(floodFill(one)) + " fF(two) " + to_string(floodFill(two)) + "\n";
    }
    undoPlay(p);
    return possible;
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

	return moveDiff * 1 + barrierDiff * 2;
}

int Board::floodFill(Player p) {

	int target = 0;
	if (p.playerOne) {
		target = 8;
	}
	int steps=0;
    bitset<9> f[9] = {}; //Array is y-Coordinate, solo bits are x
    bitset<9> new_f[9] = {};
    f[p.y][p.x] = 1; //Now having a zeroed bitset with only a 1 at the player position
    bool equal=false;

    while(!equal){ //Flood everything
        equal = true;
        if(f[target] != 0){
            return steps;
        }
        for(int i=0;i<9;i++){
            if(f[i] != 0){
                //Right Movement:
                new_f[i] |= ((~vBarriers[i] & f[i])<<1);
                //Left Movement:
                new_f[i] |= (~vBarriers[i] & (f[i]>>1));
                //Top Movement:
                if(i!= 0){
                    new_f[i-1] |= (~hBarriers[i-1]) & f[i];
                }
                //Bot Movement:
                if(i!=8){
                    new_f[i+1] |= (~hBarriers[i] & f[i]);
                }
            }
        }
        steps++;


        for(int i=0;i<9;i++){
            if(f[i] != new_f[i]) {
                f[i] |= new_f[i];
                equal = false;
            }
        }
    }

    return -1; //Unable to reach the target

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
		placeBarrier(Barrier(p.vert,p.x,p.y));
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
        removeBarrier(Barrier(p.vert,p.x,p.y));
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
	
	for(int i=0;i<8;i++){
        if(hBarriers[i] != c.hBarriers[i])
            return false;
    }
    for(int i=0;i<9;i++){
        if(vBarriers[i] != c.vBarriers[i])
            return false;
    }

    //If the Bitsets are the same the barriers have to be the same, or at least they block the same things

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

					q.push(play(false, -1, ix, iy, false));

			}
			if (barrierPossible(ix, iy, true)) {

					q.push(play(false, -1, ix, iy, true));

			}
		}
	}

	return q;
}

void Board::placeBarrier(Barrier b) {

    barriers.push_back(b);
    if(b.vert){
        vBarriers[b.y][b.x] = 1;
        vBarriers[b.y+1][b.x] = 1;
    } else {
        hBarriers[b.y][b.x] = 1;
        hBarriers[b.y][b.x+1] = 1;

    }

}

void Board::removeBarrier(Barrier b) {

    if(b.vert){
        vBarriers[b.y][b.x] = 0;
        vBarriers[b.y+1][b.x] = 0;
    } else {
        hBarriers[b.y][b.x] = 0;
        hBarriers[b.y][b.x+1] = 0;

    }

    for(auto br = barriers.begin();br != barriers.end();br++){ //Remove Barrier from the list of barriers
        if(br->vert == b.vert && b.x == br->x && b.y == br->y){
            barriers.erase(br);
            return;
        }

    }


}

