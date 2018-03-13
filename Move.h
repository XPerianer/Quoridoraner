#pragma once
#include <string>
#include "Barrier.h"
struct Move {
	bool playerMove; //true: player Movement, ignore vert; wrong: Barrier placement
	int direction;
	Barrier barrier;
	Move(bool p, int d, Barrier b) : playerMove(p), direction(d), barrier(b) {}
	Move() {}
};


