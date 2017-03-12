#pragma once
#include <string>

struct play {
	bool playerMove; //true: player Movement, ignore vert; wrong: Barrier placement
	int direction;
	int x, y;
	bool vert;
	play(bool p, int d, int x, int y, bool v) : playerMove(p), direction(d), x(x), y(y), vert(v) {}
	play() {};
};


