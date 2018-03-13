//
// Created by Dominik Meier on 13.03.2018.
//

#ifndef QUORIDOR_BARRIER_H
#define QUORIDOR_BARRIER_H

struct Barrier{
	Barrier(int x, int y, bool vert) : vert(vert), x(x), y(y) {}
    Barrier(){}
	bool vert;
	int x, y;
};

#endif //QUORIDOR_BARRIER_H
