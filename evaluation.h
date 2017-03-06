#pragma once

#include "Board.h"
struct evaluation {
	int depth = -1;
	bool exact = false;
	bool upperBound = false, lowerBound = false;
	int score = -1;
	Board b;
};