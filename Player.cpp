#include "Player.h"



Player::Player()
{
	x = 4;
	y = 0;
	barriers = 10;
}

Player::Player(int px, int py)
{
	x = px;
	y = py;
	barriers = 10;
}

Player::Player(Player& p) {
	x = p.x;
	y = p.y;
	playerOne = p.playerOne;
	barriers = p.barriers;
}


Player::~Player()
{
}
