#pragma once
#include <wx/wx.h>
class Player
{
public:
	int x, y;
	int barriers;
	bool playerOne = true;
	wxColor color;

public:
	Player();
	Player(int px, int py);
	~Player();
	Player(Player& p);
};

