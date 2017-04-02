### Quoridoraner
Artifical Intelligence and GUI for the classic board game "Quoridor". Works with the help of NegaMax and Alpha-Beta-Pruning to calculate the best moves.


##Compilation
You need a working [wxWidgets](www.wxwidgets.org)-Installation and [CMake](cmake.org) for building.
Then just generate the Makefile/Project with CMake and compile.

## Controls
# Movement
Use the Top, Right, Left, Bottom Buttons on the GUI
# Barrier Placement
Click the vertical or horizontal shaped buttons. Then click the square on the field that is at the left button corner of your planned barrier.
Example:
You want to place your barrier there, click the field with the X. (MMMMM - Barrier, X - Where to Click)
______________
|  |  |  |  |
____MMMMM_____
|  | X|  |  |
______________

