#ifndef SUDOKU_CUH
#define SUDOKU_CUH


#define N 9 // board size
#define n 3 // sub-board size

void cudaSudokuBacktrack(const unsigned int blocks,
	const unsigned int threadsPerBlock,
	int *boards,
	const int numBoards,
	int *emptySpaces,
	int *numEmptySpaces,
	int *finished,
	int *solved);

void callBFSKernel(const unsigned int blocks,
	const unsigned int threadsPerBlock,
	int *old_boards,
	int *new_boards,
	int total_boards,
	int *board_index,
	int *empty_spaces,
	int *empty_space_count);
#endif // SUDOKU_CUH