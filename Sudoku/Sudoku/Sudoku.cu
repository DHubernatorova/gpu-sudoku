#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

#include "Sudoku.cuh"


/**
* Takes array and resets all values to false.
*/
__device__
void clearArray(bool *arr, int size) {
	for (int i = 0; i < size; i++) {
		arr[i] = false;
	}
}


/**
* Checks if the state of board is valid.
* board is one-dimensional array which stores the sudoku board.
*/
__device__
bool isBoardValid(const int *board) {
	bool visited[N]; // indicates already visited values in row or column or sub-board
	clearArray(visited, N);

	// rows
	for (int row = 0; row < N; row++) {
		clearArray(visited, N);
		for (int col = 0; row < N; col++) {
			int value = board[row * N + col];

			if (value != 0) {
				if (visited[value - 1]) {
					return false;
				}
				else {
					visited[value - 1] = true;
				}
			}
		}
	}

	// columns
	for (int row = 0; row < N; row++) {
		clearArray(visited, N);
		for (int col = 0; col < N; col++) {
			int val = board[col * N + row];

			if (val != 0) {
				if (visited[val - 1]) {
					return false;
				}
				else {
					visited[val - 1] = true;
				}
			}
		}
	}

	// sub-boards
	for (int subr = 0; subr < n; subr++) {
		for (int subc = 0; subc < n; subc++) {
			clearArray(visited, N);
			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					int value = board[(subr * n + i) * N + (subc * n + j)];

					if (value != 0) {
						if (visited[value - 1]) {
							return false;
						}
						else {
							visited[value - 1] = true;
						}
					}
				}
			}
		}
	}
	//the board is valid
	return true;
}


/**
* Takes a board and an index between 0 and N * N - 1. This function assumes the board
* without the value at index is valid and checks for validity given the new change.
*
* index: index of the changed value
*/
__device__
bool isBoardValid(const int *board, int index) {

	int r = index / 9;
	int c = index % 9;

	if (index < 0) {
		return isBoardValid(board);
	}

	if ((board[index] < 1) || (board[index] > 9)) { //not the values from sudoku
		return false;
	}

	bool visited[N];// from 0 to 8
	clearArray(visited, N);
	// row (with the value at index)
	for (int i = 0; i < N; i++) {
		int value = board[r * N + i];

		if (value != 0) {
			if (visited[value - 1]) {
				return false;
			}
			else {
				visited[value - 1] = true;
			}
		}
	}

	clearArray(visited, N);
	// column (with the value at index)
	for (int j = 0; j < N; j++) {
		int value = board[j * N + c];

		if (value != 0) {
			if (visited[value - 1]) {
				return false;
			}
			else {
				visited[value - 1] = true;
			}
		}
	}

	// sub-board
	int subr = r / n;
	int subc = c / n;

	clearArray(visited, N);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int value = board[(subr * n + i) * N + (subc * n + j)];

			if (value != 0) {
				if (visited[value - 1]) {
					return false;
				}
				else {
					visited[value - 1] = true;
				}
			}
		}
	}
	//valid
	return true;
}


/**
* Each thread solves the different board using backtracking algorithm.
*
* boards:      The array of boards N*N , where the number of boards is numBoards,
*				boards[x*N*N + r*N + c] - specific value in board x.
*
* numBoards:   The total number of boards in the boards array.
*
* emptySpaces: The array which stores indices of empty spaces, the size of array is numBoards * N * N
*
* numEmptySpaces:  The array which stores number of empty spaces in each board of boards.
*
* finished:    The flag indicating to stop the kernel when solution is found.
*
* solved:      Output array with solution N*N.
*/
__global__
void sudokuBacktrack(int *boards,
	const int numBoards,
	int *emptySpaces,
	int *numEmptySpaces,
	int *finished,
	int *solved) {

	int index = blockDim.x * blockIdx.x + threadIdx.x; // the number of board

	int *currentBoard;
	int *currentEmptySpaces;
	int currentNumEmptySpaces;


	while ((*finished == 0) && (index < numBoards)) { // not finished, not all boards done

		int emptyIndex = 0;// empty spaces index

		currentBoard = boards + index * N * N;// select board
		currentEmptySpaces = emptySpaces + index * N * N;// the empty spaces indices
		currentNumEmptySpaces = numEmptySpaces[index];// the number of empty spaces

		while ((emptyIndex >= 0) && (emptyIndex < currentNumEmptySpaces)) {
			//walk through empty spaces
			currentBoard[currentEmptySpaces[emptyIndex]]++;

			if (!isBoardValid(currentBoard, currentEmptySpaces[emptyIndex])) {

				// all numbers were tried, backtrack
				if (currentBoard[currentEmptySpaces[emptyIndex]] >= 9) {
					currentBoard[currentEmptySpaces[emptyIndex]] = 0;
					emptyIndex--;
				}
			}
			// move forward
			else {
				emptyIndex++;
			}

		}

		if (emptyIndex == currentNumEmptySpaces) { //all spaces filled
			// we found the solution
			*finished = 1;

			// copy board to output
			for (int i = 0; i < N * N; i++) {
				solved[i] = currentBoard[i];
			}
		}

		index += gridDim.x * blockDim.x; // move to next board
	}
}


void cudaSudokuBacktrack(const unsigned int blocks,
	const unsigned int threadsPerBlock,
	int *boards,
	const int numBoards,
	int *emptySpaces,
	int *numEmptySpaces,
	int *finished,
	int *solved) {

	sudokuBacktrack << <blocks, threadsPerBlock >> >
		(boards, numBoards, emptySpaces, numEmptySpaces, finished, solved);
}


/**
* This is generating kernel, which genearates next boards from old one.
* Uses breadth first search to find new boards.
*
* old_boards:      Each N * N section is another board. This array stores the previous set of boards.
*
* new_boards:      This array stores the next set of boards.
*
* total_boards:    Number of old boards.
*
* board_index:     Index specifying the index of the next frontier in new_boards.
*
* empty_spaces:    Each N * N section is another board, storing the
*                  indices of empty spaces in new_boards.
*
* empty_space_count:   empty spaces number in corresponding board.
*/
__global__
void
cudaBFSKernel(int *old_boards,
	int *new_boards,
	int total_boards,
	int *board_index,
	int *empty_spaces,
	int *empty_space_count) {

	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;// index of board

	while (index < total_boards) {
		// empty space index
		int found = 0;

		for (int i = (index * N * N); (i < (index * N * N) + N * N) && (found == 0); i++) {// search in each board
			// found an empty space
			if (old_boards[i] == 0) {
				found = 1;
				int temp = i - N * N * index;
				int row = temp / N;
				int col = temp % N;

				// try numbers
				for (int attempt = 1; attempt <= N; attempt++) {
					int correct = 1;
					// row constraint, test columns
					for (int c = 0; c < N; c++) {
						if (old_boards[row * N + c + N * N * index] == attempt) {// found equal in column
							correct = 0;
						}
					}
					// column contraint, test rows
					for (int r = 0; r < N; r++) {
						if (old_boards[r * N + col + N * N * index] == attempt) {// found equal in row
							correct = 0;
						}
					}
					// sub-board
					for (int r = n * (row / n); r < n; r++) {
						for (int c = n * (col / n); c < n; c++) {
							if (old_boards[r * N + c + N * N * index] == attempt) {// equal in sub-board
								correct = 0;
							}
						}
					}
					if (correct == 1) {
						// copy the whole board to new boards

						int next_board_index = atomicAdd(board_index, 1);// stores result back at same address
						int empty_index = 0;
						for (int r = 0; r < N; r++) {
							for (int c = 0; c < N; c++) {
								new_boards[next_board_index * N * N + r * N + c] = old_boards[index * N * N + r * N + c];
								if (old_boards[index * N * N + r * N + c] == 0 && (r != row || c != col)) {
									empty_spaces[empty_index + N * N * next_board_index] = r * N + c;// the index of empty space
									empty_index++;// count empty spaces
								}
							}
						}
						empty_space_count[next_board_index] = empty_index;
						new_boards[next_board_index * N * N + row * N + col] = attempt;// put the correct number
					}
				}
			}
		}

		index += blockDim.x * gridDim.x; // move forward
	}
}


void callBFSKernel(const unsigned int blocks,
	const unsigned int threadsPerBlock,
	int *old_boards,
	int *new_boards,
	int total_boards,
	int *board_index,
	int *empty_spaces,
	int *empty_space_count) {
	cudaBFSKernel << <blocks, threadsPerBlock >> >
		(old_boards, new_boards, total_boards, board_index, empty_spaces, empty_space_count);
}
