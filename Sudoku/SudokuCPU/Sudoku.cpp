#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>

#define N 9
#define n 3

using namespace std;



void clearArray(bool *map, int size) {
	for (int i = 0; i < size; i++) {
		map[i] = false;
	}
}


bool isBoardValid(int *board) {
	bool visited[N];
	clearArray(visited, N);

	// rows
	for (int i = 0; i < N; i++) {
		clearArray(visited, N);

		for (int j = 0; j < N; j++) {
			int val = board[i * N + j];

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

	// columns
	for (int j = 0; j < N; j++) {
		clearArray(visited, N);

		for (int i = 0; i < N; i++) {
			int val = board[i * N + j];

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
	for (int ridx = 0; ridx < n; ridx++) {
		for (int cidx = 0; cidx < n; cidx++) {
			clearArray(visited, N);

			for (int i = 0; i < n; i++) {
				for (int j = 0; j < n; j++) {
					int val = board[(ridx * n + i) * N + (cidx * n + j)];

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
		}
	}

	return true;
}

// the r c value is a change
bool isBoardValid(int *board, int r, int c) {

	// if r is less than 0, then just default case
	if (r < 0) {
		return isBoardValid(board);
	}

	bool visited[N];
	clearArray(visited, N);

	// row
	for (int i = 0; i < N; i++) {
		int val = board[r * N + i];

		if (val != 0) {
			if (visited[val - 1]) {
				return false;
			}
			else {
				visited[val - 1] = true;
			}
		}
	}

	// column
	clearArray(visited, N);
	for (int j = 0; j < N; j++) {
		int val = board[j * N + c];

		if (val != 0) {
			if (visited[val - 1]) {
				return false;
			}
			else {
				visited[val - 1] = true;
			}
		}
	}

	// sub-board
	int ridx = r / n;
	int cidx = c / n;

	clearArray(visited, N);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int val = board[(ridx * n + i) * N + (cidx * n + j)];

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
	// valid
	return true;
}

// everything is filled
bool doneBoard(int *board) {
	for (int i = 0; i < N * N; i++) {
		if (board[i] == 0) { // the empty space
			return false;
		}
	}

	return true;
}

// write values to input
bool findEmptySpot(int *board, int *row, int *col) {
	for (int r = 0; r < N; r++) {
		for (int c = 0; c < N; c++) {
			if (board[r * N + c] == 0) {
				*row = r;
				*col = c;
				return true;
			}
		}
	}

	return false;
}



bool solveHelper(int *board) {
	int row = 10; //invalid values
	int col = 10;
	if (!findEmptySpot(board, &row, &col)) { 
		return true;
	}

	for (int attempt = 1; attempt <= N; attempt++) { // try
		board[row * N + col] = attempt;
		if (isBoardValid(board, row, col) && solveHelper(board)) {
			return true;
		}
		board[row * N + col] = 0; // try another
	}

	return false;
}

// recursive solution
bool solve(int *board) {

	// initial board is invalid
	if (!isBoardValid(board, -1, -1)) {

		printf("solve: invalid board\n");
		return false;
	}

	// board is already solved
	if (doneBoard(board)) {

		printf("solve: done board\n");
		return true;
	}

	// otherwise, try to solve the board
	if (solveHelper(board)) {

		// solved
		printf("solve: solved board\n");
		return true;
	}
	else {

		// unsolvable
		printf("solve: unsolvable\n");
		return false;
	}
}


void printBoard(int *board) {
	for (int i = 0; i < N; i++) {
		if (i % n == 0) {
			printf("-----------------------\n");
		}

		for (int j = 0; j < N; j++) {
			if (j % n == 0) {
				printf("| ");
			}
			printf("%d ", board[i * N + j]);
		}

		printf("|\n");
	}
	printf("-----------------------\n");
}


void load(char *FileName, int *board) {
	FILE * a_file = fopen(FileName, "r");

	if (a_file == NULL) {
		printf("File load fail!\n"); return;
	}

	char temp;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (!fscanf(a_file, "%c\n", &temp)) {
				printf("File loading error!\n");
				return;
			}

			if (temp >= '1' && temp <= '9') {
				board[i * N + j] = (int)(temp - '0');
			}
			else {
				board[i * N + j] = 0;
			}
		}
	}
}

int main() {
	int *board = new int[N * N];
	load("example.txt", board);

	if (solve(board)) {
		// solved
		cout << "solved" << endl;

		// return the solved board
		printBoard(board);
	}


	return 0;
}