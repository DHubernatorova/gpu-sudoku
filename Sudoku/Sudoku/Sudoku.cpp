#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <cstring>


#include <cuda_runtime.h>
#include <algorithm>
#include <curand.h>

#include "Sudoku.cuh"



void load(char *fileName, int *board) {
	FILE * a_file = fopen(fileName, "r");

	if (a_file == NULL) {
		printf("File load fail!\n"); return;
	}

	char c;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (!fscanf(a_file, "%c\n", &c)) {
				printf("File loading error!\n");
				return;
			}

			if (c >= '1' && c <= '9') {
				board[i * N + j] = (int)(c - '0');
			}
			else {
				board[i * N + j] = 0;
			}
		}
	}
	fclose(a_file);
}

/**
* fd: the file descriptor to print to
*/
void printBoard(FILE* fd, int *board) {
	for (int i = 0; i < N; i++) {
		if (i % n == 0) {
			fprintf(fd, "-----------------------\n");
		}

		for (int j = 0; j < N; j++) {
			if (j % n == 0) {
				fprintf(fd, "| ");
			}
			fprintf(fd, "%d ", board[i * N + j]);
		}

		fprintf(fd, "|\n");
	}
	fprintf(fd, "-----------------------\n");
}

void writeToFile(char* filename, int* board) {
	FILE* a_file = fopen(filename, "w");
	if (a_file == NULL) {
		printf("File load fail!\n"); return;
	}
	printBoard(a_file, board);
	fclose(a_file);
}

int main(int argc, char* argv[]) {
	if (argc < 5) {
		printf("Usage: (threads per block) (max number of blocks) (filename) (output filename)\n");
		exit(-1);
	}

	const unsigned int threadsPerBlock = atoi(argv[1]);
	const unsigned int maxBlocks = atoi(argv[2]);
	// the puzzle stored in this file
	char* filename = argv[3];
	// the output stored here
	char* output = argv[4];

	// load the board
	int *board = new int[N * N];
	load(filename, board);

	// the generated boards after the next iteration of breadth first search
	int *new_boards;
	// the previous boards, which form the frontier of the breadth first search
	int *old_boards;
	// stores the location of the empty spaces (indices of them) in the boards
	int *empty_spaces;
	// stores the number of empty spaces in each board
	int *empty_space_count;
	// the address of new board to store
	int *board_index;

	// maximum number of boards from breadth first search
	const int maxn = pow(2, 26); // branching factor, depth

	// allocate memory on the device
	cudaMalloc(&empty_spaces, maxn * sizeof(int));
	cudaMalloc(&empty_space_count, (maxn / 81 + 1) * sizeof(int));
	cudaMalloc(&new_boards, maxn * sizeof(int));
	cudaMalloc(&old_boards, maxn * sizeof(int));
	cudaMalloc(&board_index, sizeof(int));

	// at the beginning one board
	int total_boards = 1;

	// initialize memory
	cudaMemset(board_index, 0, sizeof(int));
	cudaMemset(new_boards, 0, maxn * sizeof(int));
	cudaMemset(old_boards, 0, maxn * sizeof(int));

	// copy the initial board to the old boards
	cudaMemcpy(old_boards, board, N * N * sizeof(int), cudaMemcpyHostToDevice);

	// call the kernel to generate boards
	callBFSKernel(maxBlocks, threadsPerBlock, old_boards, new_boards, total_boards, board_index,
		empty_spaces, empty_space_count);

	// number of boards after a call to BFS
	int host_count;
	// number of iterations to run BFS for
	int iterations = 18;

	// loop through BFS iterations to generate more boards deeper in the tree
	for (int i = 0; i < iterations; i++) {
		cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);

		printf("total boards after an iteration number %d: %d\n", i, host_count);

		cudaMemset(board_index, 0, sizeof(int));


		if (i % 2 == 0) {
			callBFSKernel(maxBlocks, threadsPerBlock, new_boards, old_boards, host_count, board_index, empty_spaces, empty_space_count);
		}
		else {
			callBFSKernel(maxBlocks, threadsPerBlock, old_boards, new_boards, host_count, board_index, empty_spaces, empty_space_count);
		}
	}

	cudaMemcpy(&host_count, board_index, sizeof(int), cudaMemcpyDeviceToHost);
	printf("new number of boards retrieved is %d\n", host_count);

	// flag to determine when a solution has been found by a device
	int *dev_finished;
	// output to store solved board in
	int *dev_solved;

	// allocate memory on the device
	cudaMalloc(&dev_finished, sizeof(int));
	cudaMalloc(&dev_solved, N * N * sizeof(int));

	// initialize memory
	cudaMemset(dev_finished, 0, sizeof(int));
	cudaMemcpy(dev_solved, board, N * N * sizeof(int), cudaMemcpyHostToDevice);

	if (iterations % 2 == 1) {
		// if odd number of iterations run, then send to device old boards not new boards
		new_boards = old_boards;
	}

	cudaSudokuBacktrack(maxBlocks, threadsPerBlock, new_boards, host_count, empty_spaces,
		empty_space_count, dev_finished, dev_solved);


	// copy back the solved board
	int *solved = new int[N * N];

	memset(solved, 0, N * N * sizeof(int));

	cudaMemcpy(solved, dev_solved, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	printBoard(stdout, solved);
	writeToFile(output, solved);

	// free memory
	delete[] board;
	delete[] solved;

	cudaFree(empty_spaces);
	cudaFree(empty_space_count);
	cudaFree(new_boards);
	cudaFree(old_boards);
	cudaFree(board_index);

	cudaFree(dev_finished);
	cudaFree(dev_solved);

	return 0;

}