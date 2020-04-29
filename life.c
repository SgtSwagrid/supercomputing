// Conway's Game of Life simulator, using OpenMP.
// Written by Alec Dorrington.

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

// Horizontal and vertical grid size.
#define SIZE 511
// Number of game iterations to simulate.
#define ITERATIONS 256
// Pixel width of each grid square.
#define SCALE 2
// Name of file(s) to write output image(s).
#define FILENAME "life%3.3d.pgm"
// Chunk size (number of rows) used for dynamic decomposition.
#define CHUNKSIZE 1

// Modulo function which works for negative numbers.
#define MOD(a, b) (((a) + (b)) % (b))

// Function type used to generate initial board configurations.
// Given a grid size and (x, y) coordinate,
// return whether this position is filled (0=no, 1=yes).
typedef int (*generator_t)(int, int, int);

// Write a Game of Life grid to an image file.
void writePgm(int **grid, int size, int scale, char *fileName) {

  int x, y;
  FILE *file = fopen(fileName, "w");

  // Write the image header specifying dimensions.
  fprintf(file, "P2\n%4d %4d\n1\n", size * scale, size * scale);

  // Write each pixel in the output image.
  for(x = 0; x < size * scale; x++) {
    for(y = 0; y < size * scale; y++) {
      fprintf(file, "%d\n", grid[x / scale][y / scale]);
    }
  }
  fclose(file);
}

// Create a size x size grid where f specifies the original configuration.
// Filled squares are represented by a 1, while empty squares are 0.
int** createGrid(int size, generator_t f) {

  int x, y;
  // Allocate the outer array.
  int **grid = malloc(size * sizeof(int*));

  // Allocate each inner array.
  for(x = 0; x < size; x++) {
    grid[x] = malloc(size * sizeof(int));

    // Use f to decide whether each square is empty or filled.
    for(y = 0; y < size; y++) {
      grid[x][y] = f(size, x, y);
    }
  }
  return grid;
}

// Free the memory allocated for a grid.
void freeGrid(int **grid, int size) {

  int x;
  for(x = 0; x < size; x++) {
    free(grid[x]);
  }
  free(grid);
}

// Determine how many of the 8 adjacent squares are filled.
// Uses cyclic boundary conditions so that the grid is wrapped like a torus.
int adjacent(int **grid, int size, int x, int y) {

  int xx, yy, adj = 0;

  // For each adjacent square.
  for(xx = -1; xx <= 1; xx++) {
    for(yy = -1; yy <= 1; yy++) {

      // Count each filled square which isn't this one.
      if(xx != 0 || yy != 0) {
        adj += grid[MOD(x + xx, size)][MOD(y + yy, size)];
      }
    }
  }
  return adj;
}

// Perform a single iteration of the Game of Life.
// Takes the state of grid1 and puts the next state in grid2.
void life(int **grid1, int **grid2, int size) {

  int x, y, adj;

  // For each grid square (in parallel).
  #pragma omp parallel for \
    shared(grid1, grid2, size) private(x, y, adj) \
    schedule(dynamic, CHUNKSIZE)
  for(x = 0; x < size; x++) {
    for(y = 0; y < size; y++) {

      // Determine the number of filled adjacent squares.
      adj = adjacent(grid1, size, x, y);

      // Progress each square according to the Game of Life rules.
      if(adj < 2 || adj > 3) grid2[x][y] = 0;
      if(adj == 2) grid2[x][y] = grid1[x][y];
      if(adj == 3) grid2[x][y] = 1;
    }
  }
}

// Function to generate an empty Game of Life grid.
int EMPTY(int size, int x, int y) {
  return 0;
}

// Function to generate a Game of Life grid with a simple cross.
int CROSS(int size, int x, int y) {
  return x == size / 2 || y == size / 2;
}

// Get the number of milliseconds since the Unix epoch.
long timeMs() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec * 1000 + time.tv_usec / 1000;
}

int main(int argc, int **argv) {

  int i; long t;
  char fileName[20];
  int **grid1 = createGrid(SIZE, CROSS),
    **grid2 = createGrid(SIZE, EMPTY), **swap;

  // Write the initial grid configuration to an image file.
  sprintf(fileName, FILENAME, 0);
  writePgm(grid1, SIZE, SCALE, fileName);

  t = timeMs();

  // Continue incrementing the grid state for some number of iterations.
  for(i = 0; i < ITERATIONS; i++) {
    life(grid1, grid2, SIZE);
    // Swap grid1 and grid2 so that grid1 is always the most recent.
    swap = grid1; grid1 = grid2; grid2 = swap;
  }

  printf("Completed %d Game of Life iterations on a %dx%d grid in %dms.\n",
    ITERATIONS, SIZE, SIZE, timeMs() - t);

  // Write the final grid configuration to an image file.
  sprintf(fileName, FILENAME, ITERATIONS);
  writePgm(grid1, SIZE, SCALE, fileName);

  freeGrid(grid1, SIZE); freeGrid(grid2, SIZE);
  return 0;
}
