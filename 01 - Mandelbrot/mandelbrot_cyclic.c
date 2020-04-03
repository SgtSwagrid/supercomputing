// Mandelbrot set visualiser using MPI for use on a supercomputer.
// Written by Alec Dorrington for HPC at Curtin University.
// Uses static cyclic decomposition.

// Horizontal pixel resolution of image.
#define WIDTH 4000

// Vertical pixel resolution of image.
#define HEIGHT 4000

// Maximum number of iterations before giving up.
#define MAX_ITR 1000

// Output image file.
#define OUTPUT "mandelbrot.ppm"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <mpi.h>

// Compute the number of mandelbrot iterations
// required before a value escapes to infinity.
int mandelbrot(float complex z, int maxItr) {
    
    float complex z0 = z;
    int itr = 0;
    
    // A value is guaranteed to escape to infinity
    // once its absolute value equals or exceeds 2.

    // If this never occurs, stop after maxItr.
    
    while(cabs(z) < 2.0 && ++itr < maxItr) {
        z = z * z + z0;
    }
    return itr;
}

// Convert an index to a value in the complex plane, within the region
// -2 < x < 2, -2 < y < 2, with the given horizontal/vertical resolution.
float complex toComplex(int i, int width, int height) {
    
    // Convert index to cartesian coordinates.
    int x = i % height;
    int y = i / height;
    
    // Convert coordinates to complex number.
    float re = 4.0F * ((float) x + 0.5F) / (float) width - 2.0F;
    float im = 4.0F * ((float) y + 0.5F) / (float) height - 2.0F;
    
    return re + im * I;
}

// Compute the number of mandelbrot iterations for all pixels in a cyclic group.
int* processCycle(int offset, int size, int stride,
        int width, int height, int maxItr) {
    
    int *results = malloc(stride * sizeof(int)), i;
    
    // For each pixel in this block.
    for(i = 0; i < stride; i++) {
        results[i] = mandelbrot(toComplex(
                offset + i * size, width, height), maxItr);
    }
    return results;
}

// Output the results of the computation to a PPM (image) file.
void writeFile(char *name, int *image, int width, int height, float maxItr) {
    
    FILE *file = fopen(name, "w");
    int i, r, g, b, x;
    double t0 = MPI_Wtime();
    
    // Write image resolution.
    fprintf(file, "P3\n%d %d\n255\n", width, height);
    
    // For each pixel.
    for(i = 0; i < width * height; i++) {
        // Determine colour based on number of iterations.
        x = (int) (255.0F * log1p(image[i]) / log(maxItr));
        r = image[i] < 127 ? 255 - 2*x : 0;
        g = image[i] < 127 ? 2*x : (511 - 2*x);
        b = image[i] < 127 ? 0 : (2*x - 255);
        // Write colour in (r, g, b) format.
        fprintf(file, "%3d\n%3d\n%3d\n", r, g, b);
    }
    fclose(file);
    printf("Image written to %s in %2.3f secs.\n", name, MPI_Wtime() - t0);
    fflush(stdout);
}

// Print performance statistics associated with this node.
void printStats(int rank, int size, int pixels,
        double pTime, double cTime) {
    
    int i;
    // Ensure output is printed in the correct order.
    for(i = 0; i < size; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        if(i == rank) {
            // Print stats for this node.
            printf("Node %d processed %d pixels "
                    "in %2.3f secs (+%2.3f secs overhead).\n",
                    rank, pixels, pTime, cTime);
            fflush(stdout);
        }
    }
}

// Unjumble the results of a cyclic decomposition.
void rearrange(int size, int pixels, int stride, int *jumbled, int *image) {
    int i;
    for(i = 0; i < pixels; i++) {
        image[i] = jumbled[(i % size) * stride + (i / size)];
    }
}

// Mandelbrot visualiser.
int main(int argc, char *argv[]) {
    
    int *results, *jumbled, *image, rank, size, stride;
    int *tasks = (int*) malloc(size * 2 * sizeof(int));
    double t0 = MPI_Wtime(), t1, t2;
    
    // Initialise MPI.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Process the pixels within this cycle.
    t1 = MPI_Wtime();
    stride = (WIDTH * HEIGHT + size - 1) / size;
    results = processCycle(rank, size, stride, WIDTH, HEIGHT, MAX_ITR);
    t2 = MPI_Wtime();
    
    //Allocate memory for completed image.
    if(rank == 0) {
        jumbled = (int*) malloc(stride * size * sizeof(int));
        image = (int*) malloc(WIDTH * HEIGHT * sizeof(int));
    }
    
    // Combine all pixels into a single image.
    MPI_Gather(results, stride, MPI_INT, jumbled,
            stride, MPI_INT, 0, MPI_COMM_WORLD);
    if(rank == 0) rearrange(size, WIDTH * HEIGHT, stride, jumbled, image);
    
    // Print performance statistics.
    printStats(rank, size, stride, t2 - t1, MPI_Wtime() - t0 + t1 - t2);
    
    // Write image to PPM file.
    if(rank == 0) writeFile(OUTPUT, image, WIDTH, HEIGHT, MAX_ITR);
    
    // Free memory and finalize MPI.
    free(results);
    if(rank == 0) {
        free(jumbled);
        free(image);
    }
    MPI_Finalize();
    return 0;
}
