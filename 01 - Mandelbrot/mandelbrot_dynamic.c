// Mandelbrot set visualiser using MPI for use on a supercomputer.
// Written by Alec Dorrington for HPC at Curtin University.
// Uses dynamic decomposition.

// Horizontal pixel resolution of image.
#define WIDTH 4000

// Vertical pixel resolution of image.
#define HEIGHT 4000

// Maximum number of iterations before giving up.
#define MAX_ITR 1000

// Number of pixels to process in each chunk.
#define CHUNK_SIZE 50000

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

// Compute the number of mandelbrot iterations for all pixels in a range.
int* processChunk(int base, int limit, int width, int height, int maxItr) {
    
    int *results = malloc(limit * sizeof(int));
    int i;
    
    // For each pixel in this block.
    for(i = 0; i < limit; i++) {
        results[i] = mandelbrot(toComplex(base + i, width, height), maxItr);
    }
    return results;
}

// Output the results of the computation to a PPM (image) file.
void writeFile(char *name, int *image, int width, int height, float maxItr) {
    
    FILE *file = fopen(name, "w");
    int i, r, g, b, x;
    
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
}

// Print statistics.
void printStats(int workers, int width, int height,
        int chunkSize, int maxItr, char *file, double time) {
    
    printf("  IMAGE SIZE: %dx%d px\n", width, height);
    printf("  CHUNK SIZE: %d px\n", chunkSize);
    printf("  WORKERS: %d (+1)\n", workers);
    printf("  MAX ITERATIONS: %d\n", maxItr);
    printf("  FILE NAME: %s\n", file);
    printf("  TOTAL TIME: %2.3f secs\n", time);
}

// Entry point for master process.
void master(int workers, int width, int height,
        int chunkSize, int maxItr, char *file) {
    
    MPI_Status status;
    int base = 0, source, offset, completed = 0;
    int *image = (int*) malloc(width * height * sizeof(int));
    int *tasks = (int*) calloc(workers * 2, sizeof(int));
    double t0 = MPI_Wtime();
    
    // While there remains work to be done.
    while(base < width * height || completed++ < workers) {
    
        // Wait for worker to send chunk.
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        source = status.MPI_SOURCE;
        offset = (source - 1) * 2;
        
        // Load new chunk into existing image array.
        MPI_Recv(image + tasks[offset], tasks[offset+1], MPI_INT,
                MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    
        // Create a new task for the worker to complete next.
        tasks[offset] = base;
        tasks[offset+1] = chunkSize < width * height - base ?
               chunkSize : width * height - base;
        
        // Send new task to worker.
        MPI_Send(tasks + offset, 2, MPI_INT, source, 0, MPI_COMM_WORLD);
        base += tasks[offset+1];
    }
    // Print statistics, and write image to file.
    printStats(workers, width, height, chunkSize,
            maxItr, file, MPI_Wtime() - t0);
    writeFile(file, image, width, height, maxItr);
}

// Entry point for worker process.
void worker(int rank, int size, int width, int height, int maxItr) {
    
    MPI_Status status;
    int task[2] = {0, 0}, *results;
    
    // While there remains work to be done.
    while(task[0] < width * height) {
        
        // Send the previous chunk (if any) to master.
        MPI_Send(results, task[1], MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        // Receive next task from master.
        MPI_Recv(task, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        if(results) free(results);
        // Perform calculations for this chunk.
        results = processChunk(task[0], task[1], width, height, maxItr);
    }
    if(results) free(results);
}

// Mandelbrot visualiser.
int main(int argc, char *argv[]) {

    int rank, size;
    
    // Initialise MPI.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Call master() or worker() accordingly.
    if(rank == 0) master(size-1, WIDTH, HEIGHT, CHUNK_SIZE, MAX_ITR, OUTPUT);
    else worker(rank, size, WIDTH, HEIGHT, MAX_ITR);
    
    MPI_Finalize();
    return 0;
}
