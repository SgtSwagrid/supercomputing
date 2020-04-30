// Radial distribution function calculator, using OpenMP.
// Written by Alec Dorrington.
// https://en.wikipedia.org/wiki/Radial_distribution_function

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

// Number of bins to divide radii into.
#define RESOLUTION 100
// Chunk size (number of molecules) used for dynamic decomposition.
#define CHUNKSIZE 10
// Name of file to write the RDF to.
#define DST_FILE "rdf.dat"

// Vector type, used for molecule coordinates.
struct vec_t { float x, y, z; };

// Fluid type, containing molecules and cell size.
struct fluid_t {
  struct vec_t *molecules;
  int count;
  float width;
};

// Read fluid data (list of molecules) from a file.
struct fluid_t readFluid(char *fileName) {

  char line[100], name[10];
  struct vec_t pos;
  struct fluid_t fluid;
  int i, j = 0, n;
  FILE *file = fopen(fileName, "r");

  // Read number of molecules and cell size.
  fscanf(file, "%d\n", &n);
  fscanf(file, "%f", &(fluid.width));
  fluid.molecules = (struct vec_t*) malloc(n * sizeof(struct vec_t));
  fluid.count = 0;

  // Read each molecule.
  for(i = 0; i < n; i++) {
    fscanf(file, "%s %f %f %f", &name, &(pos.x), &(pos.y), &(pos.z));
    // Only include oxygen molecules.
    if(!strcmp(name, "O2")) fluid.molecules[fluid.count++] = pos;
  }
  fclose(file);
  return fluid;
}

// Write the radial distribution function to file as a list of bins.
void writeRdf(float *bins, struct fluid_t fluid, int res, char* fileName) {

  int i;
  float dr = fluid.width / 2.0F / res;
  FILE *file = fopen(fileName, "w");

  // Write each bin to the file.
  for(i = 0; i < res; i++) {
    fprintf(file, "%f %f\n", dr * (i + 0.5F), bins[i]);
  }
  fclose(file);
}

// Determine the minimum Euclidean distance between two vertices.
// Assumes periodic boundary conditions with the given cell width.
float distance(struct vec_t pos1, struct vec_t pos2, float width) {

  float xd, yd, zd;

  xd = abs(pos2.x - pos1.x); if(xd > width / 2.0F) xd = width - xd;
  yd = abs(pos2.y - pos1.y); if(yd > width / 2.0F) yd = width - yd;
  zd = abs(pos2.z - pos1.z); if(zd > width / 2.0F) zd = width - zd;

  return (float) sqrt(xd*xd + yd*yd + zd*zd);
}

// Normalize the RDF as a proportion of bulk density.
void normalize(float *bins, struct fluid_t fluid, int res) {

  int i;
  float r, dv;
  // Calculate radius step size and bulk density.
  float dr = fluid.width / 2.0F / res;
  float p = fluid.count / (float) pow(fluid.width, 3);

  // For each bin.
  for(i = 0; i < res; i++) {
    // Calculate the radius and change in volume.
    r = dr * (i+1);
    dv = 4.0F * (float) M_PI * r * r * dr;
    // Average over each choice of 'centre' particle, divide by
    // volume for local density, and normalize using bulk density.
    bins[i] /= fluid.count * dv * p;
  }
}

// Calculate the RDF of a fluid using some number of bins.
float* rdf(struct fluid_t fluid, int res) {

  int i, j, bin;
  float dist;
  float *p_bins, *bins = (float*) calloc(res, sizeof(float));

  #pragma omp parallel \
    shared(fluid, res, bins) private(i, j, dist, bin, p_bins)
  {
    p_bins = (float*) calloc(res, sizeof(float));

    // For each pair of particles, in parallel.
    #pragma omp for schedule(dynamic, CHUNKSIZE)
    for(i = 0; i < fluid.count; i++) {
      for(j = i+1; j < fluid.count; j++) {

        // Calculate the Euclidean distance between the particles.
        dist = distance(fluid.molecules[i], fluid.molecules[j], fluid.width);

        // If this distance is within the range considered.
        if(dist < fluid.width / 2.0F) {
          // Increment the bin corresponding to the distance.
          bin = (int) (res * dist / (fluid.width / 2.0F));
          p_bins[bin] += 2.0F;
        }
      }
    }
    // Merge the cumulative RDF bins of each thread.
    #pragma omp critical
    for(i = 0; i < res; i++) bins[i] += p_bins[i];
    free(p_bins);
  }
  // Normalize the results.
  normalize(bins, fluid, res);
  return bins;
}

// Get the number of milliseconds since the Unix epoch.
long timeMs() {

  struct timeval time;
  gettimeofday(&time, NULL);
  return time.tv_sec * 1000 + time.tv_usec / 1000;
}

int main(int argc, char **argv) {

  struct fluid_t fluid;
  float *bins;
  long t;

  if(argc != 2) printf("Usage: ./rdf [src file]\n");
  else {

    // Read the fluid and calculate the RDF.
    fluid = readFluid(argv[1]);
    t = timeMs();
    bins = rdf(fluid, RESOLUTION);

    // Write the RDF to DST_FILE.
    printf("Calculated the radial distribution function of a ");
    printf("%d-oxygen system in %dms.\n", fluid.count, timeMs() - t);
    writeRdf(bins, fluid, RESOLUTION, DST_FILE);

    free(fluid.molecules);
    free(bins);
  }
  return 0;
}
