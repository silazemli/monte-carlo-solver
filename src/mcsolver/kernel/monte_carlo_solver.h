#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

typedef float float32_t;
typedef double float64_t;

static inline uint64_t splitmix64(uint64_t* state);
static inline float random_splitmix64(uint64_t* state);

void solve(
    uint32_t* offsets,
    uint32_t* neighbors,
    float64_t* probabilities,
    float64_t* constants,
    float64_t* norms,
    uint32_t total_nodes,
    uint32_t number_of_inner_points,
    uint32_t max_steps,
    uint32_t number_of_walks,
    float64_t* result
);

double solve_at(
    uint32_t* offsets,
    uint32_t* neighbors,
    double* probabilities,
    double* constants,
    double* norms,

    uint32_t* start_neighbors,
    double* start_probabilities,
    uint32_t start_degree,
    double start_constant,
    double start_norm,

    uint32_t max_steps,
    uint32_t number_of_walks
);