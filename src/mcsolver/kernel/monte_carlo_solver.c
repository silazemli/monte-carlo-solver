#include "monte_carlo_solver.h"

static inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline float random_splitmix64(uint64_t* state) {
    uint64_t x = splitmix64(state);
    return (x >> 11) * (1.0 / (1ULL << 53));
}

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
) {
    #pragma omp parallel for schedule(dynamic)
    for (uint32_t start = 0; start < number_of_inner_points; start++) {
        float64_t total = 0.0;
        uint64_t base = 45235236246463ULL;

        for (uint32_t walk = 0; walk < number_of_walks; walk++) {
            uint64_t rng_state = base ^ ((uint64_t)start << 32) ^ (uint64_t)walk;
            
            float64_t sum = 0.0;
            float64_t norm = 1.0;
            uint32_t position = start;

            for (uint32_t step = 0; step < max_steps; step++) {
                sum += norm*constants[position];

                if (!norms[position]) break;
                
                norm *= norms[position];
                
                uint32_t begin = offsets[position];
                uint32_t end = offsets[position + 1];

                float64_t random_number = random_splitmix64(&rng_state);
                for (uint32_t neighbor_index = begin; neighbor_index < end; neighbor_index++) {
                    if (random_number < probabilities[neighbor_index]) {
                        position = neighbors[neighbor_index];
                        break;
                    }
                }

            }

            total += sum;
        }
        result[start] = total / (float64_t)number_of_walks;
    }
}

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
) {
    double total = 0.0;
    uint64_t base = 45235236246463ULL;

    for (uint32_t walk = 0; walk < number_of_walks; walk++) {
        uint64_t rng_state = base ^ (uint64_t)walk;

        double sum = start_constant;
        double norm = start_norm;
        
        double random_number = random_splitmix64(&rng_state);

        uint32_t position = start_neighbors[start_degree - 1];
        for (uint32_t i = 0; i < start_degree; i++) {
            if (random_number < start_probabilities[i]) {
                position = start_neighbors[i];
                break;
            }
        }

        for (uint32_t step = 1; step < max_steps; step++) {
            sum += norm * constants[position];

            if (!norms[position])
                break;

            norm *= norms[position];

            uint32_t begin = offsets[position];
            uint32_t end = offsets[position + 1];

            random_number = random_splitmix64(&rng_state);

            for (uint32_t neighbor_index = begin;
                 neighbor_index < end;
                 neighbor_index++) {

                if (random_number < probabilities[neighbor_index]) {
                    position = neighbors[neighbor_index];
                    break;
                }
            }
        }

        total += sum;
    }

    return total / (double)number_of_walks;
}