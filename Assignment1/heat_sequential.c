#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>

#include "../inc/argument_utils.h"

/* Convenient definitions */
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// TODO 4: Implement the macro MAX(a,b) which is used to find the largest value of a and b.
//         It currently only gives the value of a.
#define MAX(a,b) ((a) > (b) ? (a) : (b))

typedef int64_t int_t;
typedef double real_t;

/* Simulation parameters */
int_t
    N,                                  // How many points to discretize the domain into
    max_iteration,                      // Number of time steps to simulate
    snapshot_frequency;                 // How often to write system state into files

const real_t
    x_range[2] = { -1.0, 1.0 },         // Physical extent of the domain
    threshold = 5e-9;                   // Results converge to 8 decimal points

/* Derived parameters, initialized in domain_init()  */
real_t
    dx,                                 // Space step
    dt,                                 // Time step
    a_left,                             // Lower/left diagonal coefficient
    a_right,                            // Upper/right diagonal coefficient
    a_diag,                             // Diagonal coefficient

    // TODO 2: Allocate memory for the temperature values.
    //         The global array temp should contain 3 pointers to heap-allocated memory
    //         with enough space for N real_t elements.
    *temp[3] = { NULL, NULL, NULL };    // Buffers for temperature values

/* Indexing macros for the temperature value buffers */
#define T(i)        temp[0][(i)]
#define T_prev(i)   temp[1][(i)]
#define T_prime(i)  temp[2][(i)]

/* Function definitions */
void domain_init(void);
void domain_save(int_t step);
void domain_finalize(void);

// Our three solvers
// These three functions return the number of iterations they required
// to reach convergence in the current time step
int_t time_step_jacobi(void);
int_t time_step_gauss_seidel(void);
int_t time_step_red_black_gauss_seidel(void);

// Pointer to one of the solvers, so that we don't need to implement
// a check for which solver to use inside of the time integration loop
int_t (*time_step)(void);

void swap(real_t** m1, real_t** m2) {
    // TODO 3: Implement the swap function to swap the content of two variables.
    real_t* temp = *m1;
    *m1 = *m2;
    *m2 = temp;
}

int main(int argc, char** argv) {
    OPTIONS* options = parse_args(argc, argv);
    if (!options) {
        fprintf(stderr, "Argument parsing failed\n");
        exit(1);
    }

    // TODO 1: Get N, max_iteration, snapshot_frequency from the options struct
    //         and store the values in the global fields with the same names.
    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    if (options->solver_type == 1) {
        time_step = &time_step_jacobi;
    } else if (options->solver_type == 2) {
        time_step = &time_step_gauss_seidel;
    } else {
        time_step = &time_step_red_black_gauss_seidel;
    }

    domain_init();

    int_t total_iteration_count = 0;

    struct timeval t_start, t_end;

    // TODO 6: Record the execution time of the integration loop
    //         using the structs t_start and t_end declared above.
    gettimeofday(&t_start, NULL);

    for (int_t step = 0; step <= max_iteration; step++) {
        total_iteration_count += time_step();

        if (0 == (step % snapshot_frequency)) {
            domain_save(step / snapshot_frequency);
            printf("Iteration %ld of %ld (%.2lf%% complete)\n", step, max_iteration,
                   100.0 * (real_t)step / (real_t)max_iteration);
        }

        // TODO 3: Implement the swap function to swap the content of two variables
        swap(&temp[0], &temp[1]);

        // Do not increment 'step' here; it is already incremented in the loop header.
    }

    gettimeofday(&t_end, NULL);

    printf("Total iteration count: %ld\n", total_iteration_count);
    printf("Total elapsed time: %lf seconds\n", WALLTIME(t_end) - WALLTIME(t_start));

    domain_finalize();

    exit(EXIT_SUCCESS);
}

int_t time_step_jacobi(void) {
    real_t sigma, max_update;
    int_t iter = 0;

    // Initial guess: values from the previous time step
    memcpy(&T(0), &T_prev(0), N * sizeof(real_t));

    do {
        max_update = 0.0;

        // Left boundary: Neumann condition, use 2 copies of the right neighbor
        sigma = 2.0 * a_right * T(1);
        T_prime(0) = (T_prev(0) - sigma) / a_diag;
        max_update = MAX(max_update, fabs(T_prime(0) - T(0)));

        // Internal points: both neighbors equally weighted
        for (int_t i = 1; i < N - 1; i++) {
            sigma = a_left * T(i - 1) + a_right * T(i + 1);
            T_prime(i) = (T_prev(i) - sigma) / a_diag;
            max_update = MAX(max_update, fabs(T_prime(i) - T(i)));
        }

        // Right boundary: Neumann condition, use 2 copies of the left neighbor
        sigma = 2.0 * a_left * T(N - 2);
        T_prime(N - 1) = (T_prev(N - 1) - sigma) / a_diag;
        max_update = MAX(max_update, fabs(T_prime(N - 1) - T(N - 1)));

        swap(&temp[0], &temp[2]);

        iter = iter + 1;
    } while (max_update > threshold);
    return iter;
}

int_t time_step_gauss_seidel(void) {
    real_t sigma, update, max_update;
    int_t iter = 0;

    // Initial guess: values from the previous time step
    memcpy(&T(0), &T_prev(0), N * sizeof(real_t));

    do {
        max_update = 0.0;

        // Left boundary: Neumann condition, use 2 copies of the right neighbor
        sigma = 2.0 * a_right * T(1);
        update = (T_prev(0) - sigma) / a_diag;
        max_update = MAX(max_update, fabs(update - T(0)));
        T(0) = update;

        // Internal points: both neighbors equally weighted
        for (int_t i = 1; i < N - 1; i++) {
            sigma = a_left * T(i - 1) + a_right * T(i + 1);
            update = (T_prev(i) - sigma) / a_diag;
            max_update = MAX(max_update, fabs(update - T(i)));
            T(i) = update;
        }

        // Right boundary: Neumann condition, use 2 copies of the left neighbor
        sigma = 2.0 * a_left * T(N - 2);
        update = (T_prev(N - 1) - sigma) / a_diag;
        max_update = MAX(max_update, fabs(update - T(N - 1)));
        T(N - 1) = update;

        iter = iter + 1;
    } while (max_update > threshold);
    return iter;
}

int_t time_step_red_black_gauss_seidel(void) {
    real_t sigma, update, max_update;
    int_t iter = 0;

    // Initial guess: values from the previous time step
    memcpy(&T(0), &T_prev(0), N * sizeof(real_t));

    do {
        max_update = 0.0;

        // Left boundary: Neumann condition, use 2 copies of the right neighbor
        sigma = 2.0 * a_right * T(1);
        update = (T_prev(0) - sigma) / a_diag;
        max_update = MAX(max_update, fabs(update - T(0)));
        T(0) = update;

        // Internal points: both neighbors equally weighted, update odd indices
        for (int_t i = 1; i < N - 1; i += 2) {
            sigma = a_left * T(i - 1) + a_right * T(i + 1);
            update = (T_prev(i) - sigma) / a_diag;
            max_update = MAX(max_update, fabs(update - T(i)));
            T(i) = update;
        }

        // Next, do the even indices
        for (int_t i = 2; i < N - 1; i += 2) {
            sigma = a_left * T(i - 1) + a_right * T(i + 1);
            update = (T_prev(i) - sigma) / a_diag;
            max_update = MAX(max_update, fabs(update - T(i)));
            T(i) = update;
        }

        // Right boundary: Neumann condition, use 2 copies of the left neighbor
        sigma = 2.0 * a_left * T(N - 2);
        update = (T_prev(N - 1) - sigma) / a_diag;
        max_update = MAX(max_update, fabs(update - T(N - 1)));
        T(N - 1) = update;

        iter = iter + 1;
    } while (max_update > threshold);
    return iter;
}

void domain_init(void) {
    dx = (x_range[1] - x_range[0]) / (real_t)N;
    dt = (dx * dx) / (2.0);   // Retain numerical stability with alpha = 1

    a_left = -dt / (dx * dx),
    a_right = -dt / (dx * dx),
    a_diag = 1.0 + 2.0 * dt / (dx * dx);

    // TODO 2: Allocate memory for the temperature values.
    //         The global array temp should contain 3 pointers to heap-allocated memory
    //         with enough space for N real_t elements.
    for (int_t i = 0; i < 3; i++) {
        temp[i] = (real_t*)malloc(N * sizeof(real_t));
        if (temp[i] == NULL) {
            fprintf(stderr, "Memory allocation failed.\n");
            exit(1);
        }
    }

    for (int_t i = 0; i < N; i++) {
        real_t x = x_range[0] + i * dx;
        T_prev(i) = pow(exp(-x * x), 4.0);
    }
}

void domain_finalize(void) {
    // TODO 5: Free the heap-allocated memory.
    for (int_t i = 0; i < 3; i++) {
        free(temp[i]);
    }
}

void domain_save(int_t step) {
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.dat", step);

    FILE* out = fopen(filename, "w");
    if (!out) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }
    fwrite(temp[0], sizeof(real_t), N, out);
    fclose(out);
}
    
