#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

#define T(x,y)                      temp[0][(y) * (N + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (N + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (N + 2) + (x)]

void time_step ( void );
void boundary_condition( void );
void domain_init ( void );
void domain_save ( int_t iteration );
void domain_finalize ( void );


void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}


int
main ( int argc, char **argv )
{
    OPTIONS *options = parse_args( argc, argv );
    if ( !options )
    {
        fprintf( stderr, "Argument parsing failed\n" );
        exit(1);
    }

    M = options->M;
    N = options->N;
    max_iteration = options->max_iteration;
    snapshot_frequency = options->snapshot_frequency;

    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    #pragma omp parallel
    {
        int_t thread_id = omp_get_thread_num();
        for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
        {
            boundary_condition();

            time_step();

            #pragma omp barrier
            if ( thread_id == 0 && iteration % snapshot_frequency == 0 )
            {
                printf (
                    "Iteration %ld of %ld (%.2lf%% complete)\n",
                    iteration,
                    max_iteration,
                    100.0 * (real_t) iteration / (real_t) max_iteration
                );

                domain_save ( iteration );
            }
            
            if (thread_id == 0)
            {
                swap( &temp[0], &temp[1] );
            }
            #pragma omp barrier
        }
    }

    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );

    free (options);
    domain_finalize();

    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    //Every thread has to update only their part of the grid.
    #pragma omp for
    for ( int_t y = 1; y <= M; y++ )
    {
        for ( int_t x = 1; x <= N; x++ )
        {
            c = T(x, y);

            t = T(x - 1, y);
            b = T(x + 1, y);
            l = T(x, y - 1);
            r = T(x, y + 1);
            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}


void
boundary_condition ( void )
{
    #pragma omp for
    for ( int_t x = 1; x <= N; x++ )
    {
        T(x, 0) = T(x, 2);
        T(x, M+1) = T(x, M-1);
    }

    #pragma omp for
    for ( int_t y = 1; y <= M; y++ )
    {
        T(0, y) = T(2, y);
        T(N+1, y) = T(N-1, y);
    }
}


void
domain_init ( void )
{
    temp[0] = malloc ( (M+2)*(N+2) * sizeof(real_t) );
    temp[1] = malloc ( (M+2)*(N+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (M+2)*(N+2) * sizeof(real_t) );

    dt = 0.1;

    for ( int_t y = 1; y <= M; y++ )
    {
        for ( int_t x = 1; x <= N; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - x + y) / 20.0)) / 605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}


void
domain_save ( int_t iteration )
{
    int_t index = iteration / snapshot_frequency;
    char filename[256];
    memset ( filename, 0, 256*sizeof(char) );
    sprintf ( filename, "data/%.5ld.bin", index );

    FILE *out = fopen ( filename, "wb" );
    if ( ! out ) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        exit(1);
    }

    fwrite( temp[0], sizeof(real_t), (N+2)*(M+2), out );
    fclose ( out );
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
