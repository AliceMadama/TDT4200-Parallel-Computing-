#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

// Convert 'struct timeval' into seconds in double prec. floating point
#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

typedef int64_t int_t;
typedef double real_t;

int_t
    N,
    M,
    max_iteration,
    snapshot_frequency;

real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dx,
    dt;

int
    rank,
    size,
    origin,
    length,
    up_neighbor,
    down_neighbor;

#define T(i,j)                      temp[0][(i) * (M + 2) + (j)]
#define T_next(i,j)                 temp[1][((i) * (M + 2) + (j))]
#define THERMAL_DIFFUSIVITY(i,j)    thermal_diffusivity[(i) * (M + 2) + (j)]

void time_step ( void );
void boundary_condition( void );
void border_exchange( void );
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
    // Todo1
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    //boradcasting
    if (rank == 0)
    {
        OPTIONS *options = parse_args(argc, argv);
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }
        N = options->N;
        M = options->M;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;
        free(options);
    }
    
    MPI_Bcast(&N, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_frequency, 1, MPI_INTEGER, 0, MPI_COMM_WORLD);
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {

        border_exchange();

        boundary_condition();

        time_step();

        if ( iteration % snapshot_frequency == 0 )
        {
            printf (
                "Iteration %ld of %ld (%.2lf%% complete)\n",
                iteration,
                max_iteration,
                100.0 * (real_t) iteration / (real_t) max_iteration
            );

            domain_save ( iteration );
        }

        swap( &temp[0], &temp[1] );
    }
    gettimeofday ( &t_end, NULL );
    printf ( "Total elapsed time: %lf seconds\n",
            WALLTIME(t_end) - WALLTIME(t_start)
            );
    domain_finalize();
    // todo1
    MPI_Finalize();
    exit ( EXIT_SUCCESS );
}


void
time_step ( void )
{
    real_t c, t, b, l, r, K, new_value;

    // just changing indexes

    for ( int_t x = 1; x <= length; x++ )
    {
        for ( int_t y = 1; y <= M; y++ )
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
    //Left and right boundaries
    for ( int_t x = 1; x <= length; x++ )
    {
        T(x, 0) = T(x, 2);
        T(x, M+1) = T(x, M-1);
    }
  
    //Upper boundary
    if (rank == 0)
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            T(0, y) = T(2, y);
        }
    }

    //Lower boundary
    if (rank == (size - 1))
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            T(length+1, y) = T(length-1, y);
        }
    }
}


void
border_exchange ( void )
{
    // don't exchange with 1 process
    if (size == 1) return;
    
    real_t *first_row_pr = &T(1, 1); //send
    real_t *upper_boundary_pr = &T(0, 1); //receive
    real_t *last_row_pr = &T(length, 1); //send
    real_t *lower_boundary_pr = &T(length + 1, 1); //receive

        // if even
    if (rank % 2 == 0)
    {

        if (rank != 0)
        {
            MPI_Send(first_row_pr, M, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD);
            MPI_Recv(upper_boundary_pr, M, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank != size - 1)
        {
            MPI_Send(last_row_pr, M, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD);
            MPI_Recv(lower_boundary_pr, M, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    } else
    {

        if (rank != 0)
        {
            MPI_Recv(upper_boundary_pr, M, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(first_row_pr, M, MPI_DOUBLE, up_neighbor, 0, MPI_COMM_WORLD);
        }
        if (rank != size - 1)
        {
            MPI_Recv(lower_boundary_pr, M, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(last_row_pr, M, MPI_DOUBLE, down_neighbor, 0, MPI_COMM_WORLD);
        }
    
    }
}


void
domain_init ( void )
{
    // declare the values
    real_t
        temperature,
        diffusivity;
    
    // some variables for allocation
    length = N / size;
    origin = length * rank;
    down_neighbor = (rank+1) % size;
    up_neighbor = (rank == 0) ? size - 1 : rank - 1;

    //Length of a row per number of rows
    int_t alloc_size = (M+2) * (length+2) * sizeof(real_t);

    // allocate
    temp[0] = malloc ( alloc_size );
    temp[1] = malloc ( alloc_size );
    thermal_diffusivity = malloc ( alloc_size );

    dt = 0.1;
    dx = 0.1;
    
    // start from 1
    for ( int_t x = 1; x <= length; x++ )
    {
        for ( int_t y = 1; y <= M; y++ )
        {
            temperature = 30 + 30 * sin((origin + x + y) / 20.0);
            diffusivity = 0.05 + (30 + 30 * sin((N - (x + origin) + y) / 20.0)) / 605.0;
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

    MPI_File out;
    MPI_File_open(
        MPI_COMM_WORLD,
        filename,
        MPI_MODE_CREATE | MPI_MODE_WRONLY,
        MPI_INFO_NULL,
        &out
    );
    // pointer buffers
    real_t *buffer_pr;
    MPI_Offset offset;
    int count;

    if (rank == 0)
    {

        buffer_pr =  temp[0];
        offset = 0;
        count = (length + 1) * (M+2);
    // important!!!! prints also the last row
    } else if (rank == size - 1)
    {

        buffer_pr = temp[0] + M + 2;
        offset = (rank * (length) + 1) * (M + 2) * sizeof(real_t);
        count = (length + 1) * (M + 2);

    } else
    {

        buffer_pr = temp[0] + M + 2;
        offset = (rank * (length) + 1) * (M + 2) * sizeof(real_t);
        count = (length) * (M + 2);

    }
    MPI_File_write_at_all(
        out,
        offset,
        buffer_pr,
        count,
        MPI_DOUBLE,
        MPI_STATUS_IGNORE
    );
    MPI_File_close(&out);
}


void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
