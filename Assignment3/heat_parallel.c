#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#include "../inc/argument_utils.h"

#define WALLTIME(t) ((double)(t).tv_sec + 1e-6 * (double)(t).tv_usec)

// Define constants for axis directions.
#define X 0     // moving along the rows
#define Y 1     // moving along the columns

typedef int64_t int_t;
typedef double real_t;

// Global variables
MPI_Comm cartesian_communicator;                // Cartesian communicator for the grid topology.

// MPI Datatypes for file saving and data communication.
MPI_Datatype column_datatype,                   // MPI datatype for a column of data.
             row_datatype,                      // MPI datatype for a row of data.
             grid_datatype,                     // MPI datatype for the entire grid.
             subgrid_datatype;                  // MPI datatype for a subgrid.

// Simulation parameters.
int_t
    M,
    N,
    max_iteration,
    snapshot_frequency;                         /

// Arrays to store temperature values and thermal diffusivity.
real_t
    *temp[2] = { NULL, NULL },
    *thermal_diffusivity,
    dt;

// Macros to access temperature and thermal diffusivity values.
#define T(x,y)                      temp[0][(y) * (ncols + 2) + (x)]
#define T_next(x,y)                 temp[1][((y) * (ncols + 2) + (x))]
#define THERMAL_DIFFUSIVITY(x,y)    thermal_diffusivity[(y) * (ncols + 2) + (x)]

// MPI-related global variables.
int rank,                                       // Rank of the current process.
    size,                                       // Total number of processes.
    cart_rank,                                  // Rank in the Cartesian communicator.
    nrows, ncols;                               // Number of rows and columns for the local subgrid.

// Variables to identify neighboring processes in the Cartesian grid.
int upper,                                      // the upper neighbor.
    lower,                                      // the lower neighbor.
    left,                                       // the left neighbor.
    right;                                      // the right neighbor.

// Variables for Cartesian grid topology.
int dims[2],                                    // Dimensions of the Cartesian grid.
    coord[2],                                   // Coordinates of the current process in the Cartesian grid.
    periodicity[2]={0,0},                       // Periodicity in each dimension (0 for non-periodic).
    start[2];                                   // Starting point for subarray datatype.

// Function prototypes.
void time_step ( void );                        // Function to perform a single time step of the simulation.
void boundary_condition( void );                // Function to apply boundary conditions.
void border_exchange( void );                   // Function to exchange border data with neighboring processes.
void domain_init ( void );                      // Function to initialize the domain and allocate memory.
void create_types (void);                       // Function to create custom MPI datatypes for rows and columns.
void domain_save ( int_t iteration );           // Function to save the current state of the domain to a file.
void domain_finalize ( void );                  // Function to finalize the domain and free allocated memory.


void
swap ( real_t** m1, real_t** m2 )
{
    real_t* tmp;
    tmp = *m1;
    *m1 = *m2;
    *m2 = tmp;
}

int main ( int argc, char **argv )
{
    // Initialize MPI environment and get the total number of processes and the rank of the current process.
    MPI_Init (&argc, &argv);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    // Create a 2D Cartesian grid of processes and determine the dimensions of the grid.
    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodicity, 0, &cartesian_communicator);

    // Determine the coordinates of the current process in the Cartesian grid.
    MPI_Cart_coords(cartesian_communicator, rank, 2, coord);

    // Determine the rank of the current process in the Cartesian communicator.
    MPI_Cart_rank(cartesian_communicator, coord, &cart_rank);

    // If the current process is the root (rank 0), parse the command-line arguments.
    if(rank == 0){
        OPTIONS *options = parse_args( argc, argv );
        if ( !options )
        {
            fprintf( stderr, "Argument parsing failed\n" );
            exit(1);
        }

        // Store the parsed arguments in global variables.
        M = options->M;
        N = options->N;
        max_iteration = options->max_iteration;
        snapshot_frequency = options->snapshot_frequency;

        // Print the dimensions of the Cartesian grid.
        printf("PW[%d]/[%d]: PE dims=[%dx%d]\n", rank, size, dims[0], dims[1]);
    }

    // Broadcast the parsed arguments from the root process to all other processes.
    MPI_Bcast(&N, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iteration, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&snapshot_frequency, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    
    // Initialize the domain and allocate memory for the simulation.
    domain_init();

    struct timeval t_start, t_end;
    gettimeofday ( &t_start, NULL );

    // Create custom MPI datatypes for data communication and file saving.
    create_types();

    // Main simulation loop.
    for ( int_t iteration = 0; iteration <= max_iteration; iteration++ )
    {
        // Exchange border data with neighboring processes.
        border_exchange();

        // Apply boundary conditions to the domain.
        boundary_condition();

        // Perform a single time step of the simulation.
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

    // Finalize the MPI environment.
    MPI_Finalize();

    exit ( EXIT_SUCCESS );
}



void time_step ( void )
{
   
    real_t c, t, b, l, r, K, new_value;

    // Loop over the rows of the subgrid assigned to the current process.
    for ( int_t y = 1; y <= nrows; y++ )
    {
        // Loop over the columns of the subgrid assigned to the current process.
        for ( int_t x = 1; x <= ncols; x++ )
        {
            c = T(x, y);

            t = T(x - 1, y);  // Top neighbor
            b = T(x + 1, y);  // Bottom neighbor
            l = T(x, y - 1);  // Left neighbor
            r = T(x, y + 1);  // Right neighbor

            K = THERMAL_DIFFUSIVITY(x, y);

            new_value = c + K * dt * ((l - 2 * c + r) + (b - 2 * c + t));

            T_next(x, y) = new_value;
        }
    }
}


void border_exchange(void)
{
    // Exchange values between the upper and lower borders:
    
    // Send values from the upper border of the current process to the lower border of the upper process
    // and simultaneously receive values from the lower border of the lower process to the lower ghost cells of the current process
    MPI_Sendrecv(
        &T(1, 1), 1, row_datatype, upper, 0,
        &T(1, nrows + 1), 1, row_datatype, lower, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Send values from the lower border of the current process to the upper border of the lower process
    // and simultaneously receive values from the upper border of the upper process to the upper ghost cells of the current process
    MPI_Sendrecv(
        &T(1, nrows), 1, row_datatype, lower, 0,
        &T(1, 0), 1, row_datatype, upper, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Exchange values between the left and right borders:

    // Send values from the left border of the current process to the right border of the left process
    // and simultaneously receive values from the right border of the right process to the right ghost cells of the current process
    MPI_Sendrecv(
        &T(1, 1), 1, column_datatype, left, 0,
        &T(ncols + 1, 1), 1, column_datatype, right, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );

    // Send values from the right border of the current process to the left border of the right process
    // and simultaneously receive values from the left border of the left process to the left ghost cells of the current process
    MPI_Sendrecv(
        &T(ncols, 1), 1, column_datatype, right, 0,
        &T(0, 1), 1, column_datatype, left, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE
    );
}


// Function to apply the upper boundary condition
void upper_boundary_condition(void)
{
    // Iterate over all columns
    for (int_t x = 1; x <= ncols; x++)
    {
        // Set the temperature value of the upper boundary (T at row 0)
        // to be the same as the value two rows below it (T at row 2)
        T(x, 0) = T(x, 2);
    }
}

// Function to apply the lower boundary condition
void lower_boundary_condition(void)
{
    // Iterate over all columns
    for (int_t x = 1; x <= ncols; x++)
    {
        // Set the temperature value of the lower boundary (T at the last row, nrows + 1)
        // to be the same as the value two rows above it (T at nrows - 1)
        T(x, nrows + 1) = T(x, nrows - 1);
    }
}

// Function to apply the left boundary condition
void left_boundary_condition(void)
{
    // Iterate over all rows
    for (int_t y = 1; y <= nrows; y++)
    {
        // Set the temperature value of the left boundary (T at column 0)
        // to be the same as the value two columns to its right (T at column 2)
        T(0, y) = T(2, y);
    }
}

// Function to apply the right boundary condition
void right_boundary_condition(void)
{
    // Iterate over all rows
    for (int_t y = 1; y <= nrows; y++)
    {
        // Set the temperature value of the right boundary (T at the last column, ncols + 1)
        // to be the same as the value two columns to its left (T at ncols - 1)
        T(ncols + 1, y) = T(ncols - 1, y);
    }
}

// boundary conditions using the functions before, why? It's more elegant.
void boundary_condition( void )
{
    // Implement boundary conditions here
    
    // Lower border, just taking the function created before and using it in the context
    if (coord[1] == dims[1]-1){
        lower_boundary_condition();
    }

    // Upper border, just taking the function created before and using it in the context
    if (coord[1] == 0 ){
        upper_boundary_condition();
    }

    // Left border, just taking the function created before and using it in the context
    if (coord[0] == 0){
        left_boundary_condition();
    }

    // Right border, just taking the function created before and using it in the context
    if (coord[0] == dims[0]-1)
    {
        right_boundary_condition();
    }

}


void
domain_init ( void )
{

    // Calculate number of rows and columns for each subprocess
    ncols = M/dims[0];
    nrows = N/dims[1];

    // Calculate the offset for each subprocess
    int offset_x = ncols * coord[0];
    int offset_y = nrows * coord[1];

    // Get neighbours, 1 is the displacement.
    MPI_Cart_shift(cartesian_communicator, Y, 1, &upper, &lower);
    MPI_Cart_shift(cartesian_communicator, X, 1, &left, &right);
    
    // Allocate memory for each subprocess, here I changed the nrows, ncols to adapt it to my own solution.
    temp[0] = malloc ( (nrows+2) * (ncols+2) * sizeof(real_t) );
    temp[1] = malloc ( (nrows+2) * (ncols+2) * sizeof(real_t) );
    thermal_diffusivity = malloc ( (nrows+2) * (ncols+2) * sizeof(real_t) );

    dt = 0.1;

    for ( int_t y = 1; y <= nrows; y++ )
    {
        for ( int_t x = 1; x <= ncols; x++ )
        {
            real_t temperature = 30 + 30 * sin((x + offset_x + y + offset_y) / 20.0);
            real_t diffusivity = 0.05 + (30 + 30 * sin((N - (x + offset_x) + (y + offset_y)) / 20.0)) / 605.0;

            T(x,y) = temperature;
            T_next(x,y) = temperature;
            THERMAL_DIFFUSIVITY(x,y) = diffusivity;
        }
    }
}

void create_types(void)
{
    
    // A row consists of 'ncols' contiguous doubles, excluding the left and right boundaries.
    MPI_Type_contiguous(ncols, MPI_DOUBLE, &row_datatype); // Create a contiguous datatype for the row.
    MPI_Type_commit(&row_datatype); // Commit the created row datatype.

    // A column consists of 'nrows' doubles, excluding the upper and lower boundaries.
    // The distance between each double in the column is the horizontal length of the subgrid plus 2 (for the boundaries).
    MPI_Type_vector(nrows, 1, ncols + 2, MPI_DOUBLE, &column_datatype); // Create a vector datatype for the column.
    MPI_Type_commit(&column_datatype); // Commit the created column datatype.
}


void create_subgrid_and_grid_datatypes(void)
{
    // A subgrid is a subarray of the actually allocated memory
    // To create it, the boundaries must be excluded.
    int size_array[2] = {nrows + 2, ncols + 2};
    int subgrid_size_array[2] = {nrows, ncols};
    int start_point[2] = {1, 1};
    MPI_Type_create_subarray(
        2,
        size_array,
        subgrid_size_array,
        start_point,
        MPI_ORDER_C,
        MPI_DOUBLE,
        &subgrid_datatype
    );
    MPI_Type_commit(&subgrid_datatype);

    // The grid is the position of this process's subgrid to print in the output file.
    int file_size_array[2] = {N, M};
    int file_origin[2] = {coord[1] * nrows, coord[0] * ncols};
    MPI_Type_create_subarray(
        2,
        file_size_array,
        subgrid_size_array,
        file_origin,
        MPI_ORDER_C,
        MPI_DOUBLE,
        &grid_datatype
    );
    MPI_Type_commit(&grid_datatype);
}


// Function to save domain data at specified iterations
void domain_save(int_t iteration)
{
    int_t index = iteration / snapshot_frequency;
    int offset_x = ncols * coord[0];
    int offset_y = nrows * coord[1];
    
    char filename[256];
    memset(filename, 0, 256 * sizeof(char));
    sprintf(filename, "data/%.5ld.bin", index);

    // Create the necessary MPI datatypes for subgrid and grid
    create_subgrid_and_grid_datatypes();
    
    // Declare an MPI file handle
    MPI_File out;
    
    // Open the file in write mode with the specified filename
    MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &out);
    
    // Set the file view to write data of type MPI_DOUBLE using the grid datatype
    MPI_File_set_view(out, 0, MPI_DOUBLE, grid_datatype, "native", MPI_INFO_NULL);
    
    // Write the data to the file using the subgrid datatype
    MPI_File_write_all(out, &T(0, 0), 1, subgrid_datatype, MPI_STATUS_IGNORE);
    
    // Close the MPI file handle
    MPI_File_close(&out);
}

void
domain_finalize ( void )
{
    free ( temp[0] );
    free ( temp[1] );
    free ( thermal_diffusivity );
}
