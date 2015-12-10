#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

using namespace std;

// Prototypes of Helper functions:
double x(int i);
double exact_solution(double x);
double S(double x);
int has_left_neighbor(int rank, int num_proc, int m);
int has_right_neighbor(int rank, int num_proc, int m);

const double a = 0.0;
const double b = 1.0;
const int m = 5;
const double dx = (b - a) / (m - 1);

const int TRUE = 1;
const int FALSE = 0;

const int MASTER_RANK = 0;
const int TAG = 0;
const int NUM_TO_SEND = 1;

const double tolerance = 1E-15;
const int max_iterations = 1000000;

// We need 2 arrays, local_Un and local_Unp1, for computing the solution:

int main() {
    int i;
    double iteration_error = 1.0;

    /* INITIALIZE PARALLEL VARIABLES */
    int num_proc, rank;
    int left_proc, right_proc;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    const int local_m = m / num_proc;
    double local_Un[local_m];
    double local_Unp1[local_m];

    const double local_a = a + (rank % num_proc) * (m / num_proc) * dx;
    const double local_b = local_a + (m / num_proc) * dx;
    /* DONE INITIALIZING PARALLEL VARIABLES */

    if (rank == MASTER_RANK) {
        printf("Hello from the master process\n");
        printf("p       = %d\n", num_proc);
        printf("m       = %d\n", m);
        printf("local_m = %d\n", local_m);
    }


    // INITIALIZE SMALL GRID
    for (i = 1; i < local_m - 1; i++) {
        local_Un[i] = 0.0;
    }

    // INITIALIZE LEFT NEIGHBOR
    if (has_left_neighbor(rank, num_proc, m) == TRUE) {
        left_proc = rank - 1;
    } else {
        left_proc = MPI_PROC_NULL;
        local_Un[0] = exact_solution(a);
    }

    // INITIALIZE RIGHT NEIGHBOR
    if (has_right_neighbor(rank, num_proc, m) == TRUE) {
        right_proc = rank + 1;
    } else {
        right_proc = MPI_PROC_NULL;
        local_Un[local_m - 1] = exact_solution(b);
    }

    // Initialize the interation counter:
    int iteration_count = 0;

    MPI_Request send_request[2];
    MPI_Request recv_request[2];
    MPI_Status status;

    while (iteration_error > tolerance && iteration_count < max_iterations) {
        double left_ghost_val;
        double right_ghost_val;

        int mpi_counter = 0;
        // Treat the left and right boundary conditions:
        if (has_left_neighbor(rank, num_proc, m) == TRUE) {
            MPI_Isend(&local_Un[0], NUM_TO_SEND, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &send_request[0]);
        } else {
            local_Unp1[0] = exact_solution(a);
        }
        if (has_right_neighbor(rank, num_proc, m) == TRUE) {
            MPI_Isend(&local_Un[local_m - 1], NUM_TO_SEND, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &send_request[1]);
        } else {
            local_Unp1[local_m - 1] = exact_solution(b);
        }

        // Treat the interior points using the update formula:
        for (i = 1; i < local_m - 1; i++) {
            local_Unp1[i] = .5 * ( local_Un[i + 1] + local_Un[i - 1] -  dx * dx * S(x(i)) );
        }

        if (has_left_neighbor(rank, num_proc, m) == TRUE) {
            MPI_Irecv(&left_ghost_val, NUM_TO_SEND, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &recv_request[0]);
        }
        if (has_right_neighbor(rank, num_proc, m) == TRUE) {
            MPI_Irecv(&right_ghost_val, NUM_TO_SEND, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &recv_request[1]);
        }

        // Wait for all MPI Requests to complete
        if (has_left_neighbor(rank, num_proc, m) == TRUE) {
            MPI_Wait(&send_request[0], &status);
            MPI_Wait(&recv_request[0], &status);
            local_Unp1[0] = .5 * (local_Un[1] + left_ghost_val - dx * dx * S(x(0)));
        } 
        if (has_right_neighbor(rank, num_proc, m) == TRUE) {
            MPI_Wait(&send_request[1], &status);
            MPI_Wait(&recv_request[1], &status);
            local_Unp1[local_m - 1] = .5 * (right_ghost_val + local_Un[local_m - 2] - dx * dx * S(x(local_m - 1)));
        }

        // Compute the maximum error between 2 iterates to establish whether or not
        // steady-state is reached:
        iteration_error = 0.0;
        for (i = 0; i < local_m; i++) {
            double local_iteration_error = fabs(local_Unp1[i] - local_Un[i]);
            if (local_iteration_error > iteration_error) iteration_error = local_iteration_error;
        }

        // Send iteration error to the master process
        if (rank != MASTER_RANK) {
            MPI_Send(&iteration_error, NUM_TO_SEND, MPI_DOUBLE, MASTER_RANK, TAG, MPI_COMM_WORLD);
        } else {
            double local_iteration_error;
            for (i = 1; i < num_proc; i++) {
                MPI_Recv(&local_iteration_error, NUM_TO_SEND, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (local_iteration_error > iteration_error) iteration_error = local_iteration_error;
            }
        }

        // Update the workers
        MPI_Bcast(&iteration_error, NUM_TO_SEND, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);


        // Prepare for the next iteration:
        for (i = 0; i < local_m; i++) {
            local_Un[i] = local_Unp1[i];
        }

        MPI_Barrier(MPI_COMM_WORLD);

        iteration_count++;
    }

    // Compute the maximum error between the computed and exact solutions:
    double solution_error = 0.0;
    for (i = 0; i < local_m; i++) {
        double local_solution_error = fabs(local_Unp1[i] - exact_solution(x(i)) );
        if (local_solution_error > solution_error) solution_error = local_solution_error;
    }

    // Send solution error to the master process
    if (rank != MASTER_RANK) {
        MPI_Send(&solution_error, NUM_TO_SEND, MPI_DOUBLE, MASTER_RANK, TAG, MPI_COMM_WORLD);
    } else {
        double local_solution_error;
        for (i = 1; i < num_proc; i++) {
            MPI_Recv(&local_solution_error, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (local_solution_error > solution_error) solution_error = local_solution_error;
        }

        // Output:
        std::cout                                                              << std::endl << std::endl;
        std::cout << "-------------------------------------------------------"               << std::endl;
        std::cout << "SUMMARY:"                                                 << std::endl << std::endl;
        std::cout << "The error between two iterates is "    << iteration_error << std::endl << std::endl;
        std::cout << "The maximum error in the solution is " << solution_error               << std::endl;
        std::cout << "-------------------------------------------------------"  << std::endl << std::endl;

    }


    MPI_Finalize();

    return 0;
}

// Helper functions:

double x(int i) {
    return a + i * dx;
}

double exact_solution(double x) {
    return sin(2 * M_PI * x) + cos(2 * M_PI * x);
}

double S(double x) {
    return -4 * M_PI * M_PI * exact_solution(x);
}

int has_left_neighbor(int rank, int num_proc, int m) {
    // if (rank % num_proc == 0) {
    if (rank == 0) {
        return FALSE;
    }
    return TRUE;
}

int has_right_neighbor(int rank, int num_proc, int m) {
    // if ((rank + 1) % num_proc == 0) {
    if (rank == num_proc - 1) {
        return FALSE;
    }
    return TRUE;
}