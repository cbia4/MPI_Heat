#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

using namespace std;

// Prototypes of Helper functions:
double x(double local_a, int i) ;
double y(double local_c, int i);
double exact_solution(double x, double y);
double S(double x, double y);
double heat(double bottom, double top, double left, double right, double x, double y);
bool has_left_neighbor(int rank, int num_proc);
bool has_right_neighbor(int rank, int num_proc);
bool has_top_neighbor(int rank, int num_proc);
bool has_bottom_neighbor(int rank, int num_proc);

const double a = 0.0; const double b = 1.0;
const double c = 0.0; const double d = 1.0;

const int n = 64; // n x n matrix (square)

const double dx = (b - a) / (n - 1);
const double dy = (d - c) / (n - 1);

const int MASTER_RANK = 0;
const int TAG = 0;
const int NUM_TO_SEND = 1;

const double tolerance = 1E-15;
const int max_iterations = 1000000;

const bool DEBUG = false;

int main() {
    double iteration_error = 1.0;

    /* INITIALIZE PARALLEL VARIABLES */
    int num_proc, rank;
    int left_proc, right_proc, top_proc, bottom_proc;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    const int local_n = n / num_proc;

    double local_Un[local_n][local_n];
    double local_Unp1[local_n][local_n];

    const double local_a = a + (rank % num_proc) * (n / num_proc) * dx;
    const double local_c = c + (rank % num_proc) * (n / num_proc) * dy;
    /* DONE INITIALIZING PARALLEL VARIABLES */

    // INITIALIZE SMALL GRID
    for (int i = 1; i < local_n - 1; i++) {
        for (int j = 1; j < local_n - 1; j++) {
            local_Un[i][j] = 0.0;
        }
    }

    // INITIALIZE LEFT NEIGHBOR
    if (has_left_neighbor(rank, num_proc)) {
        left_proc = rank - 1;
    } else {
        for (int i = 0; i < local_n; i++) {
            local_Un[i][0] = exact_solution(a, y(local_c, i));
        }
    }

    // INITIALIZE RIGHT NEIGHBOR
    if (has_right_neighbor(rank, num_proc)) {
        right_proc = rank + 1;
    } else {
        for (int i = 0; i < local_n; i++) {
            local_Un[i][local_n - 1] = exact_solution(b, y(local_c, i));
        }
    }

    // INITIALIZE TOP NEIGHBOR
    if (has_top_neighbor(rank, num_proc)) {
        top_proc = rank - sqrt(num_proc);
    } else {
        for (int i = 0; i < local_n; i++) {
            local_Un[0][i] = exact_solution(x(local_a, i), c);
        }
    }

    // INITIALIZE BOTTOM NEIGHBOR
    if (has_bottom_neighbor(rank, num_proc)) {
        bottom_proc = rank + sqrt(num_proc);
    } else {
        for (int i = 0; i < local_n; i++) {
            // local_Un[local_n - 1][i] = exact_solution(b, y(local_c, i));
            local_Un[local_n - 1][i] = exact_solution(x(local_a, i), d);
        }
    }

    // Initialize the interation counter:
    int iteration_count = 0;

    MPI_Request send_request[4];
    MPI_Request recv_request[4];

    while (iteration_error > tolerance && iteration_count < max_iterations) {
        if (DEBUG) cout << "iter: " << iteration_count << endl;

        double left_ghost_val[local_n];
        double right_ghost_val[local_n];
        double top_ghost_val[local_n];
        double bottom_ghost_val[local_n];

        if (DEBUG) cout << "send ghosts" << endl;

        // Treat the left and right boundary conditions:
        if (has_left_neighbor(rank, num_proc)) {
            double left_neighbors[local_n];
            for (int i = 0; i < local_n; i++) {
                left_neighbors[i] = local_Un[i][0];
            }
            MPI_Isend(&left_neighbors, local_n, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &send_request[0]);
        } else {
            for (int i = 0; i < local_n; i++) {
                // local_Unp1[i][0] = exact_solution(x(local_a, i), c);
                local_Unp1[i][0] = exact_solution(a, y(local_c, i));
            }
        }

        if (has_right_neighbor(rank, num_proc)) {
            double right_neighbors[local_n];
            for (int i = 0; i < local_n; i++) {
                right_neighbors[i] = local_Un[i][local_n - 1];
            }
            MPI_Isend(&right_neighbors, local_n, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &send_request[1]);
        } else {
            for (int i = 0; i < local_n; i++) {
                local_Unp1[i][local_n - 1] = exact_solution(b, y(local_c, i));
            }
        }

        if (has_top_neighbor(rank, num_proc)) {
            MPI_Isend(&local_Un[0], local_n, MPI_DOUBLE, top_proc, TAG, MPI_COMM_WORLD, &send_request[2]);
        } else {
            for (int i = 0; i < local_n; i++) {
                local_Unp1[0][i] = exact_solution(x(local_a, i), c);
            }
        }
        if (has_bottom_neighbor(rank, num_proc)) {
            MPI_Isend(&local_Un[local_n - 1], local_n, MPI_DOUBLE, bottom_proc, TAG, MPI_COMM_WORLD, &send_request[3]);
        } else {
            for (int i = 0; i < local_n; i++) {
                local_Unp1[local_n - 1][i] = exact_solution(x(local_a, i), d);
            }
        }

        if (DEBUG) cout << "interior" << endl;

        // Treat the interior points using the update formula:
        for (int i = 1; i < local_n - 1; i++) {
            for (int j = 1; j < local_n - 1; j++) {
                local_Unp1[i][j] = heat(local_Un[i + 1][j], local_Un[i - 1][j], local_Un[i][j - 1], local_Un[i][j + 1], x(local_a, i), y(local_c, j));
            }
        }

        if (DEBUG) cout << "receive ghosts" << endl;

        if (has_left_neighbor(rank, num_proc)) {
            MPI_Irecv(&left_ghost_val, local_n, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &recv_request[0]);
        }
        if (has_right_neighbor(rank, num_proc)) {
            MPI_Irecv(&right_ghost_val, local_n, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &recv_request[1]);
        }
        if (has_top_neighbor(rank, num_proc)) {
            MPI_Irecv(&top_ghost_val, local_n, MPI_DOUBLE, top_proc, TAG, MPI_COMM_WORLD, &recv_request[2]);
        }
        if (has_bottom_neighbor(rank, num_proc)) {
            MPI_Irecv(&bottom_ghost_val, local_n, MPI_DOUBLE, bottom_proc, TAG, MPI_COMM_WORLD, &recv_request[3]);
        }

        if (DEBUG) cout << "wait" << endl;

        // Wait for all MPI Requests to complete
        if (has_left_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "wait left start" << endl;
            MPI_Wait(&send_request[0], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request[0], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "wait left end" << endl;
        }
        if (has_right_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "wait right start" << endl;
            MPI_Wait(&send_request[1], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request[1], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "wait right end" << endl;
        }
        if (has_top_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "wait top start" << endl;
            MPI_Wait(&send_request[2], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request[2], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "wait top end" << endl;
        }
        if (has_bottom_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "wait bottom start" << endl;
            MPI_Wait(&send_request[3], MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request[3], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "wait bottom end" << endl;
        }

        if (DEBUG) cout << "update ghosts" << endl;

        if (has_left_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "update left ghost" << endl;
            for (int i = 0; i < local_n; i++) {
                local_Unp1[i][0] = heat(local_Un[i + 1][0], local_Un[i - 1][0], left_ghost_val[i], local_Un[i][1], x(local_a, i), y(local_c, i));
            }
            if (DEBUG) cout << "left done" << endl;
        }

        if (has_right_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "update right ghost" << endl;
            for (int i = 0; i < local_n; i++) {
                local_Unp1[i][local_n - 1] = heat(local_Un[i + 1][local_n - 1], local_Un[i - 1][local_n - 1], local_Un[i][local_n - 2], right_ghost_val[i], x(local_a, i), y(local_c, i));
            }

            if (DEBUG) cout << "right done" << endl;
        }

        if (has_top_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "update top ghost" << endl;
            for (int i = 0; i < local_n; i++) {
                local_Unp1[0][i] = heat(local_Un[1][i], top_ghost_val[i], local_Un[0][i - 1], local_Un[1][i + 1], x(local_a, i), y(local_c, i));
            }
            if (DEBUG) cout << "top done" << endl;
        }
        if (has_bottom_neighbor(rank, num_proc)) {
            if (DEBUG) cout << "update bottom ghost" << endl;
            for (int i = 0; i < local_n; i++) {
                local_Unp1[local_n - 1][i] = heat(bottom_ghost_val[i], local_Un[local_n - 2][i], local_Un[local_n - 1][i - 1], local_Un[local_n - 1][i + 1], x(local_a, i), y(local_c, i));
            }
            if (DEBUG) cout << "bottom done" << endl;
        }


        if (DEBUG) cout << "compute iter error" << endl;

        // Compute the maximum error between 2 iterates to establish whether or not
        // steady-state is reached:
        iteration_error = 0.0;
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < local_n; j++) {
                double local_iteration_error = fabs(local_Unp1[i][j] - local_Un[i][j]);
                if (local_iteration_error > iteration_error) iteration_error = local_iteration_error;
            }
        }

        if (DEBUG) cout << "send / rec iter error" << endl;

        // Send iteration error to the master process
        if (rank != MASTER_RANK) {
            MPI_Send(&iteration_error, NUM_TO_SEND, MPI_DOUBLE, MASTER_RANK, TAG, MPI_COMM_WORLD);
        } else {
            double local_iteration_error;
            for (int i = 1; i < num_proc; i++) {
                MPI_Recv(&local_iteration_error, NUM_TO_SEND, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (local_iteration_error > iteration_error) iteration_error = local_iteration_error;
            }
        }

        if (DEBUG) cout << "broacast iter error" << endl;

        MPI_Bcast(&iteration_error, NUM_TO_SEND, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);

        if (DEBUG) cout << "update major grid" << endl;

        // Prepare for the next iteration:
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < local_n; j++) {
                local_Un[i][j] = local_Unp1[i][j];
            }
        }

        if (DEBUG) cout << "barrier" << endl;

        MPI_Barrier(MPI_COMM_WORLD);

        if (DEBUG) cout << "increment iter count" << endl << endl;

        iteration_count++;
    }

// Compute the maximum error between the computed and exact solutions:
    double solution_error = 0.0;
    for (int i = 0; i < local_n; i++) {
        for (int j = 0; j < local_n; j++) {
            double local_solution_error = fabs(local_Unp1[i][j] - exact_solution(x(local_a, i), y(local_c, j)) );
            if (local_solution_error > solution_error) solution_error = local_solution_error;
        }
    }

// Send solution error to the master process
    if (rank != MASTER_RANK) {
        MPI_Send(&solution_error, NUM_TO_SEND, MPI_DOUBLE, MASTER_RANK, TAG, MPI_COMM_WORLD);
    } else {
        double local_solution_error;
        for (int i = 1; i < num_proc; i++) {
            MPI_Recv(&local_solution_error, NUM_TO_SEND, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

double x(double local_a, int i) {
    return local_a + i * dx;
}

double y(double local_c, int i) {
    return local_c + i * dy;
}

double heat(double bottom, double top, double left, double right, double x, double y) {
    // return ( pow(dy, 2) * (top + bottom - S(x, y) * pow(dx, 2)) + right * pow(dx, 2) + left * pow(dx, 2) ) / (2 * (pow(dx, 2) + pow(dy, 2)));
    return ( pow(dy, 2) * (right + left - S(x, y) * pow(dx, 2)) + top * pow(dx, 2) + bottom * pow(dx, 2) ) / (2 * (pow(dx, 2) + pow(dy, 2)));
}

double exact_solution(double x, double y) {
    return sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

double S(double x, double y) {
    return -8 * pow(M_PI, 2) * exact_solution(x, y);
}

bool has_bottom_neighbor(int rank, int num_proc) {
    return !(rank + sqrt(num_proc) >= num_proc);
}

bool has_top_neighbor(int rank, int num_proc) {
    return !(rank - sqrt(num_proc) < 0);
}

bool has_left_neighbor(int rank, int num_proc) {
    return !(rank % (int) sqrt(num_proc) == 0);
}

bool has_right_neighbor(int rank, int num_proc) {
    return !((rank + 1) % (int) sqrt(num_proc) == 0);
}