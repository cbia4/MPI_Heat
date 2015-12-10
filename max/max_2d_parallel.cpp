#include <iostream>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

using namespace std;

double x(double local_a, int i);
double y(double local_c, int i);
double exact_solution(double x, double y);
double S(double x, double y);
double heat(double right, double left, double top, double bottom, double x, double y);
double sqr(double i);
bool has_left_neighbor(int rank, int num_proc);
bool has_right_neighbor(int rank, int num_proc);
bool has_top_neighbor(int rank, int num_proc);
bool has_bottom_neighbor(int rank, int num_proc);

const double a = 0.0; const double b = 1.0;
const double c = 0.0; const double d = 1.0;

const int n = 64;

const double dx = (b - a) / (n - 1);
const double dy = (d - c) / (n - 1);

const int MASTER_RANK = 0;
const int TAG = 0;
const int SINGLE = 1;

const double tolerance = 1E-15;
const int max_iterations = 1000000;

const bool DEBUG = false;

int main() {
    double iteration_error = 1.0;

    /* INITIALIZE PARALLEL VARIABLES */
    int num_proc = 0;
    int rank = 0;
    int left_proc = MPI_PROC_NULL;
    int right_proc = MPI_PROC_NULL;
    int top_proc = MPI_PROC_NULL;
    int bottom_proc = MPI_PROC_NULL;


    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

    MPI_Request send_requests[4];
    MPI_Request receive_requests[4];

    const int local_n = n / num_proc;

    double local_Un[local_n][local_n];
    double local_Unp1[local_n][local_n];

    double top_ghost_val[local_n];
    double bottom_ghost_val[local_n];
    double left_ghost_val[local_n];
    double right_ghost_val[local_n];

    const double local_a = a + (rank % num_proc) * (local_n / num_proc) * dx;
    const double local_b = local_a + local_n * dx;
    const double local_c = c + (rank % num_proc) * (local_n / num_proc) * dy;
    const double local_d = local_c + local_n * dy;

    const double start_time = MPI_Wtime();
    /* DONE INITIALIZING PARALLEL VARIABLES */





    // INITIALIZE INTERIOR POINTS
    for (int i = 1; i < local_n - 1; i++) {
        for (int j = 1; j < local_n - 1; j++) {
            local_Un[i][j] = 0.0;
        }
    }






    // INITIALIZE *TOP* NEIGHBOR AND OUTER-POINTS
    if (has_top_neighbor(rank, num_proc)) {
        top_proc = rank - sqrt(num_proc);
    } else {
        top_proc = MPI_PROC_NULL;
        for (int i = 0; i < local_n; i++) {
            local_Un[i][0] = exact_solution(x(local_a, i), c);
        }
    }

    // INITIALIZE *BOTTOM* NEIGHBOR AND OUTER-POINTS
    if (has_bottom_neighbor(rank, num_proc)) {
        bottom_proc = rank - sqrt(num_proc);
    } else {
        bottom_proc = MPI_PROC_NULL;
        for (int i = 0; i < local_n; i++) {
            local_Un[i][local_n - 1] = exact_solution(x(local_a, i), d);
        }
    }

    // INITIALIZE *LEFT* NEIGHBOR AND OUTER-POINTS
    if (has_left_neighbor(rank, num_proc)) {
        left_proc = rank - 1;
    } else {
        left_proc = MPI_PROC_NULL;
        for (int i = 0; i < local_n; i++) {
            local_Un[0][i] = exact_solution(a, y(local_c, i));
        }
    }

    // INITIALIZE *RIGHT* NEIGHBOR AND OUTER-POINTS
    if (has_right_neighbor(rank, num_proc)) {
        right_proc = rank + 1;
    } else {
        right_proc = MPI_PROC_NULL;
        for (int i = 0; i < local_n; i++) {
            local_Un[local_n - 1][i] = exact_solution(b, y(local_c, i));
        }
    }









    int iteration_count = 0;

    while (iteration_error > tolerance && iteration_count < max_iterations) {

        if (DEBUG) cout << "(" << rank << ") Iteration #" << iteration_count << endl;

        int request_counter = 0;

        // if (DEBUG) cout << "(" << rank << ") Sending ghost points" << endl;

        // SEND *TOP* GHOST POINTS
        if (top_proc != MPI_PROC_NULL) {
            double top_neighbors[local_n];
            for (int i = 0; i < local_n; i++) {
                top_neighbors[i] = local_Un[i][0];
            }
            if (DEBUG) cout << "(" << rank << "):Top Sending " << top_neighbors << " to top:" << top_proc << endl;
            MPI_Isend(&top_neighbors, local_n, MPI_DOUBLE, top_proc, TAG, MPI_COMM_WORLD, &send_requests[0]);
            request_counter++;
        }

        // SEND *BOTTOM* GHOST POINTS
        if (bottom_proc != MPI_PROC_NULL) {
            double bottom_neighbors[local_n];
            for (int i = 0; i < local_n; i++) {
                bottom_neighbors[i] = local_Un[i][local_n - 1];
            }
            if (DEBUG) cout << "(" << rank << "):Bottom Sending " << bottom_neighbors << " to bottom:" << bottom_proc << endl;
            MPI_Isend(&bottom_neighbors, local_n, MPI_DOUBLE, bottom_proc, TAG, MPI_COMM_WORLD, &send_requests[1]);
            request_counter++;
        }

        // SEND *LEFT* GHOST POINTS
        if (left_proc != MPI_PROC_NULL) {
            if (DEBUG) cout << "(" << rank << "):Left Sending " << local_Un[0] << " to left:" << left_proc << endl;
            MPI_Isend(&local_Un[0], local_n, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &send_requests[2]);
            request_counter++;
        }

        // SEND *RIGHT* GHOST POINTS
        if (right_proc != MPI_PROC_NULL) {
            if (DEBUG) cout << "(" << rank << "):Right Sending " << local_Un[local_n - 1] << " to right:" << right_proc << endl;
            MPI_Isend(&local_Un[local_n - 1], local_n, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &send_requests[3]);
            request_counter++;
        }







        if (DEBUG) cout << "(" << rank << ") Calculating interior points" << endl;

        // CALCULATE INTERIOR POINTS
        for (int i = 1; i < local_n - 1; i++) {
            for (int j = 1; j < local_n - 1; j++) {
                double right = local_Un[i + 1][j];
                double left = local_Un[i - 1][j];
                double top = local_Un[i][j + 1];
                double bottom = local_Un[i][j - 1];

                local_Unp1[i][j] = heat(right, left, top, bottom, x(local_a, i), y(local_c, j));
            }
        }





        if (DEBUG) cout << "(" << rank << ") Receiving ghost points" << endl;

        // RECEIVE *TOP* GHOST POINTS
        if (top_proc != MPI_PROC_NULL) {
            MPI_Irecv(&top_ghost_val, local_n, MPI_DOUBLE, top_proc, TAG, MPI_COMM_WORLD, &receive_requests[0]);
            request_counter++;
        }

        // RECEIVE *BOTTOM* GHOST POINTS
        if (bottom_proc != MPI_PROC_NULL) {
            MPI_Irecv(&bottom_ghost_val, local_n, MPI_DOUBLE, bottom_proc, TAG, MPI_COMM_WORLD, &receive_requests[1]);
            request_counter++;
        }

        // RECEIVE *LEFT* GHOST POINTS
        if (left_proc != MPI_PROC_NULL) {
            MPI_Irecv(&left_ghost_val, local_n, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &receive_requests[2]);
            request_counter++;
        }

        // RECEIVE *RIGHT* GHOST POINTS
        if (right_proc != MPI_PROC_NULL) {
            MPI_Irecv(&right_ghost_val, local_n, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &receive_requests[3]);
            request_counter++;
        }





        // WAIT ON ALL REQUESTS
        if (top_proc != MPI_PROC_NULL) {
            if (DEBUG) cout << "(" << rank << ") Waiting on top" << endl;
            MPI_Wait(&send_requests[0], MPI_STATUS_IGNORE);
            MPI_Wait(&receive_requests[0], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "(" << rank << ") Top done waiting" << endl;
        }
        if (bottom_proc != MPI_PROC_NULL) {
            if (DEBUG) cout << "(" << rank << ") Waiting on bottom" << endl;
            MPI_Wait(&send_requests[1], MPI_STATUS_IGNORE);
            MPI_Wait(&receive_requests[1], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "(" << rank << ") Bottom done waiting" << endl;
        }
        if (left_proc != MPI_PROC_NULL) {
            if (DEBUG) cout << "(" << rank << ") Waiting on left" << endl;
            MPI_Wait(&send_requests[2], MPI_STATUS_IGNORE);
            MPI_Wait(&receive_requests[2], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "(" << rank << ") Left done waiting" << endl;
        }
        if (right_proc != MPI_PROC_NULL) {
            if (DEBUG) cout << "(" << rank << ") Waiting on right" << endl;
            MPI_Wait(&send_requests[3], MPI_STATUS_IGNORE);
            MPI_Wait(&receive_requests[3], MPI_STATUS_IGNORE);
            if (DEBUG) cout << "(" << rank << ") Right done waiting" << endl;
        }






        if (DEBUG) cout << "(" << rank << ") Calculating ghost points" << endl;
        if (top_proc != MPI_PROC_NULL) {
            int j = 0;
            for (int i = 0; i < local_n; i++) {
                double right = local_Un[i + 1][j];
                double left = local_Un[i - 1][j];
                double top = top_ghost_val[i];
                double bottom = local_Un[i][j - 1];

                local_Unp1[i][0] = heat(right, left, top, bottom, x(local_a, i), local_c);
            }
        } else {
            for (int i = 0; i < local_n; i++) {
                local_Unp1[i][0] = exact_solution(x(local_a, i), c);
            }
        }

        if (bottom_proc != MPI_PROC_NULL) {
            int j = local_n - 1;
            for (int i = 0; i < local_n; i++) {
                double right = local_Un[i + 1][j];
                double left = local_Un[i - 1][j];
                double top = local_Un[i][j + 1];
                double bottom = bottom_ghost_val[i];

                local_Unp1[i][local_n - 1] = heat(right, left, top, bottom, x(local_a, i), local_d);
            }
        } else {
            for (int i = 0; i < local_n; i++) {
                local_Unp1[i][local_n - 1] = exact_solution(x(local_a, i), d);
            }
        }
        if (left_proc != MPI_PROC_NULL) {
            int j = 0;
            for (int i = 0; i < local_n; i++) {
                double right = local_Un[i + 1][j];
                double left = left_ghost_val[i];
                double top = local_Un[i][j + 1];
                double bottom = local_Un[i][j - 1];

                local_Unp1[0][i] = heat(right, left, top, bottom, local_a, y(local_c, i));
            }
        } else {
            for (int i = 0; i < local_n; i++) {
                local_Unp1[0][i] = exact_solution(a, y(local_c, i));
            }
        }
        if (right_proc != MPI_PROC_NULL) {
            int j = local_n - 1;
            for (int i = 0; i < local_n; i++) {
                double right = right_ghost_val[i];
                double left = local_Un[i - 1][j];
                double top = local_Un[i][j + 1];
                double bottom = local_Un[i][j - 1];

                local_Unp1[local_n - 1][i] = heat(right, left, top, bottom, local_b, y(local_c, i));
            }
        } else {
            for (int i = 0; i < local_n; i++) {
                local_Unp1[local_n - 1][i] = exact_solution(b, y(local_c, i));
            }
        }





        if (DEBUG) cout << "(" << rank << ") Calculating iteration error" << endl;
        // CALCULATE THE ITERATION ERROR
        iteration_error = 0.0;
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < local_n; j++) {
                double local_iteration_error = fabs(local_Unp1[i][j] - local_Un[i][j]);
                if (local_iteration_error > iteration_error) iteration_error = local_iteration_error;
            }
        }

        // SEND / RECEIVE THE ITERATION ERROR
        if (rank != MASTER_RANK) {
            MPI_Send(&iteration_error, SINGLE, MPI_DOUBLE, MASTER_RANK, TAG, MPI_COMM_WORLD);
        } else {
            double local_iteration_error;
            for (int i = 1; i < num_proc; i++) {
                MPI_Recv(&local_iteration_error, SINGLE, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (local_iteration_error > iteration_error) iteration_error = local_iteration_error;
            }
        }

        // BROADCAST THE ITERATION ERROR
        MPI_Bcast(&iteration_error, SINGLE, MPI_DOUBLE, MASTER_RANK, MPI_COMM_WORLD);






        if (DEBUG) cout << "(" << rank << ") Updating grid" << endl << endl;
        // PREPARE FOR NEXT ITERATION
        for (int i = 0; i < local_n; i++) {
            for (int j = 0; j < local_n; j++) {
                local_Un[i][j] = local_Unp1[i][j];
            }
        }




        // CAN'T GO TO NEXT ITERATION UNTIL ALL PROCESSORS ARE DONE
        MPI_Barrier(MPI_COMM_WORLD);

        iteration_count++;
    }








    // CALCULATE MAX ERROR (SOLUTION) BETWEEN COMPUTED AND EXACT SOLUTION
    double solution_error = 0.0;
    for (int i = 0; i < local_n; i++) {
        for (int j = 0; j < local_n; j++) {
            double local_solution_error = fabs(local_Unp1[i][j] - exact_solution(x(local_a, i), y(local_c, j)) );
            if (local_solution_error > solution_error) solution_error = local_solution_error;
        }
    }

    // SEND / RECEIVE THE SOLUTION ERROR
    if (rank != MASTER_RANK) {
        MPI_Send(&solution_error, SINGLE, MPI_DOUBLE, MASTER_RANK, TAG, MPI_COMM_WORLD);
    } else {
        double local_solution_error;
        for (int i = 1; i < num_proc; i++) {
            MPI_Recv(&local_solution_error, SINGLE, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (local_solution_error > solution_error) solution_error = local_solution_error;
        }

        const double elapsed_time = MPI_Wtime() - start_time;

        // OUTPUT
        std::cout                                                               << std::endl << std::endl;
        std::cout << "-------------------------------------------------------"               << std::endl;
        std::cout << "SUMMARY:"                                                 << std::endl << std::endl;
        std::cout << "The error between two iterates is "    << iteration_error << std::endl << std::endl;
        std::cout << "The maximum error in the solution is " << solution_error               << std::endl;
        std::cout << "Elapsted time: "                       << elapsed_time    << std::endl << std::endl;
        std::cout << "-------------------------------------------------------"  << std::endl << std::endl;

    }

    MPI_Finalize();

    return 0;
}

// Helper functions:

double heat(double right, double left, double top, double bottom, double x, double y) {
    return ( sqr(dy) * (right + left - S(x, y) * sqr(dx)) + top * sqr(dx) + bottom * sqr(dx) ) / (2 * (sqr(dx) + sqr(dy)));
}

double x(double local_a, int i) {
    return local_a + i * dx;
}

double y(double local_c, int i) {
    return local_c + i * dy;
}

double exact_solution(double x, double y) {
    return sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

double S(double x, double y) {
    return -8 * sqr(M_PI) * exact_solution(x, y);
}

double sqr(double i) {
    return i * i;
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