#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

const int TRUE = 1;
const int FALSE = 0;

const int MASTER_RANK = 0;
const int TAG = 0;

// Prototypes of helper functions:
double x(double a, int i, double dx);
double exact_solution(double x);
double S(double x);

// Define matrix size
const int m = 16;

int main()
{

	// comm_sz, rank, boundary points
	// initialize local boundary points
	// value of neighbor processes
	int num_proc, my_rank;
	int left_proc, right_proc;
	double a = 0.0, b = 1.0, local_a, local_b;
	double dx = (b - a) / (m - 1);

	// Initialize MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);

	// local points, local_m, local_n should be a whole number
	const int local_m = m / num_proc;
	local_a = a + (my_rank % num_proc) * (m / num_proc) * dx;
	local_b = local_a + (m / num_proc) * dx;

	// local arrays
	double local_Un[local_m];
	double local_Unp1[local_m];
	int left_bdry_condition = 0;
	int right_bdry_condition = 0;

	// index
	int i;

	// Set all points to 0.0 initially
	for (i = 0; i < local_m; i++) {
		local_Un[i] = 0.0;
	}


	// Set Left Neighbor
	if (my_rank % (m / num_proc) == 0) {
		left_proc = MPI_PROC_NULL;
		left_bdry_condition = TRUE;
		local_Un[0] = exact_solution(a);

	} else {
		left_proc = my_rank - 1;
	}

	// Set Right Neighbor
	if ((my_rank + 1) % (m / num_proc) == 0) {
		right_proc = MPI_PROC_NULL;
		right_bdry_condition = TRUE;
		local_Un[local_m - 1] = exact_solution(b);
	} else {
		right_proc = my_rank + 1;
	}

	// Variables to establish the convergence of the Jacobi iterations
	double iteration_error = 1.0;
	double tolerance = 1E-15;
	double max_iterations = 1000000;

	// Initialize the iteration counter
	int iteration_count = 0;
	double left_ghost_val = 0;
	double right_ghost_val = 0;

	MPI_Request send_request[2];
	MPI_Request recv_request[2];
	MPI_Status status;

	while (iteration_error > tolerance && iteration_count < max_iterations) {

		// Update boundary points
		if (left_bdry_condition == TRUE) { // No left neighbor
			local_Unp1[0] = exact_solution(a);
		} else if (right_bdry_condition == TRUE) { // No right neighbor
			local_Unp1[local_m - 1] = exact_solution(b);
		}

		// Send left boundary of small grid
		if (left_proc != MPI_PROC_NULL) {
			double test = 1;
			MPI_Isend(&test, 1, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &send_request[0]);
			printf("Sent %f from process %d to left process %d\n", local_Un[0], my_rank, left_proc);
		}
		// Send right boundary of small grid
		if (right_proc != MPI_PROC_NULL) {
			MPI_Isend(&local_Un[local_m - 1], 1, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &send_request[1]);
			printf("Sent %f from process %d to right process %d\n\n", local_Un[local_m - 1], my_rank, right_proc);
		}

		// Update interior points
		for (i = 1; i < local_m - 1; i++) {
			local_Unp1[i] = .5 * ( local_Un[i + 1] + local_Un[i - 1] - dx * dx * S(x(local_a, i, dx)) );
		}

		// Receive left boundary of small grid
		if (left_proc != MPI_PROC_NULL) {
			MPI_Irecv(&left_ghost_val, 1, MPI_DOUBLE, left_proc, TAG, MPI_COMM_WORLD, &recv_request[0]);
			printf("Received %f from process %d on process %d\n", left_ghost_val, left_proc, my_rank);
		}
		if (right_proc != MPI_PROC_NULL) {
			// Receive right boundary of small grid
			MPI_Irecv(&right_ghost_val, 1, MPI_DOUBLE, right_proc, TAG, MPI_COMM_WORLD, &recv_request[1]);
			printf("Received %f from process %d on process %d\n\n", right_ghost_val, right_proc, my_rank);
		}

		MPI_Wait(&recv_request[0], &status);
		MPI_Wait(&recv_request[1], &status);

		// Update the boundary points
		if (left_bdry_condition == FALSE) {
			local_Unp1[0] = .5 * ( local_Un[1] + left_ghost_val - dx * dx * S(x(local_a, i, dx)));
		}

		if (right_bdry_condition == FALSE) {
			local_Unp1[m - 1] = .5 * (right_ghost_val + local_Un[local_m - 1] - dx * dx * S(x(local_a, i, dx)));
		}

		iteration_error = 0.0;
		for (i = 0; i < local_m; i++) {
			double local_iteration_error = fabs(local_Unp1[i] - local_Un[i]);
			if (local_iteration_error > iteration_error) {
				iteration_error = local_iteration_error;
			}
		}


		if (my_rank != MASTER_RANK) {
			MPI_Send(&iteration_error, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
		} else {
			double recvd_iteration_error;
			for (i = 1; i < num_proc; i++) {
				MPI_Recv(&recvd_iteration_error, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if (recvd_iteration_error > iteration_error) {
					iteration_error = recvd_iteration_error;
				}
			}
		}

		// Broadcast the iteration_error to all processes
		MPI_Bcast(&iteration_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Prepare for the next iteration
		for (i = 0; i < local_m; i++) {
			local_Un[i] = local_Unp1[i];
		}

		iteration_count++;
	}


	double solution_error;
	for (i = 0; i < local_m; i++) {
		double local_solution_error = fabs( local_Unp1[i] - exact_solution(x(local_a, i, dx)) );
		if (local_solution_error > solution_error)
			solution_error = local_solution_error;
	}

	if (my_rank != MASTER_RANK) {
		MPI_Send(&solution_error, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
	} else {
		double recvd_solution_error;
		for (i = 1; i < num_proc; i++) {
			MPI_Recv(&recvd_solution_error, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (recvd_solution_error > solution_error)
				solution_error = recvd_solution_error;
		}

		printf("------------------------------------------------\n");
		printf("SUMMARY:\n");
		printf("The error between two iterates is:    %.20f\n", iteration_error);
		printf("The maximum error in the solution is: %.20f\n", solution_error);
		printf("------------------------------------------------\n");


	}

	MPI_Finalize();

	return 0;
}


double x(double a, int i, double dx) {
	return a + i * dx;
}

double exact_solution(double x) {
	return sin(2 * M_PI * x) + cos(2 * M_PI * x);
}

double S(double x) {
	return -4 * M_PI * M_PI * exact_solution(x);
}

