#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>



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
	int p, my_rank;
	int Lproc, Rproc;
	double a = 0.0, b = 1.0, local_a, local_b;
	double dx = (b-a)/(m-1);

	// Arrays to load in final values
	double Un[m];
	double Unp1[m];

	// Initialize MPI
	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	// local points, local_m, local_n should be a whole number
	const int local_m = m / p;
	local_a = a + (my_rank % p) * (m/p) * dx;
	local_b = local_a + (m/p) * dx;

	// local arrays
	double local_Un[local_m];
	double local_Unp1[local_m];
	int Lbdry_condition = 0;
	int Rbdry_condition = 0;

	// index
	int i;

	// Set all points to 0.0 initially
	for(i = 0; i < local_m; i++) {
		local_Un[i] = 0.0;
	}


	// Set Left Neighbor
	if(my_rank % (m/p) == 0) {
		Lproc = MPI_PROC_NULL;
		Lbdry_condition = 1;
		local_Un[0] = exact_solution(a);

	} else {
		Lproc = my_rank - 1;
	}

	// Set Right Neighbor
	if((my_rank + 1) % (m/p) == 0) {
		Rproc = MPI_PROC_NULL;
		Rbdry_condition = 1;
		local_Un[local_m-1] = exact_solution(b);
	} else {
		Rproc = my_rank + 1;
	}

	// Variables to establish the convergence of the Jacobi iterations
	double iteration_error = 1.0;
	double tolerance = 1E-15;
	double Max_Iter = 1000000;

	// Initialize the iteration counter
	int iteration_count = 0;
	double Lghost_val = 0;
	double Rghost_val = 0;

	MPI_Request reqs[4];
	MPI_Status stats[2];

	while(iteration_error > tolerance && iteration_count < 10) {
		iteration_count++;

		//printf("iteration count = %d\n", iteration_count);


		if(Lbdry_condition == 1) {
			local_Unp1[0] = exact_solution(a);
		} else if(Rbdry_condition == 1) {
			local_Unp1[local_m-1] = exact_solution(b);
		} 

		MPI_Isend(&local_Un[0], 1, MPI_DOUBLE, Lproc, 0, MPI_COMM_WORLD, &reqs[0]);
		MPI_Isend(&local_Un[local_m - 1], 1, MPI_DOUBLE, Rproc, 1, MPI_COMM_WORLD, &reqs[1]);

		//printf("Messages sent from process %d\n", my_rank);

		// update the interior
		for(i = 1; i < local_m - 1; i++) {
			local_Unp1[i] = .5 * ( local_Un[i+1] + local_Un[i-1] - dx*dx*S(x(local_a,i,dx)) );
		}

		MPI_Irecv(&Lghost_val, 1, MPI_DOUBLE, Lproc, 0, MPI_COMM_WORLD, &reqs[2]);
		MPI_Irecv(&Rghost_val, 1, MPI_DOUBLE, Rproc, 1, MPI_COMM_WORLD, &reqs[3]);

		//printf("Messages received by process %d\n", my_rank);



		// Update the boundary points
		if(Lbdry_condition == 0)
			local_Unp1[0] = .5 * ( local_Un[1] + Lghost_val - dx*dx*S(x(local_a,i,dx)));
		
		if(Rbdry_condition == 0)
			local_Unp1[m-1] = .5 * (Rghost_val + local_Un[local_m - 1] - dx*dx*S(x(local_a,i,dx)));

		//MPI_Waitall(4, reqs, stats);

		//printf("Bottom of the loop!\n");

		iteration_error = 0.0;
		for(i = 0; i < local_m; i++) {
			double local_iteration_error = fabs(local_Unp1[i] - local_Un[i]);
			if(local_iteration_error > iteration_error)
				iteration_error = local_iteration_error;
		}


		if(my_rank != 0) {
			MPI_Send(&iteration_error, 1, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
		} else {
			double recvd_iteration_error;
			for(i = 1; i < p; i++) {
				MPI_Recv(&recvd_iteration_error, 1, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if(recvd_iteration_error > iteration_error)
					iteration_error = recvd_iteration_error;
			}
		}

		// Broadcast the iteration_error to all processes
		MPI_Bcast(&iteration_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Prepare for the next iteration
		for(i = 0; i < local_m; i++) {
			local_Un[i] = local_Unp1[i];
		}


	}


	double solution_error;
	for(i = 0; i < local_m; i++) {
		double local_solution_error = fabs( local_Unp1[i] - exact_solution(x(local_a,i,dx)) );
		if(local_solution_error > solution_error)
			solution_error = local_solution_error;
	} 

	if(my_rank != 0) {
		MPI_Send(&solution_error, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
	} else {
		double recvd_solution_error;
		for(i = 1; i < p; i++) {
			MPI_Recv(&recvd_solution_error, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(recvd_solution_error > solution_error)
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
	return a+i*dx;
}

double exact_solution(double x) {
	return sin(2*M_PI*x) + cos(2*M_PI*x);
}

double S(double x) {
	return -4 * M_PI * M_PI * exact_solution(x);
}

