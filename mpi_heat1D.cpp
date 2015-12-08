#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>



// Prototypes of helper functions:
double x(double a, int i, double dx);
double exact_solution(double x);
double S(double x);

// Define matrix size
const int m = 100;

int main() 
{

	// Initialize p(comm_sz), my_rank
	// Initialize global/local boundary points, dx
	// Initialize Lproc, Rproc
	int p, my_rank;
	int Lproc, Rproc;
	double a = 0.0, b = 1.0, local_a, local_b;
	double dx = (b-a)/(m-1);

	// Initialize MPI
	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	// local matrix points
	const int local_m = m / p;

	// Check whether m and p values are compatible with the program
	double check_value = ((double)m/(double)p);
	if(my_rank == 0) {
		printf("Hello from the master process\n");
		printf("p       = %d\n", p);
		printf("m       = %d\n", m);
		printf("local_m = %.2f\n", check_value);
	}

	if(check_value != (double)local_m) {
		if(my_rank == 0)
			printf("ERROR: (m) must be divisible by (p). \nQuitting.\n");
		
		MPI_Finalize();
		return 0;
	} 

	MPI_Barrier(MPI_COMM_WORLD);


	// local boundaries
	local_a = a + (my_rank % p) * (m/p) * dx;
	local_b = local_a + (m/p) * dx;

	// local arrays
	double local_Un[local_m];
	double local_Unp1[local_m];

	// Boundary conditions (0=inner, 1=outer)
	int Lbdry_condition = 0;
	int Rbdry_condition = 0;

	// index
	int i;

	// Set all points to 0.0 initially
	for(i = 0; i < local_m; i++) {
		local_Un[i] = 0.0;
	}

	// 2-D IMPLEMENTATIOIN
	/*
	// Set Left Neighbor
	if(my_rank % (m/p) == 0) {
		Lproc = p - 1;
		Lbdry_condition = 1;
		local_Un[0] = exact_solution(a);

	} else {
		Lproc = my_rank - 1;
	}


	// Set Right Neighbor
	if((my_rank + 1) % (m/p) == 0) {
		Rproc = 0;
		Rbdry_condition = 1;
		local_Un[local_m-1] = exact_solution(b);
	} else {
		Rproc = my_rank + 1;
	}

	*/

	// 1-D IMPLEMENTATION
	// Set left neighbor
	if(my_rank == 0) {
		Lproc = p - 1;
		Lbdry_condition = 1;
		local_Un[0] = exact_solution(a);
	} else {
		Lproc = my_rank - 1;
	}

	// Set right neighbor
	if(my_rank == (p-1)) {
		Rproc = 0;
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

	// Arrays for MPI_Irecv & MPI_Isend
	MPI_Request reqs[4];
	MPI_Status stats[4];

	while(iteration_error > tolerance && iteration_count < Max_Iter) {
		iteration_count++;

		// Initialize ghost values 
		double Lghost_val;
		double Rghost_val;

		// Send ghost values to Lproc and Rproc
		MPI_Isend(&local_Un[0], 1, MPI_DOUBLE, Lproc, 2, MPI_COMM_WORLD, &reqs[2]);
		MPI_Isend(&local_Un[local_m-1], 1, MPI_DOUBLE, Rproc, 1, MPI_COMM_WORLD, &reqs[3]);

		// Update Interior
		for(i = 1; i < local_m - 1; i++) {
			local_Unp1[i] = .5 * ( local_Un[i+1] + local_Un[i-1] - dx*dx*S(x(local_a,i,dx)) );
		}

		// Receive ghost values from Lproc and Rproc
		MPI_Irecv(&Rghost_val, 1, MPI_DOUBLE, Rproc, 2, MPI_COMM_WORLD, &reqs[0]);
		MPI_Irecv(&Lghost_val, 1, MPI_DOUBLE, Lproc, 1, MPI_COMM_WORLD, &reqs[1]);

		// Wait for all MPI Requests to complete
		MPI_Waitall(4, reqs, stats);
		
		// Update Left Boundary
		if(Lbdry_condition == 1) {
			local_Unp1[0] = exact_solution(a);
		} else {
			local_Unp1[0] = .5 * (local_Un[1] + Lghost_val - dx*dx*S(x(local_a,0,dx)));
		}

		// Update Right Boundary
		if(Rbdry_condition == 1) {
			local_Unp1[local_m-1] = exact_solution(b);
		} else {
			local_Unp1[local_m-1] = .5 * (Rghost_val + local_Un[local_m-2] - dx*dx*S(x(local_a,local_m-1,dx)));
		}

		// Calculate maximum iteration error
		iteration_error = 0.0;
		for(i = 0; i < local_m; i++) {
			double local_iteration_error = fabs(local_Unp1[i] - local_Un[i]);
			if(local_iteration_error > iteration_error)
				iteration_error = local_iteration_error;
		}

		// Send iteration error to the master process
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

		// Update the workers
		MPI_Bcast(&iteration_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Prepare for the next iteration
		for(i = 0; i < local_m; i++) {
			local_Un[i] = local_Unp1[i];
		}

		// Wait for all processes to complete the current iteration
		MPI_Barrier(MPI_COMM_WORLD);

	} // End while loop

	// Calculate maximum error in the solution
	double solution_error = 0.0;
	for(i = 0; i < local_m; i++) {
		double local_solution_error = fabs( local_Unp1[i] - exact_solution(x(local_a,i,dx)) );
		if(local_solution_error > solution_error)
			solution_error = local_solution_error;
	} 

	// Send solution error to the master process
	if(my_rank != 0) {
		MPI_Send(&solution_error, 1, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
	} else {
		double recvd_solution_error;
		for(i = 1; i < p; i++) {
			MPI_Recv(&recvd_solution_error, 1, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(recvd_solution_error > solution_error)
				solution_error = recvd_solution_error;
		}

	    // Output:
	    std::cout                                                              << std::endl << std::endl;
	    std::cout<< "-------------------------------------------------------"               << std::endl;
	    std::cout<< "SUMMARY:"                                                 << std::endl << std::endl;
	    std::cout<< "The error between two iterates is "    << iteration_error << std::endl << std::endl;
	    std::cout<< "The maximum error in the solution is " << solution_error               << std::endl;
	    std::cout<< "-------------------------------------------------------"  << std::endl << std::endl;

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

