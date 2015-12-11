#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Prototypes and helper functions:
double x(int i);
double y(int j);
double exact_solution(double x, double y);
double S(double x, double y);
double update_formula(double left, double right, double top, double bottom, double S);

// Global boundaries:
const double a = 0.0;
const double b = 1.0;
const double c = 0.0;
const double d = 1.0;

// Global m points:
const int m = 144;

// Global n points:
const int n = 144;

// Step sizes:
const double dx = (b-a)/(m-1);
const double dy = (d-c)/(n-1);

// Main function:
int main(int argc, char **argv) {


	// Initialize MPI:
	int comm_sz, my_rank;
	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

	double start_time = 0;
	double end_time = 0;
	double elapsed_time = 0;

	if(my_rank == 0) {
		std::cout << "Running mpi_heat:" << std::endl;
		std::cout << "M = " << m << std::endl;
		std::cout << "N = " << n << std::endl;
	}

	int p = sqrt(comm_sz);
	
	// Initialize sub-grid variables:
	const int loc_m = m/p;
	const int loc_n = n/p;

	// Initialize local arrays:
	double Un[loc_m][loc_n];
	double Unp1[loc_m][loc_n];

	// Boundary variables 
	bool ol_boundary = false;
	bool or_boundary = false;
	bool ob_boundary = false;
	bool ot_boundary = false;

	// Index
	int i;
	int j;

	int loc_i = floor(my_rank/p) * loc_n;
	int loc_j = (my_rank%p) * loc_m;

	// Neighbor process values:
	int Lproc, Rproc, Tproc, Bproc;

	double iteration_error = 1;
	double tolerance = 1E-15;
	double Max_Iter = 1000000;

	// Initialize Un to 0:
	for(i = 0; i < loc_m; i++) {
		for(j = 0; j < loc_n; j++) {
			Un[i][j] = 0.0;
		}
	}

	// Impose LEFT boundary conditions:
	if(my_rank % p == 0) {
		Lproc = my_rank + p - 1;
		ol_boundary = true;
		for(i = 0; i < loc_m; i++) 
			Un[i][0] = exact_solution(x(i+loc_i),c);
	} else {
		Lproc = my_rank - 1;
	}

 	// Impose TOP boundary conditions:
 	if(my_rank - p < 0) {
 		Tproc = my_rank + comm_sz - p;
 		ot_boundary = true;	
		for(j = 0; j < loc_m; j++) 
			Un[0][j] = exact_solution(a,y(j+loc_j));
	} else {
		Tproc = my_rank - p;
	}

	// Impose RIGHT boundary conditions:
	if((my_rank+1) % p == 0) {
		Rproc = my_rank - p + 1;
		or_boundary = true;
		for(i = 0; i < loc_m; i++) 
			Un[i][loc_n-1] = exact_solution(x(i+loc_i),d);
	} else {
		Rproc = my_rank + 1;
	}

	// Impose BOTTOM boundary conditions
	if(my_rank + p >= comm_sz) {
		Bproc = my_rank % p;
		ob_boundary = true;
		for(j = 0; j < loc_m; j++) 
			Un[loc_m-1][j] = exact_solution(b,y(j+loc_j));
	} else {
		Bproc = my_rank + p;
	}


	// Arrays for MPI_Isend and MPI_Irecv
	MPI_Request reqs[8];
	MPI_Status stats[8];

	// Initialize send arrays
	double *Lsend = (double*) malloc(loc_n*sizeof(double));
	double *Rsend = (double*) malloc(loc_n*sizeof(double));
	double *Tsend = (double*) malloc (loc_m*sizeof(double));
	double *Bsend = (double*) malloc (loc_m*sizeof(double));

	// Initialize ghost arrays
	double *Lghost = (double*) malloc(loc_n*sizeof(double));
	double *Rghost = (double*) malloc(loc_n*sizeof(double));
	double *Tghost = (double*) malloc(loc_m*sizeof(double));
	double *Bghost = (double*) malloc(loc_m*sizeof(double));

	// Start timer
	if(my_rank == 0) {
		start_time = MPI_Wtime();
	}

	int iteration_count = 0;
	while(iteration_error > tolerance && iteration_count < Max_Iter) {
		iteration_count++;

		// Set arrays to send
		for(i = 0; i < loc_m; i++) {
			Lsend[i] = Un[i][0];
			Rsend[i] = Un[i][loc_n-1];
			Tsend[i] = Un[0][i];
			Bsend[i] = Un[loc_m-1][i];
		}

		// Send boundary arrays to neighbors
		MPI_Isend(Rsend, loc_n, MPI_DOUBLE, Rproc, 0, MPI_COMM_WORLD, &reqs[4]);
		MPI_Isend(Lsend, loc_n, MPI_DOUBLE, Lproc, 1, MPI_COMM_WORLD, &reqs[5]);
		MPI_Isend(Tsend, loc_m, MPI_DOUBLE, Tproc, 2, MPI_COMM_WORLD, &reqs[6]);
		MPI_Isend(Bsend, loc_m, MPI_DOUBLE, Bproc, 3, MPI_COMM_WORLD, &reqs[7]);


		// Treat inner points using the update formula:
		for(i = 1; i < loc_m-1; i++) {
			for(j = 1; j < loc_n-1; j++) {
				Unp1[i][j] = update_formula(Un[i+1][j], Un[i-1][j], Un[i][j+1], Un[i][j-1], S(x(i+loc_i),y(j+loc_j)));
			}
		}

		MPI_Irecv(Lghost, loc_n, MPI_DOUBLE, Lproc, 0, MPI_COMM_WORLD, &reqs[0]);
		MPI_Irecv(Rghost, loc_n, MPI_DOUBLE, Rproc, 1, MPI_COMM_WORLD, &reqs[1]);
		MPI_Irecv(Bghost, loc_m, MPI_DOUBLE, Bproc, 2, MPI_COMM_WORLD, &reqs[2]);
		MPI_Irecv(Tghost, loc_m, MPI_DOUBLE, Tproc, 3, MPI_COMM_WORLD, &reqs[3]);

		// Wait for all MPI requests to complete
		MPI_Waitall(8,reqs,stats);

		// Update TOP LEFT corner
		Unp1[0][0] = update_formula(Un[1][0], Tghost[0], Un[0][1], Lghost[0], S(x(loc_i), y(loc_j)));

		// Update TOP RIGHT corner
		Unp1[0][loc_n-1] = update_formula(Un[1][loc_n-1], Tghost[loc_m-1], Rghost[0], Un[0][loc_n-2], S(x(loc_i),y(loc_j+loc_n-1)));

		// Update BOTTOM LEFT corner
		Unp1[loc_m-1][0] = update_formula(Bghost[0], Un[loc_m-2][0], Un[loc_m-1][1], Lghost[loc_n-1], S(x(loc_i+loc_m-1),y(loc_j)));

		// Update BOTTOM RIGHT corner
		Unp1[loc_m-1][loc_n-1] = update_formula(Bghost[loc_m-1], Un[loc_m-2][loc_n-1], Rghost[loc_n-1], Un[loc_m-1][loc_n-2], S(x(loc_i+loc_m-1),y(loc_j+loc_n-1)));

		// Treat LEFT boundary:
		if(ol_boundary) {
			for(i = 0; i < loc_m; i++) 
				Unp1[i][0] = exact_solution(x(i+loc_i),c);
		} else {
			for(i = 1; i < loc_m-1; i++) 
				Unp1[i][0] = update_formula(Un[i+1][0], Un[i-1][0], Un[i][1], Lghost[i], S(x(i+loc_i),y(loc_j)));
		}

		// Treat TOP boundary:
		if(ot_boundary) {
			for(j = 0; j < loc_m; j++) 
				Unp1[0][j] = exact_solution(a,y(j+loc_j));
		} else {
			for(j = 1; j < loc_n-1; j++) 
				Unp1[0][j] = update_formula(Un[1][j], Tghost[j], Un[0][j+1], Un[0][j-1], S(x(loc_i),y(j+loc_j)));
		}
		
		// Treat RIGHT boundary:
		if(or_boundary) {
			for(i = 0; i < loc_m; i++) 
				Unp1[i][loc_n-1] = exact_solution(x(i+loc_i),d);
		} else {
			for(i = 1; i < loc_m-1; i++) 
				Unp1[i][loc_n-1] = update_formula(Un[i+1][loc_n-1], Un[i-1][loc_n-1], Rghost[i], Un[i][loc_n-2], S(x(i+loc_i),y(loc_j+loc_n-1)));
		}

		// Treat BOTTOM boundary:
		if(ob_boundary) {
			for(j = 0; j < loc_m; j++) 
				Unp1[loc_m-1][j] = exact_solution(b,y(j + loc_j));
		} else {
			for(j = 1; j < loc_m-1; j++) {
				Unp1[loc_m-1][j] = update_formula(Bghost[j], Un[loc_m-2][j], Un[loc_m-1][j+1], Un[loc_m-1][j-1], S(x(loc_i+loc_m-1),y(j+loc_j)));
			}
		}

		// Compute the maximum error between 2 iterates to establish whether
		// or not steady state is reached:
		iteration_error = 0.0;
		for(i = 0; i < loc_m; i++) {
			for(j = 0; j < loc_n; j++) {
				double local_iteration_error = fabs(Unp1[i][j] - Un[i][j]);
				if(local_iteration_error > iteration_error)
					iteration_error = local_iteration_error;
			}
		}

		// Send iteration error to master process
		if(my_rank != 0) {
			MPI_Send(&iteration_error, 1, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
		} else {
			double recvd_iteration_error;
			for(i = 1; i < comm_sz; i++) {
				MPI_Recv(&recvd_iteration_error, 1, MPI_DOUBLE, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				if(recvd_iteration_error > iteration_error) iteration_error = recvd_iteration_error;
			}

			// This was used to check the progress of the program
			// if(iteration_count % 1000 == 0) {
			// 	printf("iteration_error=%.15f\n",iteration_error);
			// }
		}

		// Update the workers
		MPI_Bcast(&iteration_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


		// Prepare for the next iteration:
		for(i = 0; i < loc_m; i++) {
			for(j = 0; j < loc_n; j++) {
				Un[i][j] = Unp1[i][j];
			}
		}
	
	} // END WHILE LOOP

	// Record time
	if(my_rank == 0) {
		end_time = MPI_Wtime();
		elapsed_time = end_time - start_time;
	}
	// Compute the maximum error between the computed and exact solutions:
	double solution_error = 0.0;
	for(i = 0; i < loc_m; i++) {
		for(j = 0; j < loc_n; j++) {
			double local_solution_error = fabs(Unp1[i][j] - exact_solution(x(i+loc_i), y(j+loc_j)));
			if(local_solution_error > solution_error) 
				solution_error = local_solution_error;
		}
	}

	// Send solution error to the master process and output results
	if(my_rank != 0) {
		MPI_Send(&solution_error, 1, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD);
	} else {
		double recvd_solution_error;
		for(i = 1; i < comm_sz; i++) {
			MPI_Recv(&recvd_solution_error, 1, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(recvd_solution_error > solution_error) solution_error = recvd_solution_error;
		}

	    // Output:
	    std::cout                                                              << std::endl << std::endl;
	    std::cout<< "-------------------------------------------------------"               << std::endl;
	    std::cout<< "SUMMARY:"                                                 << std::endl << std::endl;
	    std::cout<< "The error between two iterates is "    << iteration_error << std::endl << std::endl;
	    std::cout<< "The maximum error in the solution is " << solution_error               << std::endl;
	    std::cout<< "Time to reach the tolerance (mpi): "  << elapsed_time << "(s)"  << std::endl << std::endl;
	    std::cout<< "-------------------------------------------------------"  << std::endl << std::endl;
	    
	}


	// Free dynamic memory
	free(Lsend);
	free(Rsend);
	free(Tsend);
	free(Bsend);
	free(Lghost);
	free(Rghost);
	free(Tghost);
	free(Bghost);

	MPI_Finalize();
	return 0;

}

double update_formula(double right, double left, double top, double bottom, double S) {
	return ((dy * dy * (right + left - S * dx * dx)) + (top * dx * dx) + (bottom * dx * dx)) / (2 * ((dx * dx) + (dy * dy)));
}


double x(int i) {
	return a+i*dx;
}

double y(int j) {
	return c+j*dy;
}

double exact_solution(double x, double y) {
	return sin(2 * M_PI * x) * cos(2 * M_PI * y);
}

double S(double x, double y) {
	return -8 * M_PI * M_PI * exact_solution(x,y);
}
