#include <iostream>
#include <stdio.h>
// #include <omp.h>

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  First, recall that the notation u_xx means the second derivative of u with respect to x.
//
//  We are interested in solving:
//
//                  u_xx = S,
//
//  where u=u(x) is the temperature at steady-state and S=S(x) is the source term.
//
//  The solution is computed in the interval [a, b].
//  The solution is given at x=a and at x=b. These are called boundary conditions.
//
//  Notations:
//      The interval [a, b] is discretized into (m-1) segments of equal size. We say that m is the
//      number of grid points (or nodes).
//      The parameter m is chosen by the user.
//      The larger m, the more precise the computation but the longer it takes.
//      The grid points are denoted by x_0, x_1, ..., x_{m-1}. Here, x_i=a+i*dx, where dx=(b-a)/(m-1).
//
//      The equation u_xx=S can be approximated (from numerical analysis) by the formula:
//
//      (u^n_{i+1} -2*u^{n+1}_{i} +u^n_{i-1})/dx/dx = S_i^n.
//
//      This approximation gives the formula to compute u^{n+1}_i from u^n_i as:
//
//      u^{n+1}_{i} = (u^{n}_{i+1} + u^{n}_{i-1} - dx*dx*S^n_i)/2.  This is called a Jacobi iteration.
//
//      The idea of the solution process is thus to start from a given initial solution u^0_i
//      (for all grid points) and use the formula above to find u^1_i (for all grid points).
//      Once u^1_i is found, the formula is used to find u^2_i and so on.
//      The steady-state is reached when u^n and u^{n+1} are very close to each other. In fact,
//      we compute the max|u^{n+1}_i - u^n_i| and if it is less than a chosen tolerance, we declare
//      that steady-state is reached.
//
//      The code below implements this process.
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#include <stdlib.h>
#include <math.h>

using namespace std;

// Prototypes of Helper functions:
double x(int i);                                // x-coordinate in the [a, b] interval corresponding to index i.
double y(int i);                                // y-coordinate in the [c, d] interval corresponding to index i.
double exact_solution(double x, double y);      // Definition of the exact solution.
double S(double x, double y);                   // Definition of the source term.
double sqr(double i);

// Givens of the problem:
const double a = 0.0; const double b = 1.0; // Interval [a, b] chosen.
const double c = 0.0; const double d = 1.0; // Interval [c, d] chosen.
const int m = 100;                       // Number of discretization points chosen in the x direction.
const int n = 100;                       // Number of discretization points chosen in the y direction.

// Step sizes:
const double dx = (b - a) / (m - 1);
const double dy = (d - c) / (n - 1);

// We need 2 arrays, Un and Unp1, for computing the solution:
double Un[m][n];
double Unp1[m][n];

int main() {
    // index:
    int i, j;

    // Variables used to establish the convergence of the Jacobi iterations:
    double iteration_error = 1.;
    double tolerance = 1E-15;
    int Max_Iter = 1000000;

    // double startTime = omp_get_wtime();

    // Initialize Un arbitrarily to 0:
    for (i = 1; i < m - 1; i++) {
        for (j = 1; j < n - 1; j++) {
            Un[i][j] = 0.0;
        }
    }

    // Treat the bottom and top boundary conditions for Un:
    for (i = 0; i < m; i++) {
        Un[i][0] = exact_solution(x(i), c);
        Un[i][n - 1] = exact_solution(x(i), d);
    }

    // Treat the left and right boundary conditions for Un:
    for (j = 0; j < n; j++) {
        Un[0][j] = exact_solution(a, y(j));
        Un[m - 1][j] = exact_solution(b, y(j));
    }

    // Initialize the interation counter:
    int iteration_count = 0;

    // Iterate until steady-state is reached.
    // Otherwise stops at Max_Iter iterations (to avoid infinite loops).
    //
    // Recall that we say that the steady-state is reached when the maximum difference between
    // two iterates is less than or equal to the tolerance, i.e. max|Unp1-Un| <= tolerance.

    while (iteration_error > tolerance && iteration_count < Max_Iter) {
        iteration_count++; // if(iteration_count % 1000 == 0) std::cout<<"iteration " << iteration_count << std::endl;


        // Treat the bottom and top boundary conditions for Unp1:
        for (i = 0; i < m; i++) {
            Unp1[i][0] = exact_solution(x(i), c);
            Unp1[i][n - 1] = exact_solution(x(i), d);
        }

        // Treat the left and right boundary conditions for Unp1:
        for (j = 0; j < n; j++) {
            Unp1[0][j] = exact_solution(a, y(j));
            Unp1[m - 1][j] = exact_solution(b, y(j));
        }

        for (i = 1; i < m - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                Unp1[i][j] = ( sqr(dy) * (Un[i + 1][j] + Un[i - 1][j] - S(x(i), y(j)) *
                                          sqr(dx)) + Un[i][j + 1] * sqr(dx) + Un[i][j - 1] * sqr(dx) )
                             / (2 * (sqr(dx) + sqr(dy)));
            }
        }

        // Compute the maximum error between 2 iterates to establish whether or not
        // steady-state is reached:
        iteration_error = 0.0;
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                double local_iteration_error = fabs(Unp1[i][j] - Un[i][j]);
                if (local_iteration_error > iteration_error) iteration_error = local_iteration_error;
            }
        }

        // Prepare for the next iteration:
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                Un[i][j] = Unp1[i][j];
            }
        }

        // if(iteration_count % 1000 == 0) std::cout<< "The error between two iterates is " << iteration_error << std::endl;
    }

    // Compute the maximum error between the computed and exact solutions:
    double solution_error = 0.0;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            double local_solution_error = fabs(Unp1[i][j] - exact_solution(x(i), y(j)) );
            if (local_solution_error > solution_error) solution_error = local_solution_error;
        }
    }

    // double endTime = omp_get_wtime();
    // double elapsedTime = endTime - startTime;

    // Output:
    // std::cout                                                              << std::endl << std::endl;
    // std::cout << "-------------------------------------------------------"               << std::endl;
    // std::cout << "SUMMARY:"                                                 << std::endl << std::endl;
    // std::cout << "The error between two iterates is "    << iteration_error << std::endl << std::endl;
    // std::cout << "The maximum error in the solution is " << solution_error               << std::endl;
    // std::cout << "-------------------------------------------------------"  << std::endl << std::endl;
    // cout << "\tElapsted time: " << elapsedTime << endl << endl;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            cout << exact_solution(x(i), y(j));
            if (j != n - 1) cout << " ";
        }
        cout << endl;
    }

    return 0;
}

// Helper functions:

double x(int i) {
    return a + i * dx;
}

double y(int i) {
    return c + i * dy;
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