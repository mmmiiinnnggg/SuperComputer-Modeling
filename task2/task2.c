#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

double f(double x, double y, double z){
    return sqrt(y*y+z*z);
}

double generate_point(double left_range, double right_range){
    return ((double) rand() / RAND_MAX) * (right_range - left_range) + left_range;
}

int main(int argc, char * argv[]){
        // to ensure the randomness for every process
        // add ten more points for every fail of approximating real integral
        srand(time(0));
        int n_count = 1, my_id, num_procs, i;
        double real_integral = 4.0 / 3 * M_PI, sum = 0.0, root_value, integral, current_gap;
        int m = atoi(argv[1]);
        double eps = atof(argv[2]);
        MPI_Init(&argc, &argv);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
        double start = MPI_Wtime();

        while(1){
            for (i = my_id + 1; i <= m * num_procs; i+= num_procs){
                    double x = generate_point(0.0, 2.0);
                    double y = generate_point(-1.0, 1.0);
                    double z = generate_point(-1.0, 1.0);
                    if (y*y+z*z<=1){
                        sum += f(x,y,z);
                    }
            }

            MPI_Reduce(&sum, &root_value, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            if (!my_id){
                integral = 8.0 * root_value / (m * num_procs * n_count);
                current_gap = fabs(integral - real_integral);

                if (current_gap <= eps){
                    printf("\nWell done! The integral is approximately %.6f \n", integral);
                    printf("The final computing error is %.6f\n", current_gap);
                    printf("Generated %d random points\n", m * num_procs * n_count);
                    break;
                }


                else {
                    //printf("simulated points: %d, The current gap: %.8f \n", m * num_procs * n_count, current_gap);
                    n_count += 1;
                }
            }
        }
        double end = MPI_Wtime();
        printf("The running time is %.6f s\n", end - start);
        MPI_Abort(MPI_COMM_WORLD, 666);
        return 0;

}