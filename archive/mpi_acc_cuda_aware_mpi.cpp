#include <mpi.h>
// #include <omp.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <openacc.h>

// TODO: decouple all of the functions to minimize overhead

#define A1 0
#define A2 4
#define B1 0
#define B2 3
#define PRINT_FREQ 1000
// #define EPS 1e-6

#define TAG_X 666
#define TAG_Y 999

using namespace std;

double u(double x, double y){
    return sqrt(4 + x * y);
}

double k(double x, double y){
    return 4 + x + y;
}

double q(double x, double y){
    return x + y;
}

double F(double x, double y){
    return ((4 + x + y) * (pow(x, 2) + pow(y, 2)) - 2.0 * (x + y)*(4 + x * y)) / (4.0 * pow(4 + x * y, 1.5)) + (x + y) * sqrt(4 + x * y);
}

// left bound w.r.t y
double left_bound(double y){
    return 1 / 4.0 * (8 - 4 * y - pow(y, 2));
}

// right bound w.r.t y
double right_bound(double y){
    return (pow(y, 2) + 16 * y + 8.0) / (4.0 * sqrt(1 + y));
}

// upper bound w.r.t. x 
double upper_bound(double x){
    return (x * (7 + x) * sqrt(4.0 + 3 * x)) / (2.0 * (4 + 3 * x));
}

// lowee bound w.r.t x
double lower_bound(double x){
    return - 1 / 4.0 * (pow(x, 2) + 4 * x);
}

// difference of two vectors
void vector_diff(double *diff, double *w1, double *w2, int size_x, int size_y){

    // do not exclude padding points
    int i, j;
    // #pragma acc parallel loop private(i ,j)
    // int size = (size_x + 2) * (size_y + 2);
    // #pragma acc data copyin(w1[0:size], w2[0:size]) copyout(diff[0:size])
    // {
    int _size = (size_x + 2) * (size_y + 2);
    #pragma acc kernels present(w1[0:_size], w2[0:_size], diff[0:_size])
    {
    #pragma acc loop independent
    for(i = 0; i <= size_x + 1; i++){
        #pragma acc loop independent
        for(j = 0; j <= size_y + 1; j++){
            diff[i * (size_y + 2) + j] = w1[i * (size_y + 2) + j] - w2[i * (size_y + 2) + j];
        }
    }
    
    }
    // }
}

// inner product of two 1-d vectors
double _inner_product(double *w0, double *w1, int size_x, int size_y, int i_x, int j_y, int M, int N, double h1, double h2)
{
    double res = 0;

    // do not count padding points
    int i, j, _i, _j;
    // int size = (size_x + 2) * (size_y + 2);
    // #pragma acc data copy(w0[0:size], w1[0:size]) copyout(res)
    // {
    // #pragma acc data copyout(res)
    // {
    int _size = (size_x + 2) * (size_y + 2);
    #pragma acc kernels present(w0[0:_size], w1[0:_size])
    {
    #pragma acc loop independent reduction(+:res)
    for (i = 1; i <= size_x; i++)
    {
        #pragma acc loop independent reduction(+:res)
        for (j = 1; j <= size_y; j++)
        {
            _i = i_x + i - 1;
            _j = j_y + j - 1;

            // corner points
            if (_i == 0 && _j == 0 || _i == 0 && _j == N || _i == M && _j == 0 || _i == M && _j == N)
                res += 0.25 * w0[i * (size_y + 2) + j] * w1[i * (size_y + 2) + j];
            // inner points
            else if (_i >= 1 && _i <= M - 1 && _j >= 1 && _j <= N - 1)
                res += 1.0 * w0[i * (size_y + 2) + j] * w1[i * (size_y + 2) + j];
            // bound points
            else res += 0.5 * w0[i * (size_y + 2) + j] * w1[i * (size_y + 2) + j];
        }
    }
    }
    // }

    return res * h1 * h2;
}

// norm of 1-d vector
double norm(double *w, int size_x, int size_y, int i_x, int j_y, int M, int N, double h1, double h2)
{
    return sqrt(_inner_product(w, w, size_x, size_y, i_x, j_y, M, N, h1, h2));
} 


// initialize matrix B
void init_B(double *B, int M, int N, int size_x, int size_y, int i_x, int j_y, double h1, double h2){
    int i, j, _i, _j;
    // #pragma acc parallel loop private(i, j, _i, _j)
    // int size = (size_x + 2) * (size_y + 2);
    // #pragma acc data copyout(B[0:size])
    // {
    // #pragma acc kernels present(B)
    int _size = (size_x + 2) * (size_y + 2);
    #pragma acc kernels present(B[0:_size])
    {
    #pragma acc loop independent
    for (i = 1; i <= size_x; i++)
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            
            _i = i_x + i - 1;
            _j = j_y + j - 1;

            // // the padding points
            // if (i == 0 || i == size_x + 1 || j == 0 || j == size_y + 1){
            //     B[i * (size_y + 2) + j] = 0;
            //     continue;
            // }

            // left bound
            if (_i == 0 && _j >= 1 && _j <= N - 1) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h1) * left_bound(B1 + _j * h2);
            // right bound
            else if (_i == M && _j >= 1 && _j <= N - 1) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h1) * right_bound(B1 + _j * h2);
            // lower bound 
            else if (_j == 0 && _i >= 1 && _i <= M - 1) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h2) * lower_bound(A1 + _i * h1);
            // upper bound 
            else if (_j == N && _i >= 1 && _i <= M - 1) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h2) * upper_bound(A1 + _i * h1);

            // the corner values aren't that important...
            // left-lower corner
            else if (_i == 0 && _j == 0) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h1 + 2 / h2 ) * (left_bound(B1 + _j * h2) + lower_bound(A1 + _i * h1)) / 2.0;
            // left-upper corner
            else if (_i == 0 && _j == N) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h1 + 2 / h2 ) * (left_bound(B1 + _j * h2) + upper_bound(A1 + _i * h1)) / 2.0;
            // right-lower corner
            else if (_i == M && _j == 0) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h1 + 2 / h2 ) * (right_bound(B1 + _j * h2) + lower_bound(A1 + _i * h1)) / 2.0;
            // right-upper corner
            else if (_i == M && _j == N) 
                B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2) + (2 / h1 + 2 / h2 ) * (right_bound(B1 + _j * h2) + upper_bound(A1 + _i * h1)) / 2.0;
            
            // the center points
            else B[i * (size_y + 2) + j] = F(A1 + _i * h1, B1 + _j * h2);
        }
    }
    // }/
}

// conduct matrix-vector multiplication
void A_vec_mult(double *Av_res, double *w, int M, int N, int size_x, int size_y, int i_x, int j_y, double h1, double h2){ 

    int i, j, _i, _j;
    // #pragma acc parallel loop private(i, j, _i, _j)
    // int size = (size_x + 2) * (size_y + 2);
    // #pragma acc data copyin(w) copyout(Av_res[0:size])
    // {
    // #pragma acc kernels present(Av_res, w)
    int _size = (size_x + 2) * (size_y + 2);
    #pragma acc kernels present(Av_res[0:_size], w[0:_size])
    {
    #pragma acc loop independent
    for (i = 1; i <= size_x; i++)
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            
            // the global step 
            _i = i_x + i - 1;
            _j = j_y + j - 1; 

            // shoule care about the offset of w
            // M -> size_x , N -> size_y
            // 1 -> 1 + 1, 0 -> 1 + 0
            // but the i and j are corresponded well

            // left bound
            if (_i == 0 && _j >= 1 && _j <= N - 1){
                Av_res[i * (size_y + 2) + j] = - (2 / h1 ) * k(A1 + (1 - 0.5) * h1, B1 + _j * h2) * (w[(1 + 1) * (size_y + 2) + j] - w[(1 + 0) * (size_y + 2) + j]) / h1
                    + (q(A1 + _i * h1, B1 + _j * h2) + 2 / h1 ) * w[(1 + 0) * (size_y + 2) + j] 
                    - (1 / h2) * k(A1 + 0 * h1, B1 + (_j + 0.5) * h2) * (w[(1 + 0) * (size_y + 2) + (j + 1)] - w[(1 + 0) * (size_y + 2) + j]) / h2
                    + (1 / h2) * k(A1 + 0 * h1, B1 + (_j - 0.5) * h2) * (w[(1 + 0) * (size_y + 2) + j] - w[(1 + 0) * (size_y + 2) + (j - 1)]) / h2;
            } 
            // right bound
            else if (_i == M && _j >= 1 && _j <= N - 1){
                Av_res[i * (size_y + 2) + j] = (2 / h1 ) * k(A1 + (M - 0.5) * h1, B1 + _j * h2) * (w[size_x * (size_y + 2) + j] - w[(size_x - 1) * (size_y + 2) + j]) / h1
                    + (q(A1 + _i * h1, B1 + _j * h2) + 2 / h1 ) * w[size_x * (size_y + 2) + j] 
                    - (1 / h2) * k(A1 + M * h1, B1 + (_j + 0.5) * h2) * (w[size_x * (size_y + 2) + (j + 1)] - w[size_x * (size_y + 2) + j]) / h2
                    + (1 / h2) * k(A1 + M * h1, B1 + (_j - 0.5) * h2) * (w[size_x * (size_y + 2) + j] - w[size_x * (size_y + 2) + (j - 1)]) / h2;
            }
            // lower bound 
            else if (_j == 0 && _i >= 1 && _i <= M - 1){
                Av_res[i * (size_y + 2) + j] = - (2 / h2 ) * k(A1 + _i * h1, B1 + (1 - 0.5) * h2) * (w[i * (size_y + 2) + (1 + 1)] - w[i * (size_y + 2) + (1 + 0)]) / h2
                    + q(A1 + _i * h1, B1 + _j * h2) * w[i * (size_y + 2) + (1 + 0)] 
                    - (1 / h1) * k(A1 + (_i + 0.5) * h1, B1 + _j * h2) * (w[(i + 1) * (size_y + 2) + (1 + 0)] - w[i * (size_y + 2) + (1 + 0)]) / h1
                    + (1 / h1) * k(A1 + (_i - 0.5) * h1, B1 + _j * h2) * (w[i * (size_y + 2) + (1 + 0)] - w[(i - 1) * (size_y + 2) + (1 + 0)]) / h1;
            }
            // upper bound 
            else if (_j == N && _i >= 1 && _i <= M - 1){
                Av_res[i * (size_y + 2) + j] = (2 / h2 ) * k(A1 + _i * h1, B1 + (N - 0.5) * h2) * (w[i * (size_y + 2) + size_y] - w[i * (size_y + 2) + (size_y - 1)]) / h2
                    + q(A1 + _i * h1, B1 + _j * h2) * w[i * (size_y + 2) + size_y] 
                    - (1 / h1) * k(A1 + (_i + 0.5) * h1, B1 + _j * h2) * (w[(i + 1) * (size_y + 2) + size_y] - w[i * (size_y + 2) + size_y]) / h1
                    + (1 / h1) * k(A1 + (_i - 0.5) * h1, B1 + _j * h2) * (w[i * (size_y + 2) + size_y] - w[(i - 1) * (size_y + 2) + size_y]) / h1;
            }

            // left-lower corner
            else if (_i == 0 && _j == 0){
                Av_res[i * (size_y + 2) + j] = - (2 / h1 ) * k(A1 + (1 - 0.5) * h1, B1 + 0 * h2) * (w[(1 + 1) * (size_y + 2) + (1 + 0)] - w[(1 + 0) * (size_y + 2) + (1 + 0)]) / h1
                    - (2 / h2 ) * k(A1 + 0 * h1, B1 + (1 - 0.5) * h2) * (w[(1 + 0) * (size_y + 2) + (1 + 1)] - w[(1 + 0) * (size_y + 2) + (1 + 0)]) / h2
                    + (q(A1 + _i * h1, B1 + _j * h2) + 2 / h1) * w[(1 + 0) * (size_y + 2) + (1 + 0)];
            }
            // left-upper corner
            else if (_i == 0 && _j == N){
                Av_res[i * (size_y + 2) + j] = - (2 / h1 ) * k(A1 + (1 - 0.5) * h1, B1 + N * h2) * (w[(1 + 1) * (size_y + 2) + size_y] - w[(1 + 0) * (size_y + 2) + size_y]) / h1
                    + (2 / h2 ) * k(A1 + 0 * h1, B1 + (N - 0.5) * h2) * (w[(1 + 0) * (size_y + 2) + size_y] - w[(1 + 0) * (size_y + 2) + (size_y - 1)]) / h2
                    + (q(A1 + _i * h1, B1 + _j * h2) + 2 / h1) * w[(1 + 0) * (size_y + 2) + size_y];
            } 
            // right-lower corner
            else if (_i == M && _j == 0){
                Av_res[i * (size_y + 2) + j] =  (2 / h1 ) * k(A1 + (M - 0.5) * h1, B1 + 0 * h2) * (w[size_x * (size_y + 2) + (1 + 0)] - w[(size_x - 1) * (size_y + 2) + (1 + 0)]) / h1
                    - (2 / h2 ) * k(A1 + M * h1, B1 + (1 - 0.5) * h2) * (w[size_x * (size_y + 2) + (1 + 1)] - w[size_x * (size_y + 2) + (1 + 0)]) / h2
                    + (q(A1 + _i * h1, B1 + _j * h2) + 2 / h1) * w[size_x * (size_y + 2) + (1 + 0)];
            }
            // right-upper corner
            else if (_i == M && _j == N){
                Av_res[i * (size_y + 2) + j] = (2 / h1 ) * k(A1 + (M - 0.5) * h1, B1 + N * h2) * (w[size_x * (size_y + 2) + size_y] - w[(size_x - 1) * (size_y + 2) + size_y]) / h1
                    + (2 / h2 ) * k(A1 + M * h1, B1 + (N - 0.5) * h2) * (w[size_x * (size_y + 2) + size_y] - w[size_x * (size_y + 2) + (size_y - 1)]) / h2
                    + (q(A1 + _i * h1, B1 + _j * h2) + 2 / h1) * w[size_x * (size_y + 2) + size_y];
            }

            // the center points
            else Av_res[i * (size_y + 2) + j] = - (1 / h1) * k(A1 + (_i + 0.5) * h1, B1 + _j * h2) * (w[(i + 1) * (size_y + 2) + j] - w[i * (size_y + 2) + j]) / h1
                + (1 / h1) * k(A1 + (_i - 0.5) * h1, B1 + _j * h2) * (w[i * (size_y + 2) + j] - w[(i - 1) * (size_y + 2) + j]) / h1
                - (1 / h2) * k(A1 + _i * h1, B1 + (_j + 0.5) * h2) * (w[i * (size_y + 2) + (j + 1)] - w[i * (size_y + 2) + j]) / h2
                + (1 / h2) * k(A1 + _i * h1, B1 + (_j - 0.5) * h2) * (w[i * (size_y + 2) + j] - w[i * (size_y + 2) + (j - 1)]) / h2
                + q(A1 + _i * h1, B1 + _j * h2) * w[i * (size_y + 2) + j];     
        }
    }
    // }
}

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);

    double start = MPI_Wtime();

    // overall sizes
    int M, N;
    // steps of approximation
    double h1, h2;
    double eps;

    M = atoi(argv[1]), N = atoi(argv[2]);
    h1 = (double)(A2 - A1) / M;
    h2 = (double)(B2 - B1) / N; 
    eps = atof(argv[3]);

    int size, rank;
    MPI_Comm cart_comm;
    MPI_Status status;
    // omp_lock_t dmax_lock;
    
    // the number of processeds in each dimension
    int proc_number[2] = {0, 0};
    // the coords of process in the topology
    int coords[2];

    // size of the block
    int _size, size_x, size_y;
    // shifted i and j 
    int i_x, j_y;
    // tau global
    double diff_local, diff, tau_numerator_local, tau_denominator_local, tau_global;
    double tau_numerator_global, tau_denominator_global;

    // the send & recv buffers 
    double *s_buf_up, *s_buf_down, *s_buf_left, *s_buf_right;
    double *r_buf_up, *r_buf_down, *r_buf_left, *r_buf_right;
    // the intermidate results
    double *Aw, *Ar, *B, *r;
    double *w, *w_pr, *diff_w_and_w_pr;

    // void create_communicator();
    // void init_processor_config();
    // void fill_data();
    // void exchange_data(double *_w);
    // void exchange_data_B();
    // one iteration for solving the linear system
    // double solve_iteration();

    // get current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // get processes number
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // divide processes to 2 dims and store in `proc_number` array
    MPI_Dims_create(size, 2, proc_number);

    // multi-gpu affinity
    int n_gpus = acc_get_num_devices(acc_device_nvidia);
    int device_num = rank % n_gpus;
    acc_set_device_num(device_num, acc_device_nvidia);
    acc_init(acc_device_nvidia);


// create communicator for Cartisan coords

    // boolean array to define periodicy of each dimension
    int Periods[2] = {0, 0};

    // (world communicator, num of dims, dim_size, periodicy for each dimention, no reorder, cart_comm)
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc_number, Periods, 0, &cart_comm);

    // (cart_comm, the given rank, num of dims, the corresponding coords)
    MPI_Cart_coords(cart_comm, rank, 2, coords);


// init size, shift, block, bounds and buffers

    // calculate the size_x & size_y
    // need to garantee that each size w.r.t x and y has diff <= 1

    // there are M + 1 and N + 1 points
    size_x = (M + 1) / proc_number[0];
    size_y = (N + 1) / proc_number[1];

    // distribute the extra nodes to the first processes
    if (coords[0] < (M + 1) % proc_number[0]) size_x += 1;
    if (coords[1] < (N + 1) % proc_number[1]) size_y += 1;

    // calculate the i_x & j_y (real start nodes of this block)
    i_x = coords[0] * ((M + 1) / proc_number[0]) + min((M + 1) % proc_number[0], coords[0]);
    j_y = coords[1] * ((N + 1) / proc_number[1]) + min((N + 1) % proc_number[1], coords[1]);

    if (rank == 0){
        cout << "basic processors and task info:" << endl
        << "proc_dims = " << proc_number[0] << " " << proc_number[1]
        << " | M, N, h1, h2= " << M << " " << N << " " << h1 << " " << h2
        << endl;
    }
    
    cout << "rank " << rank 
    << " | size_x, size_y= " << size_x << " " << size_y 
    << " | i_x, i_y= " << i_x << " " << j_y 
    << " | device = " << acc_get_device_num(acc_device_nvidia)
    << endl;

    // init send & recv buffers for every direction
    r_buf_up = new double [size_x];
    r_buf_down = new double [size_x];
    r_buf_left = new double [size_y];
    r_buf_right = new double [size_y]; 
    s_buf_up = new double [size_x];
    s_buf_down = new double [size_x];
    s_buf_left = new double [size_y];
    s_buf_right = new double [size_y];


    // allocate memory
    // padding = 1 to better perform A_vec_mult
    Aw = new double [(size_x + 2) * (size_y + 2)];
    Ar = new double [(size_x + 2) * (size_y + 2)];
    B = new double [(size_x + 2) * (size_y + 2)];
    r = new double [(size_x + 2) * (size_y + 2)];
    w = new double [(size_x + 2) * (size_y + 2)];
    w_pr = new double [(size_x + 2) * (size_y + 2)];
    diff_w_and_w_pr = new double [(size_x + 2) * (size_y + 2)];

    _size = (size_x + 2) * (size_y + 2);
    // #pragma acc enter data copyin(this)
    #pragma acc enter data copyin(diff_local, diff, tau_numerator_local, tau_denominator_local, tau_global)
    #pragma acc enter data copyin(tau_numerator_global, tau_denominator_global)
    #pragma acc enter data copyin(M, N, size_x, size_y, i_x, j_y, h1, h2)
    #pragma acc enter data copyin(Aw[0:_size], Ar[0:_size], B[0:_size], r[0:_size], w[0:_size], w_pr[0:_size], diff_w_and_w_pr[0:_size])


// fill up data for B, and allocate memory
    int i, j;

    // init w_pr
    #pragma acc kernels present(w_pr[0:_size], w[0:_size])
    {
    #pragma acc loop independent
    for(i = 0; i <= size_x + 1; i++)
        #pragma acc loop independent
        for(j = 0; j <= size_y + 1; j++)
            // init w_pr
            w_pr[i * (size_y + 2) + j] = 2.5;

            // init w
            w[i * (size_y + 2) + j] = 2.5;

    }
    // init w_0
    // #pragma acc kernels present(w_pr[0:_size], w[0:_size])
    // {
    // #pragma acc loop independent  
    // for(i = 0; i <= size_x + 1; i++)
    //     #pragma acc loop independent
    //     for(j = 0; j <= size_y + 1; j++)
    //         w[i * (size_y + 2) + j] = 2.5;
    // }

    init_B(B, M, N, size_x, size_y, i_x, j_y, h1, h2);

    #pragma acc enter data copyin(r_buf_up[0:size_x], r_buf_down[0:size_x], r_buf_left[0:size_y], r_buf_right[0:size_y])
    #pragma acc enter data copyin(s_buf_up[0:size_x], s_buf_down[0:size_x], s_buf_left[0:size_y], s_buf_right[0:size_y])


// the main solve loop
    if (rank == 0){
        cout << "Number of device : " << acc_get_num_devices(acc_device_not_host) << endl;
    }

/* .................................................................................................
........................START EXCHANGE VECTOR B ............................................................
....................................................................................................*/

// exchange data of the boundaries - B 
    int rank_recv, rank_send;
    // int c, i, j;

    // along x to the left -> dim = 0, disp = -1
    // (communicator, direction, disp, source, dest)
    // cout << "rank: " << rank << " checking -1............." << endl; 
    MPI_Cart_shift(cart_comm, 0, -1, &rank_recv, &rank_send);
    // cout << "rank: " << rank << " checking 0............." << " recv, send " << rank_recv << " " << rank_send << endl; 
    // the inner processes: send & recv
    if (coords[0] != 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        
        #pragma acc kernels present(s_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_left[j - 1] = B[1 * (size_y + 2) + j];
            // c++;
        }}
        
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_left, r_buf_right)
        MPI_Sendrecv(s_buf_left, size_y, MPI_DOUBLE, rank_send, TAG_X,
                    r_buf_right, size_y, MPI_DOUBLE, rank_recv, TAG_X,
                    cart_comm, &status);

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            B[(size_x + 1) * (size_y + 2) + j] = r_buf_right[j - 1];
            // c++;
        }}
    }
    // the left process: recv
    else if (coords[0] == 0 && coords[0] != proc_number[0] - 1){

        // cout << "rank: " << rank << " checking 1............." << endl; 

        #pragma acc host_data use_device(r_buf_right)
        {
        MPI_Recv(r_buf_right, size_y, MPI_DOUBLE,
        rank_recv, TAG_X, cart_comm, &status);
        }

        // cout << "rank: " << rank << " checking 2............." << endl; 

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            B[(size_x + 1) * (size_y + 2) + j] = r_buf_right[j - 1];
            // c++;
        }}
    }
    // the right process: send
    else if (coords[0] != 0 && coords[0] == proc_number[0] - 1){
        // generate send_buffer
        // c = 0;

        #pragma acc kernels present(s_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_left[j - 1] = B[1 * (size_y + 2) + j];
            // c++;
        }}

        // cout << "rank: " << rank << " checking 1 send ............." << endl; 
        // #pragma acc update host(s_buf_left[0:size_y])

        // cout << "try to print: rank: " << rank << " ";
        // for (j = 1; j <= size_y; j ++) cout << s_buf_left[j] << " " ;
        
        // cout << endl;

        #pragma acc host_data use_device(s_buf_left)
        MPI_Send(s_buf_left, size_y, MPI_DOUBLE,
        rank_send, TAG_X, cart_comm);

        // cout << "rank: " << rank << " checking 2 send ............." << endl; 
    }


    // along x to the right  -> dim = 0, disp = 1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 0, 1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[0] != 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_right[j - 1] = B[size_x * (size_y + 2) + j];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_right, r_buf_left)
        MPI_Sendrecv(s_buf_right, size_y, MPI_DOUBLE, rank_send, TAG_X,
                    r_buf_left, size_y, MPI_DOUBLE, rank_recv, TAG_X,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            B[0 * (size_y + 2) + j] = r_buf_left[j - 1];
            // c++;
        }}
    }
    // the left process: send
    else if (coords[0] == 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_right[j - 1] = B[size_x * (size_y + 2) + j];
            // c++;
        }}
        #pragma acc host_data use_device(s_buf_right)
        MPI_Send(s_buf_right, size_y, MPI_DOUBLE,
        rank_send, TAG_X, cart_comm);

    }
    // the right process: recv
    else if (coords[0] != 0 && coords[0] == proc_number[0] - 1){
        #pragma acc host_data use_device(r_buf_left)
        MPI_Recv(r_buf_left, size_y, MPI_DOUBLE,
        rank_recv, TAG_X, cart_comm, &status);

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            B[0 * (size_y + 2) + j] = r_buf_left[j - 1];
            // c++;
        }}
    }

    // along y to the low -> dim = 1, disp = -1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 1, -1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[1] != 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_down[i - 1] = B[i * (size_y + 2) + 1];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag, 
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_down, r_buf_up)
        MPI_Sendrecv(s_buf_down, size_x, MPI_DOUBLE, rank_send, TAG_Y,
                    r_buf_up, size_x, MPI_DOUBLE, rank_recv, TAG_Y,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            B[i * (size_y + 2) + (size_y + 1)] = r_buf_up[i - 1];
            // c++;
        }}
    }
    // the lower process: recv
    else if (coords[1] == 0 && coords[1] != proc_number[1] - 1){
        #pragma acc host_data use_device(r_buf_up)
        MPI_Recv(r_buf_up, size_x, MPI_DOUBLE,
        rank_recv, TAG_Y, cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            B[i * (size_y + 2) + (size_y + 1)] = r_buf_up[i - 1];
            // c++;
        }}
    }
    // the upper process: send
    else if (coords[1] != 0 && coords[1] == proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_down[i - 1] = B[i * (size_y + 2)  + 1];
            // c++;
        }}

        #pragma acc host_data use_device(s_buf_down)
        MPI_Send(s_buf_down, size_x, MPI_DOUBLE,
        rank_send, TAG_Y, cart_comm);
    }


    // along y to the up -> dim = 1, disp = 1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 1, 1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[1] != 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_up[i - 1] = B[i * (size_y + 2) + size_y];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_up, r_buf_down)
        MPI_Sendrecv(s_buf_up, size_x, MPI_DOUBLE, rank_send, TAG_Y,
                    r_buf_down, size_x, MPI_DOUBLE, rank_recv, TAG_Y,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            B[i * (size_y + 2) + 0 ] = r_buf_down[i - 1];
            // c++;
        }}
    }
    // the lower process: send
    else if (coords[1] == 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_up[i - 1] = B[i * (size_y + 2) + size_y];
            // c++;
        }}
        #pragma acc host_data use_device(s_buf_up)
        MPI_Send(s_buf_up, size_x, MPI_DOUBLE,
        rank_send, TAG_Y, cart_comm);
    }
    // the upper process: recv
    else if (coords[1] != 0 && coords[1] == proc_number[1] - 1){
        #pragma acc host_data use_device(r_buf_down)
        MPI_Recv(r_buf_down, size_x, MPI_DOUBLE,
        rank_recv, TAG_Y, cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            B[i * (size_y + 2) + 0] = r_buf_down[i - 1];
            // c++;
        }}
    }

/* .................................................................................................
........................END EXCHANGE VECTOR B ............................................................
....................................................................................................*/

    // double diff;
    int iter = 0;

    if(rank == 0) cout << "Starting..." << endl;

    do{
        iter++;

        // diff_local = solve_iteration();

        diff_local = 0;
        #pragma acc update device(diff_local)
        double tau_numerator_global, tau_denominator_global;

        //sync padding values
        // #pragma acc update self(w[0:_size])
        // if (rank == 0) cout << "checking on solve_iteration 1" << endl;
        /* .................................................................................................
........................START EXCHANGE VECTOR w ............................................................
....................................................................................................*/

// exchange data of the boundaries - B 
    int rank_recv, rank_send;
    int i, j;

    // along x to the left -> dim = 0, disp = -1
    // (communicator, direction, disp, source, dest)
    // cout << "rank: " << rank << " checking -1............." << endl; 
    MPI_Cart_shift(cart_comm, 0, -1, &rank_recv, &rank_send);
    // cout << "rank: " << rank << " checking 0............." << " recv, send " << rank_recv << " " << rank_send << endl; 
    // the inner processes: send & recv
    if (coords[0] != 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        
        #pragma acc kernels present(s_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_left[j - 1] = w[1 * (size_y + 2) + j];
            // c++;
        }}
        
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_left, r_buf_right)
        MPI_Sendrecv(s_buf_left, size_y, MPI_DOUBLE, rank_send, TAG_X,
                    r_buf_right, size_y, MPI_DOUBLE, rank_recv, TAG_X,
                    cart_comm, &status);

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            w[(size_x + 1) * (size_y + 2) + j] = r_buf_right[j - 1];
            // c++;
        }}
    }
    // the left process: recv
    else if (coords[0] == 0 && coords[0] != proc_number[0] - 1){

        // cout << "rank: " << rank << " checking 1............." << endl; 

        #pragma acc host_data use_device(r_buf_right)
        {
        MPI_Recv(r_buf_right, size_y, MPI_DOUBLE,
        rank_recv, TAG_X, cart_comm, &status);
        }

        // cout << "rank: " << rank << " checking 2............." << endl; 

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            w[(size_x + 1) * (size_y + 2) + j] = r_buf_right[j - 1];
            // c++;
        }}
    }
    // the right process: send
    else if (coords[0] != 0 && coords[0] == proc_number[0] - 1){
        // generate send_buffer
        // c = 0;

        #pragma acc kernels present(s_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_left[j - 1] = w[1 * (size_y + 2) + j];
            // c++;
        }}

        // cout << "rank: " << rank << " checking 1 send ............." << endl; 
        // #pragma acc update host(s_buf_left[0:size_y])

        // cout << "try to print: rank: " << rank << " ";
        // for (j = 1; j <= size_y; j ++) cout << s_buf_left[j] << " " ;
        
        // cout << endl;

        #pragma acc host_data use_device(s_buf_left)
        MPI_Send(s_buf_left, size_y, MPI_DOUBLE,
        rank_send, TAG_X, cart_comm);

        // cout << "rank: " << rank << " checking 2 send ............." << endl; 
    }


    // along x to the right  -> dim = 0, disp = 1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 0, 1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[0] != 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_right[j - 1] = w[size_x * (size_y + 2) + j];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_right, r_buf_left)
        MPI_Sendrecv(s_buf_right, size_y, MPI_DOUBLE, rank_send, TAG_X,
                    r_buf_left, size_y, MPI_DOUBLE, rank_recv, TAG_X,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            w[0 * (size_y + 2) + j] = r_buf_left[j - 1];
            // c++;
        }}
    }
    // the left process: send
    else if (coords[0] == 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_right[j - 1] = w[size_x * (size_y + 2) + j];
            // c++;
        }}
        #pragma acc host_data use_device(s_buf_right)
        MPI_Send(s_buf_right, size_y, MPI_DOUBLE,
        rank_send, TAG_X, cart_comm);

    }
    // the right process: recv
    else if (coords[0] != 0 && coords[0] == proc_number[0] - 1){
        #pragma acc host_data use_device(r_buf_left)
        MPI_Recv(r_buf_left, size_y, MPI_DOUBLE,
        rank_recv, TAG_X, cart_comm, &status);

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            w[0 * (size_y + 2) + j] = r_buf_left[j - 1];
            // c++;
        }}
    }

    // along y to the low -> dim = 1, disp = -1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 1, -1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[1] != 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_down[i - 1] = w[i * (size_y + 2) + 1];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag, 
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_down, r_buf_up)
        MPI_Sendrecv(s_buf_down, size_x, MPI_DOUBLE, rank_send, TAG_Y,
                    r_buf_up, size_x, MPI_DOUBLE, rank_recv, TAG_Y,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            w[i * (size_y + 2) + (size_y + 1)] = r_buf_up[i - 1];
            // c++;
        }}
    }
    // the lower process: recv
    else if (coords[1] == 0 && coords[1] != proc_number[1] - 1){
        #pragma acc host_data use_device(r_buf_up)
        MPI_Recv(r_buf_up, size_x, MPI_DOUBLE,
        rank_recv, TAG_Y, cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            w[i * (size_y + 2) + (size_y + 1)] = r_buf_up[i - 1];
            // c++;
        }}
    }
    // the upper process: send
    else if (coords[1] != 0 && coords[1] == proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_down[i - 1] = w[i * (size_y + 2)  + 1];
            // c++;
        }}

        #pragma acc host_data use_device(s_buf_down)
        MPI_Send(s_buf_down, size_x, MPI_DOUBLE,
        rank_send, TAG_Y, cart_comm);
    }


    // along y to the up -> dim = 1, disp = 1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 1, 1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[1] != 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_up[i - 1] = w[i * (size_y + 2) + size_y];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_up, r_buf_down)
        MPI_Sendrecv(s_buf_up, size_x, MPI_DOUBLE, rank_send, TAG_Y,
                    r_buf_down, size_x, MPI_DOUBLE, rank_recv, TAG_Y,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            w[i * (size_y + 2) + 0 ] = r_buf_down[i - 1];
            // c++;
        }}
    }
    // the lower process: send
    else if (coords[1] == 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_up[i - 1] = w[i * (size_y + 2) + size_y];
            // c++;
        }}
        #pragma acc host_data use_device(s_buf_up)
        MPI_Send(s_buf_up, size_x, MPI_DOUBLE,
        rank_send, TAG_Y, cart_comm);
    }
    // the upper process: recv
    else if (coords[1] != 0 && coords[1] == proc_number[1] - 1){
        #pragma acc host_data use_device(r_buf_down)
        MPI_Recv(r_buf_down, size_x, MPI_DOUBLE,
        rank_recv, TAG_Y, cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            w[i * (size_y + 2) + 0] = r_buf_down[i - 1];
            // c++;
        }}
    }

/* .................................................................................................
........................END EXCHANGE VECTOR B ............................................................
....................................................................................................*/


        // if (rank == 0) cout << "checking on solve_iteration 2" << endl;
        // #pragma acc update device(w[0:_size])

        A_vec_mult(Aw, w, M, N, size_x, size_y, i_x, j_y, h1, h2);

        /* .................................................................................................
........................START EXCHANGE VECTOR w ............................................................
....................................................................................................*/

// exchange data of the boundaries - B 
    // int rank_recv, rank_send;
    // int c, i, j;

    // along x to the left -> dim = 0, disp = -1
    // (communicator, direction, disp, source, dest)
    // cout << "rank: " << rank << " checking -1............." << endl; 
    MPI_Cart_shift(cart_comm, 0, -1, &rank_recv, &rank_send);
    // cout << "rank: " << rank << " checking 0............." << " recv, send " << rank_recv << " " << rank_send << endl; 
    // the inner processes: send & recv
    if (coords[0] != 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        
        #pragma acc kernels present(s_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_left[j - 1] = Aw[1 * (size_y + 2) + j];
            // c++;
        }}
        
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_left, r_buf_right)
        MPI_Sendrecv(s_buf_left, size_y, MPI_DOUBLE, rank_send, TAG_X,
                    r_buf_right, size_y, MPI_DOUBLE, rank_recv, TAG_X,
                    cart_comm, &status);

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            Aw[(size_x + 1) * (size_y + 2) + j] = r_buf_right[j - 1];
            // c++;
        }}
    }
    // the left process: recv
    else if (coords[0] == 0 && coords[0] != proc_number[0] - 1){

        // cout << "rank: " << rank << " checking 1............." << endl; 

        #pragma acc host_data use_device(r_buf_right)
        {
        MPI_Recv(r_buf_right, size_y, MPI_DOUBLE,
        rank_recv, TAG_X, cart_comm, &status);
        }

        // cout << "rank: " << rank << " checking 2............." << endl; 

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            Aw[(size_x + 1) * (size_y + 2) + j] = r_buf_right[j - 1];
            // c++;
        }}
    }
    // the right process: send
    else if (coords[0] != 0 && coords[0] == proc_number[0] - 1){
        // generate send_buffer
        // c = 0;

        #pragma acc kernels present(s_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_left[j - 1] = Aw[1 * (size_y + 2) + j];
            // c++;
        }}

        // cout << "rank: " << rank << " checking 1 send ............." << endl; 
        // #pragma acc update host(s_buf_left[0:size_y])

        // cout << "try to print: rank: " << rank << " ";
        // for (j = 1; j <= size_y; j ++) cout << s_buf_left[j] << " " ;
        
        // cout << endl;

        #pragma acc host_data use_device(s_buf_left)
        MPI_Send(s_buf_left, size_y, MPI_DOUBLE,
        rank_send, TAG_X, cart_comm);

        // cout << "rank: " << rank << " checking 2 send ............." << endl; 
    }


    // along x to the right  -> dim = 0, disp = 1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 0, 1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[0] != 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_right[j - 1] = Aw[size_x * (size_y + 2) + j];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_right, r_buf_left)
        MPI_Sendrecv(s_buf_right, size_y, MPI_DOUBLE, rank_send, TAG_X,
                    r_buf_left, size_y, MPI_DOUBLE, rank_recv, TAG_X,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            Aw[0 * (size_y + 2) + j] = r_buf_left[j - 1];
            // c++;
        }}
    }
    // the left process: send
    else if (coords[0] == 0 && coords[0] != proc_number[0] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_right[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            s_buf_right[j - 1] = Aw[size_x * (size_y + 2) + j];
            // c++;
        }}
        #pragma acc host_data use_device(s_buf_right)
        MPI_Send(s_buf_right, size_y, MPI_DOUBLE,
        rank_send, TAG_X, cart_comm);

    }
    // the right process: recv
    else if (coords[0] != 0 && coords[0] == proc_number[0] - 1){
        #pragma acc host_data use_device(r_buf_left)
        MPI_Recv(r_buf_left, size_y, MPI_DOUBLE,
        rank_recv, TAG_X, cart_comm, &status);

        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_left[0:size_y], B[0:_size])
        {
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            Aw[0 * (size_y + 2) + j] = r_buf_left[j - 1];
            // c++;
        }}
    }

    // along y to the low -> dim = 1, disp = -1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 1, -1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[1] != 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_down[i - 1] = Aw[i * (size_y + 2) + 1];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag, 
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_down, r_buf_up)
        MPI_Sendrecv(s_buf_down, size_x, MPI_DOUBLE, rank_send, TAG_Y,
                    r_buf_up, size_x, MPI_DOUBLE, rank_recv, TAG_Y,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            Aw[i * (size_y + 2) + (size_y + 1)] = r_buf_up[i - 1];
            // c++;
        }}
    }
    // the lower process: recv
    else if (coords[1] == 0 && coords[1] != proc_number[1] - 1){
        #pragma acc host_data use_device(r_buf_up)
        MPI_Recv(r_buf_up, size_x, MPI_DOUBLE,
        rank_recv, TAG_Y, cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            Aw[i * (size_y + 2) + (size_y + 1)] = r_buf_up[i - 1];
            // c++;
        }}
    }
    // the upper process: send
    else if (coords[1] != 0 && coords[1] == proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_down[i - 1] = Aw[i * (size_y + 2)  + 1];
            // c++;
        }}

        #pragma acc host_data use_device(s_buf_down)
        MPI_Send(s_buf_down, size_x, MPI_DOUBLE,
        rank_send, TAG_Y, cart_comm);
    }


    // along y to the up -> dim = 1, disp = 1
    // (communicator, direction, disp, source, dest)
    MPI_Cart_shift(cart_comm, 1, 1, &rank_recv, &rank_send);
    // the inner processes: send & recv
    if (coords[1] != 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_up[i - 1] = Aw[i * (size_y + 2) + size_y];
            // c++;
        }}
        /* (sendbuf, sendcount, sendtype, dest, sendtag,
            recvbuf, recvcount, recvtype, source, recvtag,
            comm, status) */
        #pragma acc host_data use_device(s_buf_up, r_buf_down)
        MPI_Sendrecv(s_buf_up, size_x, MPI_DOUBLE, rank_send, TAG_Y,
                    r_buf_down, size_x, MPI_DOUBLE, rank_recv, TAG_Y,
                    cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            Aw[i * (size_y + 2) + 0 ] = r_buf_down[i - 1];
            // c++;
        }}
    }
    // the lower process: send
    else if (coords[1] == 0 && coords[1] != proc_number[1] - 1){
        // generate send_buffer
        // c = 0;
        #pragma acc kernels present(s_buf_up[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            s_buf_up[i - 1] = Aw[i * (size_y + 2) + size_y];
            // c++;
        }}
        #pragma acc host_data use_device(s_buf_up)
        MPI_Send(s_buf_up, size_x, MPI_DOUBLE,
        rank_send, TAG_Y, cart_comm);
    }
    // the upper process: recv
    else if (coords[1] != 0 && coords[1] == proc_number[1] - 1){
        #pragma acc host_data use_device(r_buf_down)
        MPI_Recv(r_buf_down, size_x, MPI_DOUBLE,
        rank_recv, TAG_Y, cart_comm, &status);
        // store recv_buffer
        // c = 0;
        #pragma acc kernels present(r_buf_down[0:size_x], B[0:_size])
        {
        #pragma acc loop independent
        for (i = 1; i <= size_x; i++){
            Aw[i * (size_y + 2) + 0] = r_buf_down[i - 1];
            // c++;
        }}
    }

/* .................................................................................................
........................END EXCHANGE VECTOR Aw ............................................................
....................................................................................................*/
    
        vector_diff(r, Aw, B, size_x, size_y);
        A_vec_mult(Ar, r, M, N, size_x, size_y, i_x, j_y, h1, h2);

        // host vars
        tau_numerator_local = _inner_product(Ar, r, size_x, size_y, i_x, j_y, M, N, h1, h2);
        tau_denominator_local = _inner_product(Ar, Ar, size_x, size_y, i_x, j_y, M, N, h1, h2);

        // #pragma acc update self(tau_numerator_local, tau_denominator_local)

        // (input data, output data, data size, data type, operation type, communicator)
        #pragma acc update device(tau_numerator_local, tau_denominator_local)
        #pragma acc host_data use_device(tau_numerator_local, tau_numerator_global, tau_denominator_local, tau_denominator_global, tau_global)
        {
        MPI_Allreduce(&tau_numerator_local, &tau_numerator_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&tau_denominator_local, &tau_denominator_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tau_global = tau_numerator_global / tau_denominator_global;
        }
        
        // if (rank == 0) cout << "rank: " << rank << " tau_numerator_global: " << tau_numerator_global 
        // << " tau_denominator_global: " << tau_denominator_global << " tau_global: " << tau_global <<endl;

        // #pragma acc update device(tau_global)

        // update points
        // int i, j;
            //update points w 
            #pragma acc kernels present(w[0:_size], tau_global, r[0:_size])
            {
            #pragma acc loop independent
            for (i = 1; i <= size_x; i++)
                #pragma acc loop independent
                for (j = 1; j <= size_y; j++){
                    // update points w
                    w[i * (size_y + 2) + j] = w[i * (size_y + 2) + j] - tau_global * r[i * (size_y + 2) + j];
                    // calculate diff
                    diff_w_and_w_pr[i * (size_y + 2) + j] = w[i * (size_y + 2) + j] - w_pr[i * (size_y + 2) + j];
                    // store current w
                    w_pr[i * (size_y + 2) + j] = w[i * (size_y + 2) + j];
                }
            }

            diff_local = _inner_product(diff_w_and_w_pr, diff_w_and_w_pr, size_x, size_y, i_x, j_y, M, N, h1, h2);
        

            // calculate overall difference 
            // (input data, output data, data size, data type, operation type, communicator)
            #pragma acc update device(diff_local)
            #pragma acc host_data use_device(diff_local, diff)
            {
            MPI_Allreduce(&diff_local, &diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            diff = sqrt(diff);
            }

            #pragma acc update host(tau_global, diff)
        
            if (rank == 0 && iter % PRINT_FREQ == 0) cout << "iter: " << iter << ", tau: " << tau_global << ", err_norm: " << diff << "\n";
            // if (iter == 100) break;

        } while (diff > eps);

        // make barrier and wait for sync
        MPI_Barrier(MPI_COMM_WORLD);

        

        if (rank == 0){
            cout << "Finished !!!" << endl;
            cout << "total_iter: " << iter << ", final_tau: "<< tau_global <<", final_err_norm: " << diff << "\n";
            cout << "Record the results.." << "\n";
        }

    _size = (size_x + 2) * (size_y + 2);
    #pragma acc exit data delete(Aw[0:_size], Ar[0:_size], B[0:_size], r[0:_size], w[0:_size], w_pr[0:_size], diff_w_and_w_pr[0:_size])
    #pragma acc exit data delete(r_buf_up[0:size_x], r_buf_down[0:size_x], r_buf_left[0:size_y], r_buf_right[0:size_y])
    #pragma acc exit data delete(s_buf_up[0:size_x], s_buf_down[0:size_x], s_buf_left[0:size_y], s_buf_right[0:size_y])
    #pragma acc exit data delete(diff_local, tau_numerator_local, tau_denominator_local, tau_global)
    #pragma acc exit data delete(M, N, size_x, size_y, i_x, j_y, h1, h2)
    // #pragma acc exit data delete(this)

    delete[] w;
    delete[] w_pr;
    delete[] B;
    delete[] r;
    delete[] Ar;
    delete[] Aw;
    delete[] diff_w_and_w_pr;

    delete[] r_buf_up;
    delete[] r_buf_down;
    delete[] r_buf_left;
    delete[] r_buf_right;
    delete[] s_buf_up;
    delete[] s_buf_down;
    delete[] s_buf_left;
    delete[] s_buf_right;

    // P.solve();
    double end = MPI_Wtime();

    if (rank == 0) cout << "The total running time: "<< end - start << endl;
    MPI_Finalize();
    
    return 0;
}