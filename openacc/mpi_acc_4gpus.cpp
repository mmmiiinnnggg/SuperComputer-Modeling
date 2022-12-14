#include <mpi.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <openacc.h>

#define A1 0
#define A2 4
#define B1 0
#define B2 3
#define PRINT_FREQ 1000

#define TAG_1 666
#define TAG_2 777
#define TAG_3 888
#define TAG_4 999

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

    int i, j;
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
}

// inner product of two 1-d vectors
double _inner_product(double *w0, double *w1, int size_x, int size_y, int i_x, int j_y, int M, int N, double h1, double h2)
{
    double res = 0;

    // do not count padding points
    int i, j, _i, _j;
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
    int _size = (size_x + 2) * (size_y + 2);
    #pragma acc kernels present(B[0:_size])
    {
    #pragma acc loop independent
    for (i = 1; i <= size_x; i++)
        #pragma acc loop independent
        for (j = 1; j <= size_y; j++){
            
            _i = i_x + i - 1;
            _j = j_y + j - 1;

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

}

// conduct matrix-vector multiplication
void A_vec_mult(double *Av_res, double *w, int M, int N, int size_x, int size_y, int i_x, int j_y, double h1, double h2){ 

    int i, j, _i, _j;
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
}


struct Process
{
    // constructor 
    Process(int _M, int _N, double _eps);
    // solve the system using minimal discrepancies method
    void solve();
    // number of processors and processor rank
    int size, rank;
    
private:
    MPI_Comm cart_comm;
    MPI_Status status;
    MPI_Request request;
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
    double diff_local, tau_numerator_local, tau_denominator_local, tau_global;
    
    // overall sizes
    int M, N;
    // steps of approximation
    double h1, h2;

    // the send & recv buffers 
    // double *s_buf_up, *s_buf_down, *s_buf_left, *s_buf_right;
    // double *r_buf_up, *r_buf_down, *r_buf_left, *r_buf_right;

    // the 4 proc buffers
    double *w_l_bound, *w_r_bound, *w_l_buffer, *w_r_buffer;
    double *w_down_bound, *w_up_bound, *w_down_buffer, *w_up_buffer;

    // the intermidate results
    double *Aw, *Ar, *B, *r;
    double *w, *w_pr, *diff_w_and_w_pr;

    double eps;

    // void update_host();
    // void update_device();

    void create_communicator();
    void init_processor_config();
    void fill_data();
    // void exchange_data(double *_w);
    void exchange_data_4proc(double *_w);
    // one iteration for solving the linear system
    double solve_iteration();
};

// Process constructor
Process::Process(int _M, int _N, double _eps){
    // get M, N and eps
    M = _M, N = _N;
    h1 = (double)(A2 - A1) / M;
    h2 = (double)(B2 - B1) / N; 
    eps = _eps;

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
    // for 1-gpu usecase:
    // acc_set_device_num(0, acc_device_nvidia);
}


// create communicator for Cartisan coords
void Process::create_communicator(){

    // boolean array to define periodicy of each dimension
    int Periods[2] = {0, 0};

    // (world communicator, num of dims, dim_size, periodicy for each dimention, no reorder, cart_comm)
    MPI_Cart_create(MPI_COMM_WORLD, 2, proc_number, Periods, 0, &cart_comm);

    // (cart_comm, the given rank, num of dims, the corresponding coords)
    MPI_Cart_coords(cart_comm, rank, 2, coords);
}

// init size, shift, block, bounds and buffers
void Process::init_processor_config(){

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

    // init buffers
    w_l_bound = new double [size_y];
    w_r_bound = new double [size_y];
    w_l_buffer = new double [size_y];
    w_r_buffer = new double [size_y];
    w_down_bound = new double [size_x];
    w_up_bound = new double [size_x];
    w_down_buffer = new double [size_x];
    w_up_buffer = new double [size_x];

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
    #pragma acc enter data copyin(this)
    #pragma acc enter data copyin(diff_local, tau_numerator_local, tau_denominator_local, tau_global)
    #pragma acc enter data copyin(M, N, size_x, size_y, i_x, j_y, h1, h2)
    #pragma acc enter data create(Aw[0:_size], Ar[0:_size], B[0:_size], r[0:_size], w[0:_size], w_pr[0:_size], diff_w_and_w_pr[0:_size])
    #pragma acc enter data create(w_l_bound[0:size_y], w_r_bound[0:size_y], w_l_buffer[0:size_y], w_r_buffer[0:size_y])
    #pragma acc enter data create(w_down_bound[0:size_x], w_up_bound[0:size_x], w_down_buffer[0:size_x], w_up_buffer[0:size_x])

}

// fill up data for B, and allocate memory
void Process::fill_data(){
    int i, j;

    // init w_pr
    #pragma acc kernels present(w_pr[0:_size], w[0:_size])
    {
    #pragma acc loop independent
    for(i = 0; i <= size_x + 1; i++)
        #pragma acc loop independent
        for(j = 0; j <= size_y + 1; j++){
            w_pr[i * (size_y + 2) + j] = 2.5;
            w[i * (size_y + 2) + j] = 2.5;
        }
    }

    init_B(B, M, N, size_x, size_y, i_x, j_y, h1, h2);

}

void Process::exchange_data_4proc(double *_w){
    // with 4 process-gpus

    //  _____________________________
    // |              |              |
    // | rank 1, (0,1)| rank 3, (1,1)|           
    // |______________|______________|
    // |              |              |
    // | rank 0, (0,0)| rank 2, (1,0)|
    // |______________|______________|

    // along x axis 
    // form data to send to cpu, left process
    if (coords[0] == 0){
        #pragma acc kernels present(_w[0:_size], w_r_bound[0:size_y])
        {
        #pragma acc loop independent
        for (int j = 1; j <= size_y; j++){
            w_r_bound[j - 1] = _w[size_x * (size_y + 2) + j];
        }
        }
    }

    else{ // right process
        #pragma acc kernels present(_w[0:_size], w_l_bound[0:size_y])
        {
        #pragma acc loop independent
        for (int i = 1; i <= size_y; i++){
            w_l_bound[i - 1] = _w[1 * (size_y + 2) + i];
        }
        }
    }

    // left process rank 0 send right bound values
    if (rank < 2){
        // give back values to cpu for sending 
        #pragma acc update self(w_r_bound[0:size_y])
        // send right bound data 
        MPI_Isend(w_r_bound, size_y, MPI_DOUBLE, rank + 2, TAG_1 + rank, MPI_COMM_WORLD, &request);
    }

    // right process rank 1 recv right bound values to left buffer
    else {
        // right process recv values and put in left buffer
        MPI_Irecv(w_l_buffer, size_y, MPI_DOUBLE, rank - 2, TAG_1 + rank - 2, MPI_COMM_WORLD, &request);
        // update device values with recv-ed values
        #pragma acc update device(w_l_buffer[0:size_y])
    }

    MPI_Wait(&request, &status);

    if (rank < 2){
        // left process recv values and put in right buffer
        MPI_Irecv(w_r_buffer, size_y, MPI_DOUBLE, rank + 2, TAG_2 + rank, MPI_COMM_WORLD, &request);
        // update device data copy
        #pragma acc update device(w_r_buffer[0:size_y]) 
    }

    else {
        // give back values to cpu for sending 
        #pragma acc update self(w_l_bound[0:size_y])
        // right process send left bound data
        MPI_Isend(w_l_bound, size_y, MPI_DOUBLE, rank - 2, TAG_2 + rank - 2, MPI_COMM_WORLD, &request);
        
    }

    MPI_Wait(&request, &status);

    // update device w buffer values, left process
    if (rank < 2){
        #pragma acc kernels present(_w[0:_size], w_r_buffer[0:size_y])
        {
        #pragma acc loop independent
        for (int i = 1; i <= size_y; i++){
            _w[(size_x + 1) * (size_y + 2) + i] = w_r_buffer[i - 1];
        }
        }
    }

    else{ // right process
        #pragma acc kernels present(_w[0:_size], w_l_buffer[0:size_y])
        {
        #pragma acc loop independent
        for (int i = 1; i <= size_y; i++){
            _w[0 * (size_y + 2) + i] = w_l_buffer[i - 1];
        }
        }
    }


    // along y axis
    // form data to send to cpu, down process
    if (coords[1] == 0){
        #pragma acc kernels present(_w[0:_size], w_up_bound[0:size_x])
        {
        #pragma acc loop independent
        for (int i = 1; i <= size_x; i++){
            w_up_bound[i - 1] = _w[i * (size_y + 2) + size_y];
        }
        }
    }

    else{ // up process
        #pragma acc kernels present(_w[0:_size], w_down_bound[0:size_x])
        {
        #pragma acc loop independent
        for (int i = 1; i <= size_x; i++){
            w_down_bound[i - 1] = _w[i * (size_x + 2) + 1];
        }
        }
    }

    // down process send to up process bound values
    if (rank % 2 == 0){
        // give back values to cpu for sending 
        #pragma acc update self(w_up_bound[0:size_x])
        // send right bound data 
        MPI_Isend(w_up_bound, size_x, MPI_DOUBLE, rank + 1, TAG_3 + rank, MPI_COMM_WORLD, &request);
    }

    // up process rank recv up bound values to down buffer
    else {
        // right process recv values and put in down buffer
        MPI_Irecv(w_down_buffer, size_x, MPI_DOUBLE, rank - 1, TAG_3 + rank - 1, MPI_COMM_WORLD, &request);
        // update device values with recv-ed values
        #pragma acc update device(w_down_buffer[0:size_x])
    }

    MPI_Wait(&request, &status);

    if (rank % 2 == 0){
        // down process recv values and put in up buffer
        MPI_Irecv(w_up_buffer, size_x, MPI_DOUBLE, rank + 1, TAG_4 + rank, MPI_COMM_WORLD, &request);
        // update device data copy
        #pragma acc update device(w_up_buffer[0:size_x]) 
    }

    else {
        // give back values to cpu for sending 
        #pragma acc update self(w_down_bound[0:size_x])
        // up process send down bound data
        MPI_Isend(w_down_bound, size_x, MPI_DOUBLE, rank - 1, TAG_4 + rank - 1, MPI_COMM_WORLD, &request);
        
    }

    MPI_Wait(&request, &status);

    // update device w buffer values, down process
    if (rank % 2 == 0){
        #pragma acc kernels present(_w[0:_size], w_up_buffer[0:size_x])
        {
        #pragma acc loop independent
        for (int i = 1; i <= size_x; i++){
            _w[i * (size_y + 2) + (size_y + 1)] = w_up_buffer[i - 1];
        }
        }
    }

    else{ // up process
        #pragma acc kernels present(_w[0:_size], w_down_buffer[0:size_x])
        {
        #pragma acc loop independent
        for (int i = 1; i <= size_x; i++){
            _w[i * (size_y + 2) + 0] = w_down_buffer[i - 1];
        }
        }
    }
    
}

double Process::solve_iteration(){

    diff_local = 0;
    // #pragma acc update device(diff_local)
    double tau_numerator_global, tau_denominator_global;

    //sync padding values
    exchange_data_4proc(w);

    A_vec_mult(Aw, w, M, N, size_x, size_y, i_x, j_y, h1, h2);

    //sync padding values
    exchange_data_4proc(Aw);
    
    vector_diff(r, Aw, B, size_x, size_y);
    A_vec_mult(Ar, r, M, N, size_x, size_y, i_x, j_y, h1, h2);

    // host vars
    tau_numerator_local = _inner_product(Ar, r, size_x, size_y, i_x, j_y, M, N, h1, h2);
    tau_denominator_local = _inner_product(Ar, Ar, size_x, size_y, i_x, j_y, M, N, h1, h2);

    // (input data, output data, data size, data type, operation type, communicator)
    MPI_Allreduce(&tau_numerator_local, &tau_numerator_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&tau_denominator_local, &tau_denominator_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    tau_global = tau_numerator_global / tau_denominator_global;

    #pragma acc update device(tau_global)

    // update points
    int i, j;

        #pragma acc kernels present(w[0:_size], w_pr[0:_size], tau_global, r[0:_size], diff_w_and_w_pr[0:_size])
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

    return diff_local;
}


void Process::solve(){

    if (rank == 0){
        cout << "How many GPU devices do we have?" << endl;
        acc_device_t dev = acc_get_device_type();
        int device_num = acc_get_num_devices(dev);
        cout << "Number of device : " << device_num << endl;
        if (device_num != 4){
            cout << "The device num is not 4, please check! " << endl;
            return;
        }
    }

    create_communicator();
    init_processor_config();
    fill_data();
    //sync padding values for B
    exchange_data_4proc(B);

    double diff, diff_local;
    int iter = 0;

    if(rank == 0) cout << "Starting..." << endl;

    do{
        iter++;

        diff_local = solve_iteration();

        // calculate overall difference 
        // (input data, output data, data size, data type, operation type, communicator)
        MPI_Allreduce(&diff_local, &diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        diff = sqrt(diff);
        
        if (rank == 0 && iter % PRINT_FREQ == 0) cout << "iter: " << iter << ", tau: " << tau_global << ", err_norm: " << diff << "\n";

    } while (diff > eps);

    // make barrier and wait for sync
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0){
        cout << "Finished !!!" << endl;
        cout << "total_iter: " << iter << ", final_tau: "<< tau_global <<", final_err_norm: " << diff << "\n";
        cout << "Record the results.." << "\n";
    }

    // free the memory
    _size = (size_x + 2) * (size_y + 2);
    #pragma acc exit data delete(w_down_bound[0:size_x], w_up_bound[0:size_x], w_down_buffer[0:size_x], w_up_buffer[0:size_x])
    #pragma acc exit data delete(w_l_bound[0:size_y], w_r_bound[0:size_y], w_l_buffer[0:size_y], w_r_buffer[0:size_y])
    #pragma acc exit data delete(Aw[0:_size], Ar[0:_size], B[0:_size], r[0:_size], w[0:_size], w_pr[0:_size], diff_w_and_w_pr[0:_size])
    #pragma acc exit data delete(diff_local, tau_numerator_local, tau_denominator_local, tau_global)
    #pragma acc exit data delete(M, N, size_x, size_y, i_x, j_y, h1, h2)
    #pragma acc exit data delete(this)

    delete[] w;
    delete[] w_pr;
    delete[] B;
    delete[] r;
    delete[] Ar;
    delete[] Aw;
    delete[] diff_w_and_w_pr;

}

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);

    double start = MPI_Wtime();
    Process P(atoi(argv[1]), atoi(argv[2]), atof(argv[3]));
    P.solve();
    double end = MPI_Wtime();

    if (P.rank == 0) cout << "The total running time: "<< end - start << endl;
    MPI_Finalize();
    
    return 0;
}
