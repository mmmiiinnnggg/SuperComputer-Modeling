
// An non-optimized version of sparse implementation

#include <math.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <ctime>

#define A1 0
#define A2 4
#define B1 0
#define B2 3
#define PRINT_FREQ 1000
#define EPS 1e-6
#define MAXSIZE 1000000

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

double rho_i(int i, int M, int N)
{
    if (i == 0 || i == M)
        return 0.5;
    else
        return 1;
}

double rho_j(int j, int M, int N)
{
    if (j == 0 || j == N)
        return 0.5;
    else
        return 1;
}

// difference of two vectors
void vector_diff(double* diff, double *w1, double *w2, int M, int N){
    for(int i = 0; i <= M; i++){
        for(int j = 0; j <= N; j++){
            diff[i * (N + 1) + j] = w1[i * (N + 1) + j] - w2[i * (N + 1) + j];
        }
    }
}

// inner product of two 1-d vectors
double _inner_product(double *w0, double *w1, int M, int N, double h1, double h2)
{
    double res = 0;

    for (int i = 0; i <= M; i++)
    {
        for (int j = 0; j <= N; j++)
        {
            res += rho_i(i, M, N) * rho_j(j, M, N) * w0[i * (N + 1) + j] * w1[i * (N + 1) + j];
        }
    }
    return res * h1 * h2;
}

// norm of 1-d vector
double norm(double *w, int M, int N, double h1, double h2)
{
    return sqrt(_inner_product(w, w, M, N, h1, h2));
} 

// initialization of vector B in the linear system 
void init_B(double* B, int M, int N, double h1, double h2){
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= N; j++){
            // left bound
            if (i == 0 && j >= 1 && j <= N - 1) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1) * left_bound(B1 + j * h2);
            // right bound
            else if (i == M && j >= 1 && j <= N - 1) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1) * right_bound(B1 + j * h2);
            // lower bound 
            else if (j == 0 && i >= 1 && i <= M - 1) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h2) * lower_bound(A1 + i * h1);
            // upper bound 
            else if (j == N && i >= 1 && i <= M - 1) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h2) * upper_bound(A1 + i * h1);

            // the corner values aren't that important...
            // left-lower corner
            else if (i == 0 && j == 0) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * (left_bound(B1 + j * h2) + lower_bound(A1 + i * h1)) / 2.0;
            // else if (i == 0 && j == 0) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * lower_bound(A1 + i * h1);
            // left-upper corner
            else if (i == 0 && j == N) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * (left_bound(B1 + j * h2) + upper_bound(A1 + i * h1)) / 2.0;
            // else if (i == 0 && j == N) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * upper_bound(A1 + i * h1);
            // right-lower corner
            else if (i == M && j == 0) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * (right_bound(B1 + j * h2) + lower_bound(A1 + i * h1)) / 2.0;
            // else if (i == M && j == 0) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * lower_bound(A1 + i * h1);
            // right-upper corner
            else if (i == M && j == N) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * (right_bound(B1 + j * h2) + upper_bound(A1 + i * h1)) / 2.0;
            // else if (i == M && j == N) B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * upper_bound(A1 + i * h1);

            // the center points
            else B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2);
        }
}

// triplet structure: index and value
typedef struct Triple{
    int i = -1, j;
    double e;
}Triple;


// Sparse Matrix structure
struct SMatrix{
    SMatrix(int _M, int _N);
    void set_value(int i_eq, int j_eq, int k, int i, int j, double value);
    void A_vector_multi(double *Aw, double *w, int M, int N, double h1, double h2);
    void PrintA();

private:
    int M, N;
    double h1, h2;
    Triple** data;
};

// set value of index i,j with value at k position of data[(i_eq, j_eq)]
void SMatrix::set_value(int i_eq, int j_eq, int k, int i, int j, double value){
        data[i_eq * (N + 1) + j_eq][k].i = i;
        data[i_eq * (N + 1) + j_eq][k].j = j;
        data[i_eq * (N + 1) + j_eq][k].e = value;
}

// constructor 
SMatrix::SMatrix(int _M, int _N){
    M = _M; N = _N;
    h1 = (double)(A2 - A1) / M;
    h2 = (double)(B2 - B1) / N;

    data = new Triple* [(M + 1) * (N + 1)];
    for(int i = 0; i < (M + 1) * (N + 1); i++){
        data[i] = new Triple[5];
    }

    // init A
    for (int i = 0; i <= M; i++){
        for (int j = 0; j <= N; j++){
            // left bound
            if (i == 0 && j >= 1 && j <= N - 1){
                set_value(i, j, 0, 0, j, 2 / (h1 * h1) * k(A1 + (1 - 0.5) * h1, B1 + j * h2)
                            + q(A1 + i * h1, B1 + j * h2) + 2 / h1
                            + 1 / (h2 * h2) * k(A1 + 0 * h1, B1 + (j + 0.5) * h2)
                            + 1 / (h2 * h2) * k(A1 + 0 * h1, B1 + (j - 0.5) * h2));
                set_value(i, j, 1, 1, j, - 2 / (h1 * h1) * k(A1 + (1 - 0.5) * h1, B1 + j * h2));
                set_value(i, j, 2, 0, j + 1, - 1 / (h2 * h2) * k(A1 + 0 * h1, B1 + (j + 0.5) * h2));
                set_value(i, j, 3, 0, j - 1, - 1 / (h2 * h2) * k(A1 + 0 * h1, B1 + (j - 0.5) * h2));
            } 
            // right bound
            else if (i == M && j >= 1 && j <= N - 1){
                set_value(i, j, 0, M, j, 2 / (h1 * h1) * k(A1 + (M - 0.5) * h1, B1 + j * h2)
                                + q(A1 + i * h1, B1 + j * h2) + 2 / h1
                                + 1 / (h2 * h2) * k(A1 + M * h1, B1 + (j + 0.5) * h2)
                                + 1 / (h2 * h2) * k(A1 + M * h1, B1 + (j - 0.5) * h2));
                set_value(i, j, 1, M - 1, j, - 2 / (h1 * h1) * k(A1 + (M - 0.5) * h1, B1 + j * h2));
                set_value(i, j, 2, M, j + 1, - 1 / (h2 * h2) * k(A1 + M * h1, B1 + (j + 0.5) * h2));
                set_value(i, j, 3, M, j - 1, - 1 / (h2 * h2) * k(A1 + M * h1, B1 + (j - 0.5) * h2));
            }
            // lower bound 
            else if (j == 0 && i >= 1 && i <= M - 1){
                set_value(i, j, 0, i, 0, 2 / (h2 * h2) * k(A1 + i * h1, B1 + (1 - 0.5) * h2)
                                + q(A1 + i * h1, B1 + j * h2)
                                + 1 / (h1 * h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2)
                                + 1 / (h1 * h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2));
                set_value(i, j, 1, i, 1, - 2 / (h2 * h2) * k(A1 + i * h1, B1 + (1 - 0.5) * h2));
                set_value(i, j, 2, i + 1, 0, - 1 / (h1 * h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2));
                set_value(i, j, 3, i - 1, 0, - 1 / (h1 * h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2));
            }
            // upper bound 
            else if (j == N && i >= 1 && i <= M - 1){
                set_value(i, j, 0, i, N, 2 / (h2 * h2) * k(A1 + i * h1, B1 + (N - 0.5) * h2)
                                + q(A1 + i * h1, B1 + j * h2)
                                + 1 / (h1 * h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2)
                                + 1 / (h1 * h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2));
                set_value(i, j, 1, i, N - 1, - 2 / (h2 * h2) * k(A1 + i * h1, B1 + (N - 0.5) * h2));
                set_value(i, j, 2, i + 1, N, - 1 / (h1 * h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2));
                set_value(i, j, 3, i - 1, N, - 1 / (h1 * h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2));
            }

            // left-lower corner
            else if (i == 0 && j == 0){
                set_value(i, j, 0, 0, 0, 2 / (h1 * h1) * k(A1 + (1 - 0.5) * h1, B1 + 0 * h2)
                                + 2 / (h2 * h2) * k(A1 + 0 * h1, B1 + (1 - 0.5) * h2)
                                + q(A1 + i * h1, B1 + j * h2) + 2 / h1);
                set_value(i, j, 1, 0, 1, - 2 / (h2 * h2) * k(A1 + 0 * h1, B1 + (1 - 0.5) * h2));
                set_value(i, j, 2, 1, 0, - 2 / (h1 * h1) * k(A1 + (1 - 0.5) * h1, B1 + 0 * h2));
            }
            // left-upper corner
            else if (i == 0 && j == N){
                set_value(i, j, 0, 0, N, + 2 / (h1 * h1) * k(A1 + (1 - 0.5) * h1, B1 + N * h2)
                                + 2 / (h2 * h2) * k(A1 + 0 * h1, B1 + (N - 0.5) * h2)
                                + q(A1 + i * h1, B1 + j * h2) + 2 / h1);
                set_value(i, j, 1, 1, N, - 2 / (h1 * h1) * k(A1 + (1 - 0.5) * h1, B1 + N * h2));
                set_value(i, j, 2, 0, N - 1, - 2 / (h2 * h2) * k(A1 + 0 * h1, B1 + (N - 0.5) * h2));
            } 
            // right-lower corner
            else if (i == M && j == 0){
                set_value(i, j, 0, M, 0, 2 / (h1 * h1) * k(A1 + (M - 0.5) * h1, B1 + 0 * h2)
                                + 2 / (h2 * h2) * k(A1 + M * h1, B1 + (1 - 0.5) * h2)
                                + q(A1 + i * h1, B1 + j * h2) + 2 / h1);
                set_value(i, j, 1, M - 1, 0, - 2 / (h1 * h1) * k(A1 + (M - 0.5) * h1, B1 + 0 * h2));
                set_value(i, j, 2, M, 1, - 2 / (h2 * h2) * k(A1 + M * h1, B1 + (1 - 0.5) * h2));
            }
            // right-upper corner
            else if (i == M && j == N){
                set_value(i, j, 0, M, N, 2 / (h1 * h1) * k(A1 + (M - 0.5) * h1, B1 + N * h2)
                                + 2 / (h2 * h2) * k(A1 + M * h1, B1 + (N - 0.5) * h2)
                                + q(A1 + i * h1, B1 + j * h2) + 2 / h1);
                set_value(i, j, 1, M - 1, N, - 2 / (h1 * h1) * k(A1 + (M - 0.5) * h1, B1 + N * h2));
                set_value(i, j, 2, M, N - 1, - 2 / (h2 * h2) * k(A1 + M * h1, B1 + (N - 0.5) * h2)); 
            }

            // the center points
            else {
                set_value(i, j, 0, i, j, 1 / (h1 * h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2)
                                + 1 / (h1 * h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2)
                                + 1 / (h2 * h2) * k(A1 + i * h1, B1 + (j + 0.5) * h2)
                                + 1 / (h2 * h2) * k(A1 + i * h1, B1 + (j - 0.5) * h2)
                                + q(A1 + i * h1, B1 + j * h2));
                set_value(i, j, 1, i - 1, j, - 1 / (h1 * h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2));
                set_value(i, j, 2, i + 1, j, - 1 / (h1 * h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2));
                set_value(i, j, 3, i, j + 1, - 1 / (h2 * h2) * k(A1 + i * h1, B1 + (j + 0.5) * h2));
                set_value(i, j, 4, i, j - 1, - 1 / (h2 * h2) * k(A1 + i * h1, B1 + (j - 0.5) * h2));
            }
        }
    }
}


// perform the matrix-vector multiplication
void SMatrix::A_vector_multi(double *Aw, double *w, int M, int N, double h1, double h2){
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= N; j++){
            Aw[i * (N + 1) + j] = data[i * (N + 1) + j][0].e * w[data[i * (N + 1) + j][0].i * (N + 1) + data[i * (N + 1) + j][0].j] 
                                + data[i * (N + 1) + j][1].e * w[data[i * (N + 1) + j][1].i * (N + 1) + data[i * (N + 1) + j][1].j]
                                + data[i * (N + 1) + j][2].e * w[data[i * (N + 1) + j][2].i * (N + 1) + data[i * (N + 1) + j][2].j];
            if (data[i * (N + 1) + j][3].i != -1){
                Aw[i * (N + 1) + j] += data[i * (N + 1) + j][3].e * w[data[i * (N + 1) + j][3].i * (N + 1) + data[i * (N + 1) + j][3].j];
            }
            if (data[i * (N + 1) + j][4].i != -1){
                Aw[i * (N + 1) + j] += data[i * (N + 1) + j][4].e * w[data[i * (N + 1) + j][4].i * (N + 1) + data[i * (N + 1) + j][4].j];
            } 
        }
}


// print matrix A 
void SMatrix::PrintA(){
    for(int i = 0; i < (M + 1) * (N + 1); i ++){
        for (int j = 0; j < 5; j ++){
            if (data[i][j].i != -1){
                cout << data[i][j].i << " " << data[i][j].j << " " << data[i][j].e << " ";
            }
        }
        cout << endl;
    }
}


// solve the linear system using minimal discrepancies method
void solve(int M, int N, double h1, double h2){
    double *Aw = new double [(M + 1) * (N + 1)];
    double *Ar = new double [(M + 1) * (N + 1)];
    double *B = new double [(M + 1) * (N + 1)];
    double *r = new double [(M + 1) * (N + 1)];
    double *w = new double [(M + 1) * (N + 1)];
    double *w_pr = new double [(M + 1) * (N + 1)];
    double *diff_w_and_w_pr = new double [(M + 1) * (N + 1)];
    double tau;

    cout << "Start solving..." << "\n";

    int iter = 0;
    bool is_not_solved = true;

    clock_t start = clock();
    
    // init w_pr
    for(int i = 0; i <= M; i++)
        for(int j = 0; j <= N; j++)
            w_pr[i * (N + 1) + j] = 0;

    // init w_0 
    for(int i = 0; i <= M; i++)
        for(int j = 0; j <= N; j++)
            w[i * (N + 1) + j] = 0;
    
    // init A and B
    SMatrix A(M, N);
    init_B(B, M, N, h1, h2);

    cout << "Init A and B finished!!!" << endl;

    while(is_not_solved){
        iter ++;
        A.A_vector_multi(Aw, w, M, N, h1, h2);
        vector_diff(r, Aw, B, M, N);
        A.A_vector_multi(Ar, r, M, N, h1, h2);
        tau = _inner_product(Ar, r, M, N, h1, h2) / _inner_product(Ar, Ar, M, N, h1, h2); 

        //update w
        for(int i = 0; i <= M; i++)
            for(int j = 0; j <= N; j++)
                w[i * (N + 1) + j] = w[i * (N + 1) + j] - tau * r[i * (N + 1) + j];

        // calculate difference
        vector_diff(diff_w_and_w_pr, w, w_pr, M, N);

        // store current w
        for(int i = 0; i <= M; i++)
            for(int j = 0; j <= N; j++)
                w_pr[i * (N + 1) + j] = w[i * (N + 1) + j];
        
        if (norm(diff_w_and_w_pr, M, N, h1, h2) < EPS) is_not_solved = false;

        if (iter % PRINT_FREQ == 0){
            cout << "iter: " << iter << ", tau: "<< tau <<", err_norm: " << norm(diff_w_and_w_pr, M, N, h1, h2) << "\n";
        }
    }

    cout << "Finished!!!" << "\n";
    cout << "total_iter: " << iter << ", final_tau: "<< tau <<", final_err_norm: " << norm(diff_w_and_w_pr, M, N, h1, h2) << "\n";
    clock_t end = clock();
    cout << "The total running time is: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
    // A.PrintA();

    // cout << "Record the results..." << "\n";
    // // record the results: two columns, the real values and the approximate values
    // string file_name = "./serial_results/res_" + to_string(M) + "_" + to_string(N) + ".txt";
    // ofstream res_file(file_name, ios::out);
    // res_file << M << "," << N << "\n";
    // for(int i = 0; i <= M; i++)
    // {
    //     for(int j = 0; j <= N; j++)
    //     {
    //         res_file << u((A1 + i * h1), (B1 + j * h2)) << ',' << w[i * (N + 1) + j] << "\n";
    //     }
    // }
    // res_file.close();

    cout << "Free memory...\n";  
    delete[] w;
    delete[] w_pr;
    delete[] r;
    delete[] Ar;
    delete[] Aw;
    delete[] B;
    delete[] diff_w_and_w_pr;

    cout << "PrintA..." << endl;
    A.PrintA();
}

int main(int argc, char* argv[]){

    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    double h1 = (double)(A2 - A1) / M;
    double h2 = (double)(B2 - B1) / N;
    
    solve(M, N, h1, h2);
    
    return 0;
}
