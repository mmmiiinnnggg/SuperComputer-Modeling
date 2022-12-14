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
#define PRINT_FREQ 100
#define EPS 1e-6

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
    double ratio = h1 / (h1 + h2);
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
            else if (i == 0 && j == 0){
                B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * ((1 - ratio) * left_bound(B1 + j * h2) + ratio * lower_bound(A1 + i * h1));
            } 
            // left-upper corner
            else if (i == 0 && j == N){
                B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * ((1 - ratio) * left_bound(B1 + j * h2) + ratio * upper_bound(A1 + i * h1));
            } 
            // right-lower corner
            else if (i == M && j == 0){
                B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * ((1 - ratio) * right_bound(B1 + j * h2) + ratio * lower_bound(A1 + i * h1));
            } 
            // right-upper corner
            else if (i == M && j == N){
                B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2) + (2 / h1 + 2 / h2 ) * ((1 - ratio) * right_bound(B1 + j * h2) + ratio * upper_bound(A1 + i * h1));
            } 
            

            // the center points
            else B[i * (N + 1) + j] = F(A1 + i * h1, B1 + j * h2);
        }
}


// perform the matrix-vector multiplication
void A_vector_multi(double *Aw, double *w, int M, int N, double h1, double h2){
    for (int i = 0; i <= M; i++)
        for (int j = 0; j <= N; j++){
            // left bound
            if (i == 0 && j >= 1 && j <= N - 1){
                Aw[i * (N + 1) + j] = - (2 / h1 ) * k(A1 + (1 - 0.5) * h1, B1 + j * h2) * (w[1 * (N + 1) + j] - w[0 * (N + 1) + j]) / h1
                    + (q(A1 + i * h1, B1 + j * h2) + 2 / h1 ) * w[0 * (N + 1) + j] 
                    - (1 / h2) * k(A1 + 0 * h1, B1 + (j + 0.5) * h2) * (w[0 * (N + 1) + (j + 1)] - w[0 * (N + 1) + j]) / h2
                    + (1 / h2) * k(A1 + 0 * h1, B1 + (j - 0.5) * h2) * (w[0 * (N + 1) + j] - w[0 * (N + 1) + (j - 1)]) / h2;
            } 
            // right bound
            else if (i == M && j >= 1 && j <= N - 1){
                Aw[i * (N + 1) + j] = (2 / h1 ) * k(A1 + (M - 0.5) * h1, B1 + j * h2) * (w[M * (N + 1) + j] - w[(M - 1) * (N + 1) + j]) / h1
                    + (q(A1 + i * h1, B1 + j * h2) + 2 / h1 ) * w[M * (N + 1) + j] 
                    - (1 / h2) * k(A1 + M * h1, B1 + (j + 0.5) * h2) * (w[M * (N + 1) + (j + 1)] - w[M * (N + 1) + j]) / h2
                    + (1 / h2) * k(A1 + M * h1, B1 + (j - 0.5) * h2) * (w[M * (N + 1) + j] - w[M * (N + 1) + (j - 1)]) / h2;
            }
            // lower bound 
            else if (j == 0 && i >= 1 && i <= M - 1){
                Aw[i * (N + 1) + j] = - (2 / h2 ) * k(A1 + i * h1, B1 + (1 - 0.5) * h2) * (w[i * (N + 1) + 1] - w[i * (N + 1) + 0]) / h2
                    + q(A1 + i * h1, B1 + j * h2) * w[i * (N + 1) + 0] 
                    - (1 / h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2) * (w[(i + 1) * (N + 1) + j] - w[i * (N + 1) + j]) / h1
                    + (1 / h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2) * (w[i * (N + 1) + j] - w[(i - 1) * (N + 1) + j]) / h1;
            }
            // upper bound 
            else if (j == N && i >= 1 && i <= M - 1){
                Aw[i * (N + 1) + j] = (2 / h2 ) * k(A1 + i * h1, B1 + (N - 0.5) * h2) * (w[i * (N + 1) + N] - w[i * (N + 1) + (N - 1)]) / h2
                    + q(A1 + i * h1, B1 + j * h2) * w[i * (N + 1) + N] 
                    - (1 / h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2) * (w[(i + 1) * (N + 1) + j] - w[i * (N + 1) + j]) / h1
                    + (1 / h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2) * (w[i * (N + 1) + j] - w[(i - 1) * (N + 1) + j]) / h1;
            }

            // left-lower corner
            else if (i == 0 && j == 0){
                Aw[i * (N + 1) + j] = - (2 / h1 ) * k(A1 + (1 - 0.5) * h1, B1 + 0 * h2) * (w[1 * (N + 1) + 0] - w[0 * (N + 1) + 0]) / h1
                    - (2 / h2 ) * k(A1 + 0 * h1, B1 + (1 - 0.5) * h2) * (w[0 * (N + 1) + 1] - w[0 * (N + 1) + 0]) / h2
                    + (q(A1 + i * h1, B1 + j * h2) + 2 / h1) * w[i * (N + 1) + j];
            }
            // left-upper corner
            else if (i == 0 && j == N){
                Aw[i * (N + 1) + j] = - (2 / h1 ) * k(A1 + (1 - 0.5) * h1, B1 + N * h2) * (w[1 * (N + 1) + N] - w[0 * (N + 1) + N]) / h1
                    + (2 / h2 ) * k(A1 + 0 * h1, B1 + (N - 0.5) * h2) * (w[0 * (N + 1) + N] - w[0 * (N + 1) + (N - 1)]) / h2
                    + (q(A1 + i * h1, B1 + j * h2) + 2 / h1) * w[i * (N + 1) + j];
            } 
            // right-lower corner
            else if (i == M && j == 0){
                Aw[i * (N + 1) + j] =  (2 / h1 ) * k(A1 + (M - 0.5) * h1, B1 + 0 * h2) * (w[M * (N + 1) + 0] - w[(M - 1) * (N + 1) + 0]) / h1
                    - (2 / h2 ) * k(A1 + M * h1, B1 + (1 - 0.5) * h2) * (w[M * (N + 1) + 1] - w[M * (N + 1) + 0]) / h2
                    + (q(A1 + i * h1, B1 + j * h2) + 2 / h1) * w[i * (N + 1) + j];
            }
            // right-upper corner
            else if (i == M && j == N){
                Aw[i * (N + 1) + j] = (2 / h1 ) * k(A1 + (M - 0.5) * h1, B1 + N * h2) * (w[M * (N + 1) + N] - w[(M - 1) * (N + 1) + N]) / h1
                    + (2 / h2 ) * k(A1 + M * h1, B1 + (N - 0.5) * h2) * (w[M * (N + 1) + N] - w[M * (N + 1) + (N - 1)]) / h2
                    + (q(A1 + i * h1, B1 + j * h2) + 2 / h1) * w[i * (N + 1) + j];
            }

            // the center points
            else Aw[i * (N + 1) + j] = - (1 / h1) * k(A1 + (i + 0.5) * h1, B1 + j * h2) * (w[(i + 1) * (N + 1) + j] - w[i * (N + 1) + j]) / h1
                + (1 / h1) * k(A1 + (i - 0.5) * h1, B1 + j * h2) * (w[i * (N + 1) + j] - w[(i - 1) * (N + 1) + j]) / h1
                - (1 / h2) * k(A1 + i * h1, B1 + (j + 0.5) * h2) * (w[i * (N + 1) + (j + 1)] - w[i * (N + 1) + j]) / h2
                + (1 / h2) * k(A1 + i * h1, B1 + (j - 0.5) * h2) * (w[i * (N + 1) + j] - w[i * (N + 1) + (j - 1)]) / h2
                + q(A1 + i * h1, B1 + j * h2) * w[i * (N + 1) + j];     
        }

}

// Print vector:
void print(double *w, int M, int N){
    cout << "printing..." << endl;
    for(int i = 0; i <= M; i++){
        for(int j = 0; j <= N; j++){
            cout << w[i * (N + 1) + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// solve the linear system using minimal discrepancies method
void solve(int M, int N, double h1, double h2){

    double *Ar = new double [(M + 1) * (N + 1)];
    double *B = new double [(M + 1) * (N + 1)];
    double *r = new double [(M + 1) * (N + 1)];
    double *w = new double [(M + 1) * (N + 1)];

    double *g = new double [(M + 1) * (N + 1)];
    double *g_pr = new double [(M + 1) * (N + 1)];
    double alpha, beta, norm_B, error;

    clock_t start = clock();
    cout << "Start solving..." << "\n";

    int iter = 0;
    bool is_not_solved = true;
    
    // CG Method:
    // g0 = B - Aw0
    // r0 = g0
    // k = 0
    // while norm(gk) > eps norm(b):
    //     alpha_k = norm(gk)^2 / (Ark, rk)
    //     w_k+1 = w_k + alpha_k*rk
    //     g_k+1 = g_k - alpha_kAr_k
    //     beta_k = norm(g_k+1)^2 / norm(g_k)^2
    //     r_k+1 = g_k+1 + beta_k r_k
    //     k = k + 1

    // init w_0 
    for(int i = 0; i <= M; i++)
        for(int j = 0; j <= N; j++)
            w[i * (N + 1) + j] = 0;
    
       
    init_B(B, M, N, h1, h2);

    // init g_pr
    for(int i = 0; i <= M; i++)
        for(int j = 0; j <= N; j++)
            g_pr[i * (N + 1) + j] = 0;
    
    // init g: B - Aw0 = B
    for(int i = 0; i <= M; i++)
        for(int j = 0; j <= N; j++)
            g[i * (N + 1) + j] = B[i * (N + 1) + j];

    // init r： r = B
    for(int i = 0; i <= M; i++)
        for(int j = 0; j <= N; j++)
            r[i * (N + 1) + j] = B[i * (N + 1) + j];

    // norm_B
    norm_B = norm(B, M, N, h1, h2);
 
    while(is_not_solved){
        iter ++;
        A_vector_multi(Ar, r, M, N, h1, h2);
        // alpha_k = norm(gk)^2 / (Ark, rk)
        alpha = _inner_product(g, g, M, N, h1, h2) / _inner_product(Ar, r, M, N, h1, h2);
        
        //update w: w_k+1 = w_k + alpha_k*rk
        for(int i = 0; i <= M; i++)
            for(int j = 0; j <= N; j++)
                w[i * (N + 1) + j] = w[i * (N + 1) + j] + alpha * r[i * (N + 1) + j];

        // store current g
        for(int i = 0; i <= M; i++)
            for(int j = 0; j <= N; j++)
                g_pr[i * (N + 1) + j] = g[i * (N + 1) + j];
        
        // update g: g_k+1 = g_k - alpha_kAr_k
        for(int i = 0; i <= M; i++)
            for(int j = 0; j <= N; j++)
                g[i * (N + 1) + j] = g[i * (N + 1) + j] - alpha * Ar[i * (N + 1) + j];
        
        // norm(g_k+1)^2 / norm(g_k)^2
        beta = _inner_product(g, g, M, N, h1, h2) / _inner_product(g_pr, g_pr, M, N, h1, h2);

        // update r: r_k+1 = g_k+1 + beta_k r_k
        for(int i = 0; i <= M; i++)
            for(int j = 0; j <= N; j++)
                r[i * (N + 1) + j] = g[i * (N + 1) + j] + beta * r[i * (N + 1) + j];
        
        if (iter == 100000) is_not_solved = false;
        
        error = norm(g, M, N, h1, h2) / norm(B, M, N, h1, h2);
        if (error < EPS) is_not_solved = false;

        if (iter % PRINT_FREQ == 0){
            cout << "iter: " << iter << ", alpha: "<< alpha << ", beta: " << beta 
            << ", error: " << error << "\n";
        }

    }

    cout << "Finished!!!" << "\n";
    cout << "total_iter: " << iter << ", final_err_norm: " << error << "\n";
    clock_t end = clock();
    cout << "The total running time is: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;

    cout << "Record the results..." << "\n";
    // record the results: two columns, the real values and the approximate values
    string file_name = "./serial_cg_results/res_" + to_string(M) + "_" + to_string(N) + ".txt";
    ofstream res_file(file_name, ios::out);
    res_file << M << "," << N << "\n";
    for(int i = 0; i <= M; i++)
    {
        for(int j = 0; j <= N; j++)
        {
            res_file << u((A1 + i * h1), (B1 + j * h2)) << ',' << w[i * (N + 1) + j] << "\n";
        }
    }
    res_file.close();

    cout << "Free memory...\n";  
    delete[] w;
    delete[] r;
    delete[] Ar;
    delete[] B;
    delete[] g;
    delete[] g_pr;
}

int main(int argc, char* argv[]){

    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    double h1 = (double)(A2 - A1) / M;
    double h2 = (double)(B2 - B1) / N;
    
    solve(M, N, h1, h2);
    
    return 0;
}
