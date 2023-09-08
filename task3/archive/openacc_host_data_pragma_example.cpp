#include<stdio.h>
#include<openacc.h>
#include<mpi.h>

#define TAG_X 666
using namespace std;

int main(int argc, char* argv[])
{
    cout << "Number of device :%d \n" << acc_get_num_devices(acc_device_not_host);

    MPI_Init(&argc, &argv);

    MPI_Status status;

    int size, rank;

    // get current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // get processes number
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // multi-gpu affinity
    int n_gpus = acc_get_num_devices(acc_device_nvidia);
    int device_num = rank % n_gpus;
    acc_set_device_num(device_num, acc_device_nvidia);


    int size_y = 5;
    double send_array[5] = {.5, .5, .5, .5, .5};
    double recv_array[5] = {.6, .6, .6, .6, .6};

    if (rank == 0) cout << "Starting...." << endl;

    #pragma acc enter data copyin(send_array[0:5], recv_array[0:5])

    if (rank < 2){
        #pragma acc host_data use_device(send_array)
        MPI_Send(send_array, size_y, MPI_DOUBLE, rank + 2, TAG_X, MPI_COMM_WORLD);
    }

    else{
        #pragma acc host_data use_device(recv_array)
        MPI_Recv(recv_array, size_y, MPI_DOUBLE, rank - 2, TAG_X, MPI_COMM_WORLD, &status);
    }

    #pragma acc update host(send_array[0:5], recv_array[0:5])
    #pragma acc exit data delete(send_array[0:5], recv_array[0:5])


    cout << "rank: " << rank << " send_buf: " << send_array << " recv_buf" << recv_array << endl;  

    // double res = 0;
    // int i, j;
    // #pragma acc kernels
    // {
    //     #pragma acc loop independent
    //     for (i = 0; i <= 2; i++)
    //     {
    //         #pragma acc loop independent
    //         for (j = 0; j <= 2; j ++)
    //             res += 1;
    //     }

    //     res += 2;

    // }

    // printf("res : %f", res);

    MPI_Finalize();

    return 0;
    
}