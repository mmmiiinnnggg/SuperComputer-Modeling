#include <mpi.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <openacc.h>

#define TAG_X 666
using namespace std;


struct TestClass
{
    TestClass();
    void exchange();

    double send_array[5] = {.5, .5, .5, .5, .5};
    double recv_array[5] = {.6, .6, .6, .6, .6};
private:
    MPI_Status status;
    int rank, size;
    int size_y = 5;
    
};

TestClass::TestClass(){

    // get current process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // get processes number
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // multi-gpu affinity
    int n_gpus = acc_get_num_devices(acc_device_nvidia);
    int device_num = rank % n_gpus;
    acc_set_device_num(device_num, acc_device_nvidia);
}

void TestClass::exchange(){

    if (rank == 0){
        cout << "Starting.... ";
    }

    cout << "rank: " << rank << " checking ...... 0, size_y" <<  size_y << endl;

    #pragma acc enter data copyin(send_array[0:size_y], recv_array[0:size_y])

    if (rank < 2){ 
        
        #pragma acc kernels present(send_array[0:size_y])
        {
        #pragma acc loop independent
        for(int k = 0; k < 5; k ++ )
            send_array[k] = .4;
        }
    }

    cout << "rank: " << rank << " checking ...... 1" << endl;

    // https://stackoverflow.com/questions/65184286/pragma-acc-host-data-use-device-with-complex-variables
    if (rank < 2){
        // double* tmp_s_pointer = send_array;
        // tmp_s_pointer = send_array;
        #pragma acc host_data use_device(send_array)
        MPI_Send(send_array, size_y, MPI_DOUBLE,
         rank + 2, TAG_X, MPI_COMM_WORLD);
    }

    else{
        // double* tmp_r_pointer = recv_array;
        // tmp_r_pointer = recv_array;
        #pragma acc host_data use_device(recv_array)
        MPI_Recv(recv_array, size_y, MPI_DOUBLE,
         rank - 2, TAG_X, MPI_COMM_WORLD, &status);
    }

    cout << "rank: " << rank << " checking ...... 2" << endl;

    #pragma acc update host(send_array[0:5], recv_array[0:5])
    #pragma acc exit data delete(send_array[0:5], recv_array[0:5])

    cout << "rank: " << rank << " device: " << acc_get_device_num(acc_device_nvidia) <<" send_buf: " << send_array[0] << " recv_buf: " << recv_array[0] << endl;  

}

// to verify the work of #pragma acc host_data use_device
int main(int argc, char **argv) {
	
	printf("Number of device :%d \n",acc_get_num_devices(acc_device_not_host));

    MPI_Init(&argc, &argv);
    
    TestClass t;
    t.exchange();

    MPI_Finalize();

	return 0;
}
