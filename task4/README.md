some clarification about `mpi_acc.cpp`

- The submitted program didn't optimized for the 1-GPU scenario: the unnecessary MPI exchange data procedure (because only 1 process, no MPI communication), unnecessary CPU-GPU data exchange. So the 2x+ acceleration on 2-GPU is observed.
- So, the final running time of 1-GPU program is not the optimal one.
- In theory the program should be rewritten based on serial version, not mpi version. It shouldn't be too hard :)