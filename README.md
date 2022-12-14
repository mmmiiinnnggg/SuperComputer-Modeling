# SuperComputer-Modeling
HPC technologies practical works including MPI, OpenMP, OpenACC

## Task description  
Parallel programs to solve the **Poisson's differential equation** with potential $-\Delta u + q(x,y)u = F(x,y)$  

in the area of $\Pi = \{(x,y): 0 \leqslant x \leqslant 4, 0 \leqslant y \leqslant 3\}$, in which the Laplace operator is  

$\Delta u = \frac{\partial}{\partial x} \left(k(x,y)\frac{\partial u}{\partial x} \right) + \frac{\partial}{\partial y}\left(k(x,y)\frac{\partial u}{\partial y} \right)$

We need boundary conditions to determine the unique solution.  

For the left and right boundary conditions of the third type are given:  
$\left( k\frac{\partial u}{\partial n}\right)(x,y) + u(x,y) = \psi(x,y)$  

For the upper and lower bounds a second type condition is given:  
$\left( k\frac{\partial u}{\partial n}\right)(x,y) = \psi(x,y)$  

where $n$ is the unit external normal to the boundary.

The true function is $u(x,y)=\sqrt{4+xy}$, $k(x,y) = 4 + x + y$, $q(x,y) = x + y$. 

For the numerical solution we will use **minimal discrepancies method (MD)** and **conjugate gradients method (CG)**.  

## Results

- The results of MPI programs

| #MPI Processes | Grid | Time(s) - MD | Times(s) - CG | Acceleration for MD|
| :----: | :----: | :----: | :----: | :----: |
| 4 | 500*500 | 1198.100 | 16.868 | 1 |
| 8 | 500*500 | 645.545 | 9.058 | 1.856 |
| 16 | 500*500 | 356.502 | 5.027 | 3.361 |
| 32 | 500*500 | 198.376 | 3.928 | 6.040 | 
| 4 | 500*1000 | 877.178 | 58.074 | 1 |
| 8 | 500*1000 | 499.297 | 30.015 | 1.757 |
| 16 | 500*1000 | 287.097 | 16.691 | 3.055 |
| 32 | 500*1000 | 158.011 | 12.328 | 5.551 |

- The results of MPI+OpenMP programs

| #MPI Processes | #Threads | Grid | Time(s) - MD | Acceleration for MD|
| :----: | :----: | :----: | :----: | :----: |
| 1 | 4 | 500*500 | 1073.640| 1 |
| 2 | 4 | 500*500 | 531.197 | 2.021 |
| 4 | 4 | 500*500 | 268.924  | 3.992 |
| 8 | 4 | 500*500 | 168.822 | 6.350 | 
| 1 | 4 | 500*1000 | 791.297  | 1 |
| 2 | 4 | 500*1000 | 417.500 | 1.895 |
| 4 | 4 | 500*1000 | 198.427  | 3988 |
| 8 | 4 | 500*1000 | 104.533  | 7.570 |

- The results with big grid(15000×15000) and fixed number of iterations(100 iterations). The GPU device is Tesla P100. This size of the problem takes about 12GB of GPU RAM

| Configuration | Time(s) | Acceleration |
| :----: | :----: | :----: |
| Serial | 6125.610 | 1 |
| MPI-20 processes | 310.412 | 19.734 |
| MPI-40 processes | 161.982 | 37.816 |
| MPI-20 processes-2 threads | 263.950 | 23.207 |
| MPI-40 processes-2 threads | 133.614 | 45.846 |
| MPI-ACC-1 gpu| 10.154 | 603.271 |
| MPI-ACC-2 gpu | 6.292 | 973.555 |
| MPI-ACC-4 gpu | 3.324 | 1842.842 |

