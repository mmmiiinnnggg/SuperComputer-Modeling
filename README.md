# SuperComputer-Modeling
HPC technologies practical works including MPI, OpenMP, OpenACC

Parallel programs to solve the Poisson's differential equation with potential
\begin{equation*}
    -\Delta u + q(x,y)u = F(x,y)
\end{equation*}
in the area of $\Pi = \{(x,y): 0 \leqslant x \leqslant 4, 0 \leqslant y \leqslant 3\}$, in which the Laplace operator is
\begin{equation*}
    \Delta u = \frac{\partial}{\partial x} \left(k(x,y)\frac{\partial u}{\partial x} \right) + \frac{\partial}{\partial y}\left(k(x,y)\frac{\partial u}{\partial y} \right)
\end{equation*}
The true function is $u(x,y)=\sqrt{4+xy}$, $k(x,y) = 4 + x + y, q(x,y) = x + y$. 

| Configuration | Time(s) | Acceleration |
| :-----| ----: | :----: |
| Serial | 6125.610 | 1 |
| MPI-20 processes | 310.412 | 19.734 |
| MPI-40 processes | 161.982 | 37.816 |
| MPI-20 processes-2 threads | 263.950 | 23.207 |
| MPI-40 processes-2 threads | 133.614 | 45.846 |
| MPI-ACC-1 gpu| 10.154 | 465.684 |
| MPI-ACC-2 gpu | 6.292 | 973.555 |
| MPI-ACC-4 gpu | 3.324 | 1842.842 |

