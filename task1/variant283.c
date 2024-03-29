for (i = 2; i <= n + 1; ++i)
   C[i] = C[i + 1] + D[i];
for (i = 2; i <= n + 1; ++i)
#pragma omp parallel for
   for (j = 2; j <= m + 1; ++j)
      B[i][j] = B[i - 2][j + 1];
for (i = 2; i <= n + 1; ++i)
{
   A[i][1][1] = B[i][m + 1] + C[i];
#pragma omp parallel for
   for (j = 2; j <= m + 1; ++j)
   {
#pragma omp parallel for
      for (k = 1; k <= n; ++k)
         A[i][j][k] = A[i - 1][j][k] + A[i][j][k];
   }
}
