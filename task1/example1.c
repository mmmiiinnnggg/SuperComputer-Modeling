//  the example c program
   for (i = 2; i <= n + 1; ++i)
   {
      C[i] = C[i - 2] * e;
   }
   for (i = 2; i <= n + 1; ++i)
   {
      for (j = 2; j <= m + 1; ++j)
      {
         B[i][j] = B[i - 2][j - 2];
      }
   }
   for (i = 2; i <= n + 1; ++i)
   {
      A[i][1][1] = B[i][m] + C[i];
      for (j = 2; j <= m + 1; ++j)
      {
         for (k = 1; k <= n - 1; ++k)
         {
            A[i][j][k] = A[i][j][k] + A[i - 1][j][k];
         }
      }
   }