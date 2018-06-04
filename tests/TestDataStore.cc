#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

size_t rows = 100000;
size_t cols = 1000;

#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <ctime>
#include <iostream>

int main (void)
{
  auto *a = (double*) malloc(rows * cols * sizeof(double));
  auto *b = (double*) malloc(rows * cols * sizeof(double));
  auto *c = (double*) malloc(rows * rows * sizeof(double));

  int i, j = 0;

  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      a[i * cols + j] = 0.23 + i + j;
      b[i * cols + j] = 0.24 + i + j;

  gsl_matrix_view A = gsl_matrix_view_array(a, rows, cols);
  gsl_matrix_view B = gsl_matrix_view_array(b, cols, rows);
  gsl_matrix_view C = gsl_matrix_view_array(c, rows, rows);

  auto start = std::clock();

  /* Compute C = A B */
  gsl_blas_dgemm (CblasNoTrans, CblasNoTrans, 1.0, &A.matrix, &B.matrix, 0.0, &C.matrix);

  std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;

  printf ("[ %g, %g\n", c[0], c[1]);
  printf ("  %g, %g ]\n", c[2], c[3]);

  free(a);
  free(b);
  free(c);

  return 0;
}