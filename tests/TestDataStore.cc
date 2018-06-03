#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

size_t size = 10000;

#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <ctime>
#include <iostream>

int main (void)
{
  auto *a = (double*) malloc(size * size * sizeof(double));
  auto *b = (double*) malloc(size * size * sizeof(double));
  auto *c = (double*) malloc(size * size * sizeof(double));

  int i, j = 0;

  for (i = 0; i < size; i++)
    for (j = 0; j < size; j++)
      a[i * size + j] = 0.23 + i + j;
      c[i * size + j] = 0.24 + i + j;

  gsl_matrix_view A = gsl_matrix_view_array(a, size, size);
  gsl_matrix_view B = gsl_matrix_view_array(b, size, size);
  gsl_matrix_view C = gsl_matrix_view_array(c, size, size);

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