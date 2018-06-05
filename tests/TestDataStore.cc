#include <stdio.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>

size_t rows = 100000;
size_t cols = 1000;

#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <ctime>
#include <iostream>
#include <vector>

int main (void)
{

  std::vector<double*> blocks;

  for(int i = 0; i < 8; i++){
    blocks.push_back(new double[10000 * 10000]);
  }

  for(int i = 0; i < 8; i++){
    for(auto j = 0; j < 10000 * 10000; ++j){
      blocks[i][j] = i + j;
    }
  }

  double x = 0;

  for(int i = 0; i < 8; i++){
    for(auto j = 0; j < 10000 * 10000; ++j){
      x += blocks[i][j];
    }
  }

  std::cout << x << std::endl;

  return 0;
}