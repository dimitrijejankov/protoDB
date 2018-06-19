#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <ResourceManager.h>
#include <thread>
#include <vector>
#include <iostream>
#include <boost/thread/barrier.hpp>
#include <fstream>
#include <atomic>

size_t size = 10;

boost::barrier *bar;

std::atomic<long> globalTime = 0;

void initMatrix(double *data) {
  for(int i = 0; i < size; ++i) {
    for(int j = 0; j < size; ++j) {
      data[i * size + j] = (double)(i + j);
    }
  }
}

void multiply() {

  // how many matrices we want to multiply
  int numMatrices = 2;

  std::vector<double*> matrices;

  // allocate the memory for the matrices
  for(int i = 0; i < (numMatrices + 2); i++) {
    matrices.push_back((double*) malloc(size * size * sizeof(double)));
    initMatrix(matrices.back());
  }

  bar->wait();

  long totalTime = 0;

  for(int i = 0; i < numMatrices; i++) {

    // create the views
    gsl_matrix_view c = gsl_matrix_view_array(matrices[i+2], size, size);
    gsl_matrix_view b = gsl_matrix_view_array(matrices[i+1], size, size);
    gsl_matrix_view a = gsl_matrix_view_array(matrices[i], size, size);

    // start time
    auto start = std::chrono::steady_clock::now();

    // do that multiply
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &a.matrix, &b.matrix, 0.0, &c.matrix);

    // end time
    auto end = std::chrono::steady_clock::now();

    // add the total time
    totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  // sum up the matrices so it does not get optimized out
  double tmp = 0;
  for(int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      tmp += matrices.back()[i + j * size];
    }
  }

  // just print this out so it does not get optmized out
  std::cout << "Value " << tmp << std::endl;
  globalTime += totalTime / numMatrices;

  // free the matrices
  for(int i = 0; i < (numMatrices + 2); i++) {
    free(matrices[i]);
  }
}

int main(int argc, char *argv[]) {

  // convert the argument to the size
  size = (size_t)atoi(argv[1]);
  size_t threadNo = (size_t)atoi(argv[2]);

  // init the barrier
  bar = new boost::barrier(threadNo);

  // create the resource manager
  ResourceManagerPtr resourceManager = (new ResourceManager())->getHandle()->to<ResourceManager>();

  // threads
  std::vector<std::thread*> threads;

  for(int i = 0; i < threadNo; ++i) {
    threads.push_back(new std::thread(multiply));
  }

  for(auto t : threads) {

    // wait to finish
    t->join();

    // delete
    delete t;
  }

  std::ofstream outfile("results_matrix_multiply.csv", std::ios_base::app);
  outfile << size << ", " << threadNo << ", " << globalTime / threadNo << std::endl;
  outfile.close();

  delete bar;
}