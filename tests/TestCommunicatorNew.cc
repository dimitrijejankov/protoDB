#include <cstddef>
#include <iostream>

struct column_matrix {

  /**
   * The size of the matrix (rows num and col num)
   */
  size_t size;

  /**
   * The size of each chunk
   */
  size_t chunkSize;

  /**
   * The values in this
   */
  double *values;

  /**
   * The column IDs
   */
  size_t *colIDs;

  /**
   * The rowIDs of the chunks
   */
  size_t *rowIDs;
};


template<typename Functor>
column_matrix* generateMatrix(Functor valueFunc, size_t size, size_t chunkSize) {

  // figure out the chunk size
  size_t chunksPerDimension = size / chunkSize;

  // the matrix tmp in chunks this will be
  auto tmp = new column_matrix();

  // initialize the sizes
  tmp->chunkSize = chunkSize;
  tmp->size = size;

  // allocate the values and the columns
  tmp->values = new double[size * size];
  tmp->rowIDs = new size_t[chunksPerDimension * chunksPerDimension];
  tmp->colIDs = new size_t[chunksPerDimension * chunksPerDimension];

  // generate matrix tmp
  for (size_t i = 0; i < size; ++i) {

    // row id
    size_t rowID = i / chunkSize;

    for (size_t j = 0; j < size; ++j) {

      size_t colID = j / chunkSize;

      // set the row id and column id (we set this multiple times to avoid complicated logic)
      tmp->rowIDs[rowID * chunksPerDimension + colID] = colID;
      tmp->colIDs[rowID * chunksPerDimension + colID] = rowID;

      // get the indices within the chunk
      size_t block_i = i - rowID * chunkSize;
      size_t block_j = j - colID * chunkSize;

      size_t block_offset = (rowID * chunksPerDimension + colID) * chunkSize * chunkSize;
      size_t data_offset = block_i * chunkSize + block_j;

      // set the value
      tmp->values[block_offset + data_offset] = valueFunc(i, j);
    }
  }

  return tmp;
}




/**
 * This test does a multiply of square matrices a and b (a * b) using the new relational method
 * @return - 0 if we succeed if we fail it is undefined
 */
int main() {

  // create the identity lambda
  auto identityLambda = [](size_t i, size_t j) { return i == j; };
  column_matrix* matrix = generateMatrix(identityLambda, 4, 2);

  auto chunksPerDimension = matrix->size / matrix->chunkSize;
  auto chunkSize = matrix->chunkSize;

  for(int i = 0; i < chunksPerDimension; ++i) {
    for(int j = 0; j < chunksPerDimension; ++j) {

      size_t block_offset = (i * chunksPerDimension + j) * matrix->chunkSize * matrix->chunkSize;

      std::cout << "Chunk (" << i << ", " << j << ")" << std::endl;

      for(int k = 0; k < matrix->chunkSize; ++k) {
        for(int l = 0; l < matrix->chunkSize; l++) {
          std::cout << matrix->values[block_offset + k * chunkSize + l];
        }

        std::cout << std::endl;
      }
    }
  }

  return 0;
}