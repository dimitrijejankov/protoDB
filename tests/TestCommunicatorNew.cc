#include <cstddef>
#include <iostream>
#include <vector>
#include <Communicator.h>
#include <StandardOutputLogger.h>
#include <ResourceManager.h>
#include <thread>

const int32_t ROW_IDX_TAG = 1;
const int32_t COL_IDX_TAG = 2;
const int32_t DOUBLE_TAG = 3;


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

class column_matrix_chunks {

 public:

  column_matrix_chunks(size_t size, size_t chunkSize, size_t n) : size(size), chunkSize(chunkSize) {

    // reserve the column ids
    colIDs.reserve(n);

    // reserve the row ids
    rowIDs.reserve(n);

    // reserve values
    values.reserve(chunkSize * chunkSize * n);
  }

  /**
   * The size of the matrix (rows num and col num)
   */
  size_t size;

  /**
   * The size of each chunk
   */
  size_t chunkSize;

  /**
   * The ids of the columns
   */
  std::vector<size_t> colIDs;

  /**
   * The ids of the rows
   */
  std::vector<size_t> rowIDs;

  /**
   * The values
   */
  std::vector<double> values;
};

void receiveRandomShuffled(CommunicatorPtr &communicator,
                           std::vector<size_t> &rowIDs,
                           std::vector<size_t> &colIDs,
                           std::vector<double> &values,
                           size_t size,
                           size_t chunksPerDimension) {

  // reserve the right amount of stuff
  rowIDs.reserve(chunksPerDimension * chunksPerDimension);
  colIDs.reserve(chunksPerDimension * chunksPerDimension);
  values.reserve(size * size);

  // grab the rows
  communicator->recv(rowIDs, 0, ROW_IDX_TAG);

  // grab the columns
  communicator->recv(colIDs, 0, COL_IDX_TAG);

  // grab the double
  communicator->recv(values, 0, DOUBLE_TAG);
}

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

void createAndRandomShuffle(CommunicatorPtr communicator,
                            size_t size,
                            size_t chunkSize,
                            size_t chunksPerDimension,
                            bool (*identityLambda)(size_t, size_t)) {

  column_matrix* matrix = generateMatrix(identityLambda, 4, 2);

  // initialize the permutation array
  std::vector<size_t > permutation;
  permutation.reserve(chunksPerDimension * chunksPerDimension);

  // this keeps track on how many were assigned
  std::vector<size_t > numberOnNode((size_t)communicator->getNumNodes());

  // grab the number of nodes
  auto numNodes = communicator->getNumNodes();

  // seed with a constant so we can repeat the experiment
  uint32_t seed = 205;
  srand(seed);

  // generate a random permutation
  for (int i = 0; i < chunksPerDimension * chunksPerDimension; ++i) {

    // node to send to
    int node = rand() % numNodes;

    // ok this is the selected node
    permutation.push_back((size_t) node);

    // we added one more to this node
    numberOnNode[node] += 1;
  }

  // initialize the chunks
  std::vector<column_matrix_chunks*> chunks;
  for(int i = 0; i < numNodes; ++i){

    // create the chunk columns
    chunks.push_back(new column_matrix_chunks(size, chunkSize, numberOnNode[i]));
  }

  // go through each value in the permutation
  for(auto i = 0; i < permutation.size(); ++i) {

    // store the row id
    chunks[permutation[i]]->rowIDs.emplace_back(matrix->rowIDs[i]);

    // store the column id
    chunks[permutation[i]]->colIDs.emplace_back(matrix->colIDs[i]);

    // store the matrix
    chunks[permutation[i]]->values.insert(chunks[permutation[i]]->values.end(),
                                          &matrix->values[i * chunkSize * chunkSize],
                                          &matrix->values[i * chunkSize * chunkSize] + chunkSize * chunkSize);
  }

  // send the stuff to the appropriate node
  for(int i = 0; i < numNodes; ++i) {

    // send rows
    communicator->send(chunks[i]->colIDs.data(), chunks[i]->colIDs.size(), i, ROW_IDX_TAG);

    // send cols
    communicator->send(chunks[i]->rowIDs.data(), chunks[i]->rowIDs.size(), i, COL_IDX_TAG);

    // send the values
    communicator->send(chunks[i]->values.data(), chunks[i]->values.size(), i, DOUBLE_TAG);
  }
}

/**
 * This test does a multiply of square matrices a and b (a * b) using the new relational method
 * @return - 0 if we succeed if we fail it is undefined
 */
int main() {

  // create the communicator
  CommunicatorPtr communicator = (new Communicator())->getHandle()->to<Communicator>();

  // create a logger
  AbstractLoggerPtr logger;

  if (communicator->isMaster()) {
    logger = (new StandardOutputLogger("Master"))->getHandle()->to<AbstractLogger>();
  } else {
    logger = (new StandardOutputLogger("Worker"))->getHandle()->to<AbstractLogger>();
  }

  // create the resource manager
  ResourceManagerPtr resourceManager = (new ResourceManager())->getHandle()->to<ResourceManager>();

  // dimensions of the matrices A and B
  size_t size = 4;
  size_t chunkSize = 2;
  size_t chunksPerDimension = size / chunkSize;

  // this is where our stuff will be stored
  std::vector<size_t> rowIDs;
  std::vector<size_t> colIDs;
  std::vector<double> values;

  if(communicator->isMaster()) {

    // create the identity lambda
    auto identityLambda = [](size_t i, size_t j) { return i == j; };

    // create the matrix and random shuffle it
    std::thread shuffleReceiveAndThread(createAndRandomShuffle,
                                        communicator,
                                        size,
                                        chunkSize,
                                        chunksPerDimension,
                                        identityLambda);

    // receive the shuffled data
    receiveRandomShuffled(communicator, rowIDs, colIDs, values, size, chunksPerDimension);

    // wait for the shuffle to finish
    shuffleReceiveAndThread.join();
  }
  else {

    // receive the shuffled data
    receiveRandomShuffled(communicator, rowIDs, colIDs, values, size, chunksPerDimension);
  }

  std::cout << rowIDs.size() << " " << colIDs.size() << " " << values.size() << std::endl;

  return 0;
}
