#include <cstddef>
#include <iostream>
#include <vector>
#include <Communicator.h>
#include <StandardOutputLogger.h>
#include <ResourceManager.h>
#include <thread>
#include <mpi.h>
#include <fcmm.hpp>
#include <bwtree.h>
#include <omp.h>
#include <boost/functional/hash/hash.hpp>

const int32_t ROW_IDX_TAG = 1;
const int32_t COL_IDX_TAG = 2;
const int32_t DOUBLE_TAG = 3;

typedef protoDB::bwtree::BwTree<size_t, size_t> matrixIndexBTree;
typedef protoDB::bwtree::BwTree<std::pair<size_t, size_t>, uint8_t,
                                std::less<std::pair<size_t, size_t>>,
                                std::equal_to<std::pair<size_t, size_t>>,
                                boost::hash<std::pair<size_t, size_t>>> aggregatedIndexBtree;

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

struct column_matrix_chunks {

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

struct BroadcastedIndices {

  BroadcastedIndices(size_t numberOfNodes, size_t numberOfIndices) {

    // allocate the colIDs
    colIDs.resize(numberOfIndices);

    // allocate the row ids
    rowIDs.resize(numberOfIndices);

    // allocate the counts for each node
    nodeCounts.resize(numberOfNodes);
  }

  /**
   * How many
   */
  std::vector<int32_t> nodeCounts;

  /**
   * The ids of the columns
   */
  std::vector<size_t> colIDs;

  /**
   * The ids of the rows
   */
  std::vector<size_t> rowIDs;
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
column_matrix *generateMatrix(Functor valueFunc, size_t size, size_t chunkSize) {

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

  column_matrix *matrix = generateMatrix(identityLambda, 4, 2);

  // initialize the permutation array
  std::vector<size_t> permutation;
  permutation.reserve(chunksPerDimension * chunksPerDimension);

  // this keeps track on how many were assigned
  std::vector<size_t> numberOnNode((size_t) communicator->getNumNodes());

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
  std::vector<column_matrix_chunks *> chunks;
  for (int i = 0; i < numNodes; ++i) {

    // create the chunk columns
    chunks.push_back(new column_matrix_chunks(size, chunkSize, numberOnNode[i]));
  }

  // go through each value in the permutation
  for (auto i = 0; i < permutation.size(); ++i) {

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
  for (int i = 0; i < numNodes; ++i) {

    // send rows
    communicator->send(chunks[i]->colIDs.data(), chunks[i]->colIDs.size(), i, ROW_IDX_TAG);

    // send cols
    communicator->send(chunks[i]->rowIDs.data(), chunks[i]->rowIDs.size(), i, COL_IDX_TAG);

    // send the values
    communicator->send(chunks[i]->values.data(), chunks[i]->values.size(), i, DOUBLE_TAG);
  }
}

void createMatrix(CommunicatorPtr &communicator,
                  size_t &size,
                  size_t chunkSize,
                  size_t chunksPerDimension,
                  std::vector<double> &values,
                  std::vector<size_t> &rowIDs,
                  std::vector<size_t> &colIDs) {
  if (communicator->isMaster()) {

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
  } else {

    // receive the shuffled data
    receiveRandomShuffled(communicator, rowIDs, colIDs, values, size, chunksPerDimension);
  }
}

void broadcastAllIndices(CommunicatorPtr &communicator,
                         std::vector<size_t> &rowIDs,
                         std::vector<size_t> &colIDs,
                         BroadcastedIndices &indices) {

  // grab the stats from each node
  auto localCount = (int32_t)rowIDs.size();
  communicator->allGather(localCount, indices.nodeCounts);

  // grab the indices from each node
  communicator->allGather(rowIDs, indices.rowIDs, indices.nodeCounts);
  communicator->allGather(colIDs, indices.colIDs, indices.nodeCounts);
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

  // this is where our stuff will be stored for matrix A
  std::vector<size_t> aRowIDs;
  std::vector<size_t> aColIDs;
  std::vector<double> aValues;

  // this is where our stuff will be stored for matrix B
  std::vector<size_t> bRowIDs;
  std::vector<size_t> bColIDs;
  std::vector<double> bValues;

  // create the matrices
  createMatrix(communicator, size, chunkSize, chunksPerDimension, aValues, aRowIDs, aColIDs);
  createMatrix(communicator, size, chunkSize, chunksPerDimension, bValues, bRowIDs, bColIDs);

  /// 1. Broadcast all the indices to every node

  // initialize the indices for A
  BroadcastedIndices aIndices((size_t) communicator->getNumNodes(), size);
  broadcastAllIndices(communicator, aRowIDs, aColIDs, aIndices);

  // initialize the indices for b
  BroadcastedIndices bIndices((size_t) communicator->getNumNodes(), size);
  broadcastAllIndices(communicator, bRowIDs, bColIDs, bIndices);

  /// 2. Join and Aggregate to get the indices of the final result

  // create a btree
  auto a_indexed = new matrixIndexBTree{true};

  // set the maximum number of threads
  a_indexed->UpdateThreadLocal((size_t) omp_get_max_threads());

  // do the parallel insert
  #pragma omp parallel
  {

    // register the thread
    a_indexed->AssignGCID(omp_get_thread_num());

    #pragma omp for
    for(int i = 0; i < aIndices.rowIDs.size(); ++i)
    {
      // store the thing into the btree
      a_indexed->Insert(aIndices.colIDs[i], aIndices.rowIDs[i]);
    }

    // unregister the thread
    a_indexed->UnregisterThread(omp_get_thread_num());
  }

  // aggregator map
  std::map<std::pair<size_t, size_t>, int> aggregator;

  // execute the probing and aggregating
  #pragma omp parallel
  {

    // register the thread
    a_indexed->AssignGCID(omp_get_thread_num());

    // local aggregation map
    std::map<std::pair<size_t, size_t>, int> localAggregator;

    // go through the indices in be join and then aggregate
    #pragma omp for
    for (int i = 0; i < bIndices.rowIDs.size(); ++i) {

      // grab the iterator
      auto it = a_indexed->Begin(bIndices.rowIDs[i]);

      // go through each match
      while(!it.IsEnd() && it->first == bIndices.rowIDs[i]) {

        // we are joining two tuples
        auto key = std::make_pair(it->second, bIndices.colIDs[i]);

        // if we don't have it set it to 0
        localAggregator.try_emplace(key, 0);

        // increment the thing
        localAggregator.find(key)->second =+ 1;

        // go to the next one
        it++;
      }
    }

    // unregister the thread
    a_indexed->UnregisterThread(omp_get_thread_num());

    // the merging has to be executed sequentially
    #pragma omp critical

    for(auto it : localAggregator) {

      // if we don't have it set it to 0
      aggregator.try_emplace(it.first, 0);

      // sum it up
      aggregator.find(it.first)->second += it.second;
    }

  }

}
