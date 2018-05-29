#include <mpi.h>
#include <cstddef>
#include <stdio.h>
#include <Communicator.h>
#include <DataStore.h>
#include <AbstractLogger.h>
#include <StandardOutputLogger.h>
#include <Server.h>
#include <ResourceManager.h>
#include <thread>
#include <iostream>
#include <omp.h>
#include <gsl/gsl_blas.h>
#include "../third-party/lockfree/fcmm.hpp"

// the total size of both matrices
const size_t matrixSize = 8;

// the chunk size
const size_t chunkSize = 2;

// the number of chunks per dimension our matrices have
const size_t chunksPerDimension = matrixSize / chunkSize;

// the partition size in terms of chunks
const size_t partitionSize = 32;

// chunk tag
const int notificationTag = 12;
const int chunkTag = 11;

struct matrix_chunk {

  /**
   * The row id of the chunk
   */
  size_t rowId;

  /**
   * The column id of the chunk
   */
  size_t colId;

  /**
   * the chunk
   */
  double data[chunkSize * chunkSize];

};

struct matrix_multiplied_chunk {

  /**
   * The row id of the chunk
   */
  size_t a_rowId;

  /**
   * The column id of the chunk
   */
  size_t a_colId;

  /**
 * The row id of the chunk
 */
  size_t b_rowId;

  /**
   * The column id of the chunk
   */
  size_t b_colId;

  /**
   * the chunk
   */
  double data[chunkSize * chunkSize];
};

template<typename Functor>
matrix_chunk *generateMatrix(Functor valueFunc) {

  // the matrix tmp in chunks this will be
  auto tmp = new matrix_chunk[chunksPerDimension * chunksPerDimension];

  // generate matrix tmp
  for (size_t i = 0; i < matrixSize; ++i) {

    // row id
    size_t rowID = i / chunkSize;

    for (size_t j = 0; j < matrixSize; ++j) {

      size_t colID = j / chunkSize;

      // set the row id and column id (we set this multiple times to avoid complicated logic)
      tmp[rowID * chunksPerDimension + colID].colId = colID;
      tmp[rowID * chunksPerDimension + colID].rowId = rowID;

      // get the indices within the chunk
      size_t block_i = i - rowID * chunkSize;
      size_t block_j = j - colID * chunkSize;

      // set the value
      tmp[rowID * chunksPerDimension + colID].data[block_i * chunkSize + block_j] = valueFunc(i, j);
    }
  }

  return tmp;
}

void receiveRandom(MPI_Datatype type,
                   std::vector<matrix_chunk> *out,
                   CommunicatorPtr communicator,
                   AbstractLoggerPtr logger) {

  int numValues = 1;
  matrix_chunk tmp{};

  // wait for a notify message
  MPI_Recv(&numValues, 1, MPI_INT, communicator->masterID(), notificationTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  // log action
  logger->info() << "Got massage to store " << numValues << "chunks" << logger->endl;

  for (int i = 0; i < numValues; ++i) {

    // wait for a block message
    MPI_Recv(&tmp, 1, type, communicator->masterID(), chunkTag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // add the chunk
    out->push_back(tmp);

    // log what we got
    logger->info() << "Got chunk with rowID : " << tmp.rowId << " colID : " << tmp.colId << logger->endl;
  }

  logger->info() << "Finished receiving chunks" << logger->endl;
}

void shuffleRandom(MPI_Datatype type,
                   matrix_chunk *matrix,
                   CommunicatorPtr communicator,
                   AbstractLoggerPtr logger,
                   int seed) {

  // get the number of nodes in this mpi cluster
  int32_t numNodes = communicator->getNumNodes();

  // seed with a constant so we can repeat the experiment
  srand(seed);

  // generate a random sequence
  int randomShuffleSequence[chunksPerDimension * chunksPerDimension];

  // number on each node
  auto numberOnNode = (int32_t *) calloc((size_t) numNodes, sizeof(int32_t));

  for (int &i : randomShuffleSequence) {

    // node to send to
    int node = rand() % numNodes;

    // ok this is the selected node
    i = node;

    // we added one more to this node
    numberOnNode[node] += 1;
  }

  // to each node we send how may things to expect
  for (auto i = 0; i < numNodes; ++i) {

    // notifying node
    logger->info() << "Notifying node " << i << " that it has " << numberOnNode[i] << " chunks" << logger->endl;

    // inform that we should continue
    MPI_Send(&numberOnNode[i], 1, MPI_INT, i, notificationTag, MPI_COMM_WORLD);
  }

  // send of the chunks randomly
  for (int i = 0; i < chunksPerDimension * chunksPerDimension; ++i) {

    // send the matrix
    MPI_Send(&matrix[i], 1, type, randomShuffleSequence[i], chunkTag, MPI_COMM_WORLD);

    logger->info() << "Sent chunk to matrix to node " << randomShuffleSequence[i] << logger->endl;
  }
}

template <typename T>
MPI_Datatype registerTypes() {

  MPI_Datatype chunkType;

  // create the type for the inner block
  MPI_Type_contiguous(sizeof(T), MPI_BYTE, &chunkType);
  MPI_Type_commit(&chunkType);

  return chunkType;
}

template<typename Functor>
void createAndShuffleMatrix(const CommunicatorPtr &communicator,
                            AbstractLoggerPtr &logger,
                            const MPI_Datatype type,
                            Functor matrixFunction,
                            std::vector<matrix_chunk> &out,
                            int seed) {

  // log the action
  logger->info() << "Creating matrix" << logger->endl;

  // the matrix a in chunks will be based on the matrix function
  auto tmp = generateMatrix(matrixFunction);

  // log the action
  logger->info() << "Matrix generated" << logger->endl;

  // start the shuffling
  std::thread shuffleThread(shuffleRandom, type, tmp, communicator, logger, seed);

  // start the receiving
  std::thread shuffleReceiveThread(receiveRandom, type, &out, communicator, logger);

  // wait until they finish
  shuffleThread.join();
  shuffleReceiveThread.join();

  // free matrix a
  delete[] tmp;
}

template<bool colOrRow>
void ShuffleMatrix(CommunicatorPtr communicator,
                   const ResourceManagerPtr resourceManager,
                   const AbstractLoggerPtr logger,
                   const std::vector<matrix_chunk> *Local_A,
                   const MPI_Datatype type) {

  // log what we are doing
  logger->info() << "Allocating partitions" << logger->endl;

  // allocate enough memory for each partition of A
  std::vector<std::vector<std::vector<matrix_chunk>>> A_Partitions;
  for (int i = 0; i < resourceManager->getNumCores(); ++i) {

    std::vector<std::vector<matrix_chunk>> partition;

    // each partition has a slice for each node
    for (int j = 0; j < communicator->getNumNodes(); j++) {

      // we preallocate the same amount for each node
      std::vector<matrix_chunk> partitionSlice;
      partitionSlice.reserve(partitionSize / communicator->getNumNodes());

      // store the slice
      partition.push_back(partitionSlice);
    }

    // store the partition
    A_Partitions.push_back(partition);
  }

  // log what we are doing
  logger->info() << "Partitioning the matrix data" << logger->endl;

  int32_t numNodes = communicator->getNumNodes();

  // go through stuff stored in A and partition it
  #pragma omp parallel for
  for (auto i = 0; i < Local_A->size(); i++) {

    // to which node are we sending this
    size_t node;
    if (colOrRow) {
      node = (*Local_A)[i].colId % numNodes;
    } else {
      node = (*Local_A)[i].rowId % numNodes;
    }

    // grab the thread id
    int tid = omp_get_thread_num();

    // store the node
    A_Partitions[tid][node].emplace_back((*Local_A)[i]);
  }

  // log what we are doing
  logger->info() << "Sending the data around" << logger->endl;

  // send each partition to the appropriate node
  for (auto i = 0; i < A_Partitions.size(); i++) {

    #pragma omp parallel for
    for (auto j = 0; j < numNodes; j++) {

      logger->info() << "Sending to node " << j << logger->endl;

      // send the chunks
      MPI_Send(A_Partitions[i][j].data(), (int) A_Partitions[i][j].size(), type, j, chunkTag, MPI_COMM_WORLD);
    }
  }
}

void receiveShuffledMatrix(CommunicatorPtr &communicator,
                           const ResourceManagerPtr &resourceManager,
                           const AbstractLoggerPtr &logger,
                           const MPI_Datatype type,
                           std::vector<matrix_chunk> &partitioned_A) {

  // log what is happening
  logger->info() << "Started receiving shuffled matrix" << logger->endl;

  // for each node we start a thread
  #pragma omp for schedule(static, 1)
  for (int i = 0; i < communicator->getNumNodes(); ++i) {

    for (int j = 0; j < resourceManager->getNumCores(); ++j) {

      auto size = (unsigned) ((partitionSize * sizeof(matrix_chunk)) / communicator->getNumNodes());
      auto *buffer = (matrix_chunk *) malloc(size);

      // the status of the MPI_Recv
      MPI_Status status{};

      // wait for a block message
      MPI_Recv(buffer, size, type, i, chunkTag, MPI_COMM_WORLD, &status);

      // grab the count
      int count;
      MPI_Get_count(&status, type, &count);

      // ok we have to copy this here
      #pragma omp critical

      // log what happened
      logger->info() << "Received  " << count << " matrix_chunks" << logger->endl;

      // copy the buffer
      partitioned_A.insert(partitioned_A.end(), buffer, buffer + count);

      free(buffer);
    }
  }
}

void multiplyAndShuffle(CommunicatorPtr communicator,
                        AbstractLoggerPtr logger,
                        ResourceManagerPtr resourceManager,
                        std::vector<matrix_chunk> *partitioned_A,
                        std::vector<matrix_chunk> *partitioned_B,
                        const fcmm::Fcmm<size_t, size_t> *a_indexed,
                        MPI_Datatype type) {

  // log what we are doing
  logger->info() << "Allocating partitions" << logger->endl;

  // allocate enough memory for each partition of A
  std::vector<std::vector<std::vector<matrix_multiplied_chunk>>> c_partitions;
  for (int i = 0; i < resourceManager->getNumCores(); ++i) {

    std::vector<std::vector<matrix_multiplied_chunk>> partition;

    // each partition has a slice for each node
    for (int j = 0; j < communicator->getNumNodes(); j++) {

      // we preallocate the same amount for each node
      std::vector<matrix_multiplied_chunk> partitionSlice;
      partitionSlice.reserve(partitionSize / communicator->getNumNodes());

      // store the slice
      partition.push_back(partitionSlice);
    }

    // store the partition
    c_partitions.push_back(partition);
  }

  // grab the number of nodes
  auto numNodes = communicator->getNumNodes();

  matrix_multiplied_chunk tmp{};
  gsl_matrix_view c = gsl_matrix_view_array(tmp.data, chunkSize, chunkSize);

  #pragma omp parallel for schedule(static, 1)
  for (size_t i = 0; i < partitioned_B->size(); ++i) {

    // go through each chunk
    for (size_t j = 0; j < chunksPerDimension; ++j) {

      // the key we are searching for
      size_t key = (*partitioned_B)[i].rowId * chunksPerDimension + j;

      // find the column that corresponds to this row id and this j in matrix A
      auto it = a_indexed->find(key);

      // did we find it
      if (it != a_indexed->end()) {

        // the index
        size_t idx = it->second;

        // grab the thread id
        int tid = omp_get_thread_num();

        // create views to a and b
        gsl_matrix_view a = gsl_matrix_view_array((*partitioned_A)[idx].data, chunkSize, chunkSize);
        gsl_matrix_view b = gsl_matrix_view_array((*partitioned_B)[i].data, chunkSize, chunkSize);

        // multiply them
        gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &a.matrix, &b.matrix, 0.0, &c.matrix);

        // set the columns and rows
        tmp.a_colId = (*partitioned_A)[idx].colId;
        tmp.a_rowId = (*partitioned_A)[idx].rowId;
        tmp.b_colId = (*partitioned_B)[i].colId;
        tmp.b_rowId = (*partitioned_B)[i].rowId;

        // hash and get node
        size_t node = (tmp.a_rowId * chunksPerDimension + tmp.b_colId) % numNodes;

        // store the node
        c_partitions[tid][node].emplace_back(tmp);
      }
    }
  }

  // log what we are doing
  logger->info() << "Sending the data around" << logger->endl;

  // send each partition to the appropriate node
  for (auto i = 0; i < c_partitions.size(); i++) {

    #pragma omp parallel for
    for (auto j = 0; j < numNodes; j++) {

      logger->info() << "Sending to node " << j << " size of partition" << c_partitions[i][j].size() << logger->endl;

      // send the chunks
      MPI_Send(c_partitions[i][j].data(), (int) c_partitions[i][j].size(), type, j, chunkTag, MPI_COMM_WORLD);
    }
  }
}

void receiveMultipliedShuffledMatrix(CommunicatorPtr &communicator,
                                     const ResourceManagerPtr &resourceManager,
                                     const AbstractLoggerPtr &logger,
                                     const MPI_Datatype type,
                                     std::vector<matrix_multiplied_chunk> &partitioned_c) {

  // log what is happening
  logger->info() << "Started receiving shuffled matrix" << logger->endl;

  // for each node we start a thread
  #pragma omp for schedule(static, 1)
  for (int i = 0; i < communicator->getNumNodes(); ++i) {

    for (int j = 0; j < resourceManager->getNumCores(); ++j) {

      auto size = (unsigned) ((partitionSize * sizeof(matrix_multiplied_chunk)) / communicator->getNumNodes());
      auto *buffer = (matrix_multiplied_chunk *) malloc(size);

      // the status of the MPI_Recv
      MPI_Status status{};

      // wait for a block message
      MPI_Recv(buffer, size, type, i, chunkTag, MPI_COMM_WORLD, &status);

      // grab the count
      int count;
      MPI_Get_count(&status, type, &count);

      // ok we have to copy this here
      #pragma omp critical

      // log what happened
      logger->info() << "Received  " << count << " matrix_chunks" << logger->endl;

      // copy the buffer
      partitioned_c.insert(partitioned_c.end(), buffer, buffer + count);

      free(buffer);
    }
  }
}

/**
 * This test does a multiply of square matrices a and b (a * b) using the traditional relational method
 * @return - 0 if we succeed if we fail it is undefined
 */
int main() {

  // create the communicator
  CommunicatorPtr communicator = (new Communicator())->getHandle()->to<Communicator>();

  // this makes the memory management easier
  assert(chunksPerDimension % communicator->getNumNodes() == 0);

  // create a logger
  AbstractLoggerPtr logger;

  if (communicator->isMaster()) {
    logger = (new StandardOutputLogger("Master"))->getHandle()->to<AbstractLogger>();
  } else {
    logger = (new StandardOutputLogger("Worker"))->getHandle()->to<AbstractLogger>();
  }

  // create the resource manager
  ResourceManagerPtr resourceManager = (new ResourceManager())->getHandle()->to<ResourceManager>();

  // the two local matrix chunks A and B
  std::vector<matrix_chunk> Local_A;
  std::vector<matrix_chunk> Local_B;

  // register the types we need
  MPI_Datatype matrixType = registerTypes<matrix_chunk>();
  MPI_Datatype joinMatrixType = registerTypes<matrix_multiplied_chunk>();

  /// Step 1 generated the matrices and shuffle them randomly

  // if we are the master we generate the matrices a we shuffle them randomly
  if (communicator->isMaster()) {

    // create the identity lambda
    auto identityLambda = [](size_t i, size_t j) { return i == j; };

    // log what we are doing
    logger->info() << "Doing matrix A" << logger->endl;

    // create and shuffle the A matrix
    createAndShuffleMatrix(communicator, logger, matrixType, identityLambda, Local_A, 100);

    // log what we are doing
    logger->info() << "Doing matrix B" << logger->endl;

    // the matrix b will will fit a sequence of 0,1,2,3,4,5 row-wise
    auto sequenceLambda = [](size_t i, size_t j) { return i * matrixSize + j; };

    // create and shuffle the B matrix
    createAndShuffleMatrix(communicator, logger, matrixType, sequenceLambda, Local_B, 101);

  } else {

    // create a thread to receive the shuffled parts of A
    std::thread shuffleReceiveAThread(receiveRandom, matrixType, &Local_A, communicator, logger);

    // wait until finished receiving A
    shuffleReceiveAThread.join();

    // create a thread to receive the shuffled parts of B
    std::thread shuffleReceiveBThread(receiveRandom, matrixType, &Local_B, communicator, logger);

    // wait until finished receiving B
    shuffleReceiveBThread.join();
  }

  // log the number of chunks
  logger->info() << "Number of chunks in A " << Local_A.size() << logger->endl;
  logger->info() << "Number of chunks in B " << Local_B.size() << logger->endl;

  /// Step 2 take the generated matrices and shuffle them by their join attribute

  // allocate space for partitioned A
  std::vector<matrix_chunk> partitioned_A;
  partitioned_A.reserve((chunksPerDimension / communicator->getNumNodes()) * chunksPerDimension);

  {
    // start shuffling
    std::thread shuffleThread(ShuffleMatrix<true>, communicator, resourceManager, logger, &Local_A, matrixType);

    // start receiving the matrix
    receiveShuffledMatrix(communicator, resourceManager, logger, matrixType, partitioned_A);

    // wait to finish shuffling
    shuffleThread.join();
  }

  // allocate space for partitioned B
  std::vector<matrix_chunk> partitioned_B;
  partitioned_B.reserve((chunksPerDimension / communicator->getNumNodes()) * chunksPerDimension);

  {
    // start shuffling
    std::thread shuffleThread(ShuffleMatrix<false>, communicator, resourceManager, logger, &Local_B, matrixType);

    // start receiving the matrix
    receiveShuffledMatrix(communicator, resourceManager, logger, matrixType, partitioned_B);

    // wait to finish shuffling
    shuffleThread.join();
  }

  /// Step 4 create the b-tree as a secondary index

  fcmm::Fcmm<size_t, size_t> a_indexed;

  #pragma omp prallel for
  for (size_t i = 0; i < partitioned_A.size(); ++i) {

    // insert this thing into the b-tree
    a_indexed.insert(std::make_pair(partitioned_A[i].colId * chunksPerDimension + partitioned_A[i].rowId, i));
  }

  /// Step 5 do the multiply and shuffle TODO (have a concurrency issue here)

  std::vector<matrix_multiplied_chunk> partitioned_c;

  {
    // run the shuffle thread
    std::thread shuffleThread(multiplyAndShuffle, communicator, logger, resourceManager, &partitioned_A, &partitioned_B, &a_indexed, joinMatrixType);

    // receive the multiplied shuffled matrix
    receiveMultipliedShuffledMatrix(communicator, resourceManager, logger, joinMatrixType, partitioned_c);

    // wait to finish the shuffling
    shuffleThread.join();
  }

  logger->info() << "number of " << partitioned_c.size() << logger->endl;

  /// Step 6 aggregate

  // grab the number of nodes
  auto numNodes = communicator->getNumNodes();

  // preallocate the aggregation chunks
  std::vector<matrix_chunk> aggregationChunks(chunksPerDimension * chunksPerDimension / numNodes);

  // this indexes the aggregated matrices
  std::unordered_map<size_t, matrix_chunk*> c_aggregated;

  size_t idx = 0;

  // go through chunks
  for(size_t i = 0; i < chunksPerDimension; i++){
    for(size_t j = 0; j < chunksPerDimension; j++) {

      // is this a chunk on this node
      if(((i * chunksPerDimension + j) % numNodes) == communicator->getNodeID()){

        // store a pointer to the chunk
        auto key = i * chunksPerDimension + j;
        c_aggregated[key] = aggregationChunks.data() + idx;

        aggregationChunks[idx].rowId = i;
        aggregationChunks[idx].colId = j;

        // init the memory to zero
        for (double &k : aggregationChunks[idx].data) {
          k = 0;
        }

        // go to the next spot
        idx++;
      }
    }
  }

  for(auto &it : partitioned_c) {

    std::cout << it.a_rowId << " : " << it.a_colId << " : " << it.b_rowId << " : "<< it.b_colId << std::endl;

    // get the key for this
    auto key = it.a_rowId * chunksPerDimension + it.b_colId;

    // grab the matrix
    auto matrix = c_aggregated[key];

    // sum up the matrix
    for(int i = 0; i < chunkSize * chunkSize; ++i) {
      matrix->data[i] += it.data[i];
    }
  }

  return 0;
}

