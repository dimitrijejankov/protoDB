#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <Communicator.h>
#include <AbstractLogger.h>
#include <StandardOutputLogger.h>
#include <ResourceManager.h>
#include <thread>
#include <omp.h>
#include <mutex>
#include <blockingconcurrentqueue.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <unistd.h>
#include "bwtree.h"

//#define DEBUG_ME

// tags
const int32_t COUNTS_TAG = 1;
const int32_t CHUNK_TAG = 2;
const int32_t AGG_CHUNK_TAG = 3;

// dimensions of the matrices A and B
size_t size = 256000;
size_t chunkSize = 3200;

typedef std::atomic<int32_t> atomic_int32_t;

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
   * How many are there on each node
   */
  std::vector<int32_t> nodeCounts;

  /**
   * On which node is the particular tuple
   */
  std::vector<int32_t> nodes;

  /**
   * The ids of the columns
   */
  std::vector<size_t> colIDs;

  /**
   * The ids of the rows
   */
  std::vector<size_t> rowIDs;
};

struct MatrixChunk {

  /**
   * the col id of the block
   */
  size_t colID;

  /**
   * the row id of the block
   */
  size_t rowID;

  /**
   * Returns the data
   * @return
   */
  inline double* getData() {

    // get the data
    return (double*)((std::int8_t*)this + sizeof(MatrixChunk));
  }

  inline MatrixChunk* getChunkAt(size_t n) {

    // grab the chunk
    size_t chunkSizeSquared = chunkSize * chunkSize * sizeof(double) + sizeof(MatrixChunk);
    return (MatrixChunk*)((std::int8_t*)this + chunkSizeSquared * n);
  }

  /**
   * Allocate the memory for a number of matrices
   * @param n - the number of elements to allocate
   * @param size - the size of the matrix
   * @return - the allocated matrix
   */
  static MatrixChunk* allocateMemory(std::size_t n) {
    return (MatrixChunk*) calloc(n, sizeof(MatrixChunk) + (chunkSize * chunkSize * sizeof(double)));
  }

  /**
   * Returns the structure
   * @return the mpi structure
   */
  static MPI_Datatype getStructure() {

    // create a new continuous chunk of memory
    MPI_Datatype newType;
    MPI_Type_contiguous(sizeof(size_t) * 2 + chunkSize * chunkSize * sizeof(double), MPI_BYTE, &newType);

    return newType;
  }
};

// define the containers
typedef protoDB::bwtree::BwTree<size_t, int> matrixIndexBTree;
typedef protoDB::bwtree::BwTree<int, size_t> matrixReverseIndexBTree;
typedef concurent::BlockingConcurrentQueue<MatrixChunk*> chunkQueue;

void generateMatrix(size_t (*valueFunc)(size_t, size_t), size_t size, size_t chunkSize, CommunicatorPtr communicator) {

  // figure out the chunk size
  size_t chunksPerDimension = size / chunkSize;

  // grab the number of nodes
  auto numNodes = (size_t) communicator->getNumNodes();

  /// 1. Generate a random permutation for each node and store the counts

  // allocate the memory for the permutation
  std::vector<int32_t> permutation(chunksPerDimension * chunksPerDimension);

  // allocate the memory for the counts
  std::vector<size_t> counts(numNodes);

  // go through each chunk
  for(auto c_i = 0; c_i < chunksPerDimension; ++c_i) {

    // go through each chunk
    for(auto c_j = 0; c_j < chunksPerDimension; ++c_j) {

      // the node we are sending this chunk to
      int32_t node = rand() % (int32_t) numNodes;

      // set the permutation
      permutation[c_i * chunksPerDimension + c_j] = node;

      // increase the counts
      counts[node]++;
    }
  }

  /// 2. Send our counts

  // go through each node
  for(auto node = 0; node < numNodes; ++node) {

    // send node
    communicator->send(counts[node], node, COUNTS_TAG);
  }

  /// 3. Generate chunks and send them

  // allocate a temporary chunk
  auto tmp = MatrixChunk::allocateMemory(1);
  auto tmpData = tmp->getData();

  // go through each chunk
  for(size_t c_i = 0; c_i < chunksPerDimension; ++c_i) {

    // go through each chunk
    for(size_t c_j = 0; c_j < chunksPerDimension; ++c_j) {

      // initialize the chunk
      for(auto i = 0; i < chunkSize; ++i) {
        for(auto j = 0; j < chunkSize; ++j) {

          // calculate the index of the value within the chunk
          auto valueIndex = i * chunkSize + j;

          // set the value
          tmpData[valueIndex] = valueFunc(c_i * chunkSize + i, c_j * chunkSize + j);
        }
      }

      // set the rows and columns
      tmp->rowID = c_i;
      tmp->colID = c_j;

      //clock_t start = clock();

      // send the chunk
      communicator->send(tmp, 1, permutation[c_i * chunksPerDimension + c_j], CHUNK_TAG);

      //clock_t end = clock();
      //double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
      //std::cout << elapsed_secs << std::endl;
    }
  }

  // free the temporary chunk
  free(tmp);
}

void receiveRandomShuffled(CommunicatorPtr &communicator,
                           std::vector<size_t> &rowIDs,
                           std::vector<size_t> &colIDs,
                           MatrixChunk **matrixChunks) {

  // grab the number of tuples
  auto num = communicator->recv<size_t>(0, COUNTS_TAG);

  // reserve the approximate amount of memory
  rowIDs.reserve(num);
  colIDs.reserve(num);

  // allocate the chunks
  (*matrixChunks) = MatrixChunk::allocateMemory(num);

  // receive each value
  for(size_t i = 0; i < num; ++i) {

    MatrixChunk* currentChunk = (*matrixChunks)->getChunkAt(i);

    // grab the double chunk
    communicator->recv(currentChunk, 0, CHUNK_TAG);

    // copy the row ids and column ids
    rowIDs.push_back(currentChunk->rowID);
    colIDs.push_back(currentChunk->colID);
  }
}

void createMatrix(size_t (*valueFunc)(size_t, size_t),
                  CommunicatorPtr &communicator,
                  size_t &size,
                  size_t chunkSize,
                  MatrixChunk **values,
                  std::vector<size_t> &rowIDs,
                  std::vector<size_t> &colIDs) {

  if (communicator->isMaster()) {

    // create the matrix and random shuffle it
    std::thread shuffleReceiveAndThread(generateMatrix, valueFunc, size, chunkSize, communicator);

    // receive the shuffled data
    receiveRandomShuffled(communicator, rowIDs, colIDs, values);

    // wait for the shuffle to finish
    shuffleReceiveAndThread.join();
  } else {

    // receive the shuffled data
    receiveRandomShuffled(communicator, rowIDs, colIDs, values);
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

  // initialize the node assignments
  for(auto i = 0; i < indices.nodeCounts.size(); ++i) {
    indices.nodes.insert(indices.nodes.end(), indices.nodeCounts[i], i);
  }
}

matrixIndexBTree* indexByColumnID(BroadcastedIndices &indices) {

  // create a btree
  auto indexed = new matrixIndexBTree{true};

  // set the maximum number of threads
  indexed->UpdateThreadLocal((size_t) omp_get_max_threads());

  // do the parallel insert
  #pragma omp parallel
  {

    // register the thread
    indexed->AssignGCID(omp_get_thread_num());

    #pragma omp for
    for(int i = 0; i < indices.rowIDs.size(); ++i)
    {
      // store the thing into the btree
      indexed->Insert(indices.colIDs[i], i);
    }

    // unregister the thread
    indexed->UnregisterThread(omp_get_thread_num());
  }

  return indexed;
}

size_t getNodeOffset(BroadcastedIndices &indices, int32_t myNodeID){

  size_t nodeOffset = 0;

  // sum up the offsets before me
  for(auto i = 0; i < myNodeID; ++i) {
    nodeOffset += indices.nodeCounts[i];
  }

  // return the offset
  return nodeOffset;
}



void preprocessJoinAndAggregation(int32_t myNodeID,
                                  int32_t numNodes,
                                  BroadcastedIndices aIndices,
                                  BroadcastedIndices bIndices,
                                  matrixIndexBTree* aIndexed,
                                  size_t bNodeOffset,
                                  std::map<std::pair<size_t, size_t>, int> &aggregator,
                                  matrixReverseIndexBTree *bReverseIndexed,
                                  std::vector<atomic_int32_t> &sentMultiCounts) {

  // set the number of threads
  bReverseIndexed->UpdateThreadLocal((size_t) omp_get_max_threads());

  // execute the probing and aggregating
  #pragma omp parallel
  {

    // register the thread
    aIndexed->AssignGCID(omp_get_thread_num());
    bReverseIndexed->AssignGCID(omp_get_thread_num());

    // local aggregation map
    std::map<std::pair<size_t, size_t>, int> localAggregator;

    // go through the indices in be join and then aggregate
    #pragma omp for
    for (size_t i = 0; i < bIndices.rowIDs.size(); ++i) {

      // grab the iterator
      auto it = aIndexed->Begin(bIndices.rowIDs[i]);

      // go through each match
      while(!it.IsEnd() && it->first == bIndices.rowIDs[i]) {

        //logger->info() << "Joining " << "A : (" << aIndices.rowIDs[it->second] << ", " << aIndices.colIDs[it->second] << ") and (" << bIndices.rowIDs[i] << ", " << bIndices.colIDs[i] << ")" << logger->endl;

        // grab the row id
        auto rowID = aIndices.rowIDs[it->second];

        // we are joining two tuples
        auto key = std::make_pair(rowID, bIndices.colIDs[i]);

        // is this on my node
        if(bIndices.nodes[i] == myNodeID) {

          // insert the reverse index
          bReverseIndexed->Insert(aIndices.nodes[it->second], i - bNodeOffset);
        }

        // if we don't have it set it to 0
        if(localAggregator.find(key) == localAggregator.end()) {
          localAggregator.emplace(key, 0);
        }

        // increment the thing
        localAggregator.find(key)->second =+ 1;

        // figure out where the join is done
        auto joinNode = aIndices.nodes[it->second];

        // figure out where the aggregation is done
        auto aggregationNode = (key.first + key.second) % numNodes;

        // increment the damn counter
        if(aggregationNode == myNodeID && joinNode != aggregationNode) {

          sentMultiCounts[joinNode]++;
        }

        // go to the next one
        it++;
      }
    }

    // unregister the thread
    aIndexed->UnregisterThread(omp_get_thread_num());
    bReverseIndexed->UnregisterThread(omp_get_thread_num());

    // the merging has to be executed sequentially
    #pragma omp critical
    {
      for(auto it : localAggregator) {

        // if we don't have it set it to 0
        if(aggregator.find(it.first) == aggregator.end()) {
          aggregator.emplace(it.first, 0);
        }

        // sum it up
        aggregator.find(it.first)->second += it.second;
      }
    }
  }
}

std::pair<chunkQueue*, MatrixChunk*> allocateFreeQueue(size_t n) {

  // the queue where we put the chunks
  auto *queue = new chunkQueue();

  // allocate the memory for the chunks
  auto chunks = MatrixChunk::allocateMemory(n);

  // build up the queue
  #pragma omp parallel for
  for(size_t i = 0; i < n; ++i) {

    // the chunk we want to add
    auto currentChunk = chunks->getChunkAt(i);

    // add the chunk to the queue
    queue->enqueue(currentChunk);
  }

  // return the build free queue
  return std::make_pair(queue, chunks);
}

void joinSenderStage(CommunicatorPtr communicator,
                     int32_t node,
                     MatrixChunk *bValues,
                     matrixReverseIndexBTree *reverseB) {

  // register node
  reverseB->AssignGCID(node);

  // we just copy all the indices here
  std::vector<size_t> indices;

  // grab an iterator to the node
  reverseB->GetValue(node, indices);

  // unregister node
  reverseB->UnregisterThread(node);

  // send the counts
  communicator->send(indices.size(), node, COUNTS_TAG);

  // go through all the things we need to send
  for(auto &it : indices) {

    // grab the chunk index
    auto currentChunk = bValues->getChunkAt(it);

    // send the chunk
    communicator->send(currentChunk, 1, node, CHUNK_TAG);
  }

  std::cout << "Ended joinSenderStage" << std::endl;
}

void joinReceiverStage(CommunicatorPtr communicator,
                       chunkQueue *freeMultiplyQueue,
                       chunkQueue *multiplyQueue,
                       int32_t node,
                       atomic_int32_t *unfinishedNodes) {

  // grab the counts
  auto counts = communicator->recv<size_t>(node, COUNTS_TAG);

  for(size_t i = 0; i < counts; ++i) {

    // grab a free chunk
    MatrixChunk *chunk;
    freeMultiplyQueue->wait_dequeue(chunk);

    // grab the chunk
    communicator->recv(chunk, node, CHUNK_TAG);

    // add the chunk to the processing list
    multiplyQueue->enqueue(chunk);
  }

  std::cout << "Ended joinReceiverStage"<< std::endl;

  // we finished with this node this
  (*unfinishedNodes)--;
}

void multiplyStage(int32_t myNode,
                   int32_t threadID,
                   chunkQueue *multiplyQueue,
                   chunkQueue *freeQueue,
                   MatrixChunk *aValues,
                   matrixIndexBTree *aIndexed,
                   int32_t aNodeOffset,
                   BroadcastedIndices *aIndices,
                   atomic_int32_t *unfinishedJoinReceiverNodes,
                   atomic_int32_t *unfinishedMultiplyThreads,
                   chunkQueue *freeMultipliedQueue,
                   chunkQueue *multipliedQueue) {

  // register node
  aIndexed->AssignGCID(threadID);

  // wait to grab a matrix from the queue
  MatrixChunk *chunk;

  // wait to grab a matrix from the queue
  MatrixChunk *processingChunk;

  std::vector<double> tmp(chunkSize * chunkSize);

  // do this while you can
  while(true) {

    // try to grab something with a timeout of 10 us
    auto success = multiplyQueue->wait_dequeue_timed(chunk, 50000);

    // should we end this
    if (!success && (*unfinishedJoinReceiverNodes) == 0) {
      break;
    }

    // if we have a chunk
    if(success) {

      // grab the iterator
      auto it = aIndexed->Begin(chunk->rowID);

      // go through each match
      while (!it.IsEnd() && it->first == chunk->rowID) {

        // is this on our node
        if(aIndices->nodes[it->second] == myNode) {

          // wrap the B chunk in a gsl vector
          gsl_matrix_view b = gsl_matrix_view_array(chunk->getData(), chunkSize, chunkSize);

          // wrap the A chunk in a gsl vector
          auto blockOffset = (size_t)(it->second - aNodeOffset);
          gsl_matrix_view a = gsl_matrix_view_array(aValues->getChunkAt(blockOffset)->getData(), chunkSize, chunkSize);

          // grab a free processing chunk
          freeMultipliedQueue->wait_dequeue(processingChunk);

          // set the indices
          processingChunk->colID = chunk->colID;
          processingChunk->rowID = aIndices->rowIDs[it->second];

          // wrap the c block in a gsl vector
          gsl_matrix_view c = gsl_matrix_view_array(processingChunk->getData(), chunkSize, chunkSize);

          // do that multiply
          gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &a.matrix, &b.matrix, 0.0, &c.matrix);

          // forward the result to the aggregation thread
          multipliedQueue->enqueue(processingChunk);
        }

        it++;
      }

      // return it
      freeQueue->enqueue(chunk);
    }
  }

  // unregister this thread
  aIndexed->UnregisterThread(threadID);

  std::cout << "Ended joinProcessingStage"<< std::endl;

  // we finished the threads
  (*unfinishedMultiplyThreads)--;
}

bool processAggregation(std::map<std::pair<size_t, size_t>, std::vector<double>> *aggregatedMatrix,
                        std::map<std::pair<size_t, size_t>, std::mutex*> *aggregateLocks,
                        MatrixChunk* aggregationChunk) {

  // grab the key
  auto key = std::make_pair(aggregationChunk->rowID, aggregationChunk->colID);

  // is it on this node
  if (aggregateLocks->find(key) != aggregateLocks->end()) {

    // grab the lock
    aggregateLocks->find(key)->second->lock();

    // grab the aggregation chunk
    auto agg = aggregatedMatrix->find(key)->second.data();

    // chunk data
    auto chunkData = aggregationChunk->getData();

    for (int i = 0; i < chunkSize; ++i) {
      for (int j = 0; j < chunkSize; ++j) {
        agg[i * chunkSize + j] += chunkData[i * chunkSize + j];
      }
    }

    // unlock the chunk
    aggregateLocks->find(key)->second->unlock();

    // we aggregated this chunk
    return true;
  }

  // this chunk is not aggregated it needs to be send to another node
  return false;
}

void aggregationProcessingStage(std::map<std::pair<size_t, size_t>, std::vector<double>> *aggregatedMatrix,
                                std::map<std::pair<size_t, size_t>, std::mutex*> *aggregateLocks,
                                chunkQueue *freeProcessedQueue,
                                chunkQueue *processedQueue,
                                chunkQueue *sendingQueue,
                                chunkQueue *freeSendingQueue,
                                atomic_int32_t *unfinishedMultiplyThreads) {

  // wait to grab a matrix from the queue
  MatrixChunk* aggregationChunk;

  // do this while you can
  while(true) {

    // try to grab something with a timeout of 10 us
    auto success = processedQueue->wait_dequeue_timed(aggregationChunk, 50000);

    // should we end this
    if (!success && (*unfinishedMultiplyThreads) == 0) {
      break;
    }

    // if we have a chunk
    if(success) {

      // process the aggregation
      if(processAggregation(aggregatedMatrix, aggregateLocks, aggregationChunk)) {

        // add the chunk back to the free queue
        freeProcessedQueue->enqueue(aggregationChunk);

      } else {

        // grab a free chunk from the
        MatrixChunk *sendingChunk;
        freeSendingQueue->wait_dequeue(sendingChunk);

        // forward this chunk to the sender
        sendingQueue->enqueue(aggregationChunk);

        // add one chunk back to the sending queue
        freeProcessedQueue->enqueue(sendingChunk);
      }
    }
  }

  std::cout << "Ended aggregationProcessingStage " << std::endl;
}

void recievedAggregationProcessingStage(std::map<std::pair<size_t, size_t>, std::vector<double>> *aggregatedMatrix,
                                        std::map<std::pair<size_t, size_t>, std::mutex*> *aggregateLocks,
                                        chunkQueue *freeReceivedQueue,
                                        chunkQueue *receivedQueue,
                                        atomic_int32_t *unfinishedReceivedThreads) {

  // wait to grab a matrix from the queue
  MatrixChunk* aggregationChunk;

  // do this while you can
  while(true) {

    // try to grab something with a timeout of 10 us
    auto success = receivedQueue->wait_dequeue_timed(aggregationChunk, 50000);

    // should we end this
    if (!success && (*unfinishedReceivedThreads) == 0) {
      break;
    }

    // if we have a chunk
    if(success) {

      // process the aggregation
      processAggregation(aggregatedMatrix, aggregateLocks, aggregationChunk);

      // add the chunk back to the free queue
      freeReceivedQueue->enqueue(aggregationChunk);
    }
  }

  std::cout << "Ended aggregationProcessingStage" << std::endl;
}

void aggregationSender(CommunicatorPtr communicator,
                       chunkQueue *sendingQueue,
                       chunkQueue *freeSendingQueue,
                       atomic_int32_t *unfinishedMultiplyThreads) {

  // wait to grab a matrix from the queue
  MatrixChunk *aggregationChunk;

  // do this while you can
  while(true) {

    // try to grab something with a timeout of 10 us
    auto success = sendingQueue->wait_dequeue_timed(aggregationChunk, 50000);

    // should we end this
    if (!success && (*unfinishedMultiplyThreads) == 0) {
      break;
    }

    // if we have a chunk
    if(success) {

      auto node = (int32_t) ((aggregationChunk->rowID + aggregationChunk->colID) % communicator->getNumNodes());

      // send stuff
      communicator->send(aggregationChunk, 1, node, AGG_CHUNK_TAG);

      // return the sending chunk
      freeSendingQueue->enqueue(aggregationChunk);
    }
  }

  std::cout << "Ended aggregationSender" << std::endl;
}

void aggregationReceiver(CommunicatorPtr communicator,
                         int32_t node,
                         std::vector<atomic_int32_t> *counts,
                         chunkQueue *freeReceivedQueue,
                         chunkQueue *receivedQueue,
                         atomic_int32_t *unfinishedReceiverThreads) {

  // wait to grab a matrix from the queue
  MatrixChunk *chunk;

  // how may messages we need to receive
  int myCounts = (*counts)[node];

  for(int i = 0; i < myCounts; ++i) {

    // grab a free chunk
    freeReceivedQueue->wait_dequeue(chunk);

    // receive
    communicator->recv(chunk, node, AGG_CHUNK_TAG);

    // forward this to the
    receivedQueue->enqueue(chunk);
  }

  std::cout << "Ended aggregationReceiver" << myCounts << std::endl;

  // we finished
  (*unfinishedReceiverThreads)--;
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

  // chunks per dimension
  size_t chunksPerDimension = size / chunkSize;

  /// 1. Create and shuffle the matrices

  // this is where our stuff will be stored for matrix A
  std::vector<size_t> aRowIDs;
  std::vector<size_t> aColIDs;
  MatrixChunk *aValues;

  // create the identity lambda
  auto identityLambda = [](size_t i, size_t j) { return (size_t) (i + j); };
  createMatrix(identityLambda, communicator, size, chunkSize, &aValues, aRowIDs, aColIDs);

  // this is where our stuff will be stored for matrix A
  std::vector<size_t> bRowIDs;
  std::vector<size_t> bColIDs;
  MatrixChunk *bValues;

  // the matrix b will will fit a sequence of 0,1,2,3,4,5 row-wise
  auto sequenceLambda = [](size_t i, size_t j) { return i * size + j; };
  createMatrix(sequenceLambda, communicator, size, chunkSize, &bValues, bRowIDs, bColIDs);

  // start time
  auto start = std::chrono::steady_clock::now();

  /// 2. Broadcast all local copies of the indices to each node

  // initialize the indices for A
  BroadcastedIndices aIndices((size_t) communicator->getNumNodes(), chunksPerDimension * chunksPerDimension);
  broadcastAllIndices(communicator, aRowIDs, aColIDs, aIndices);

  // initialize the indices for b
  BroadcastedIndices bIndices((size_t) communicator->getNumNodes(), chunksPerDimension * chunksPerDimension);
  broadcastAllIndices(communicator, bRowIDs, bColIDs, bIndices);

  /// 3. Index the A matrix by column id

  // index the a matrix by column id
  auto aIndexed = indexByColumnID(aIndices);

  /// 4. Calculate the node offsets for the indices

  // grab my node id
  auto myNodeID = communicator->getNodeID();

  // calculate the A matrix node offset
  size_t aNodeOffset = getNodeOffset(aIndices, myNodeID);

  // calculate the B matrix node offset
  size_t bNodeOffset = getNodeOffset(bIndices, myNodeID);

  /// 5. Do the join and aggregation on the indices

  // aggregator map
  std::map<std::pair<size_t, size_t>, int> aggregator;

  // this tells us how many multiplied chunks we are going to have
  std::vector<atomic_int32_t> sentMultiCounts((size_t) communicator->getNumNodes());

  // create a btree
  auto bReverseIndexed = new matrixReverseIndexBTree{true};

  // run the thing
  preprocessJoinAndAggregation(myNodeID,
                               communicator->getNumNodes(),
                               aIndices,
                               bIndices,
                               aIndexed,
                               bNodeOffset,
                               aggregator,
                               bReverseIndexed,
                               sentMultiCounts);

  /// 6. Allocate the memory for the aggregation

  // aggregator map
  std::map<std::pair<size_t, size_t>, std::vector<double>> aggregateMatrix;
  std::map<std::pair<size_t, size_t>, std::mutex*> aggregateLocks;

  for(auto &it : aggregator) {

    // figure out on which node this belongs to
    auto onNode = (it.first.first + it.first.second) % communicator->getNumNodes();

    // is this our node
    if(onNode == communicator->getNodeID()) {

      // make a new vector
      aggregateMatrix.emplace(it.first, std::vector<double>());

      // make a new lock
      aggregateLocks.emplace(it.first, new std::mutex());

      // allocate the right chunk size
      aggregateMatrix.find(it.first)->second.resize(chunkSize * chunkSize);
    }
  }

  /// 7. Setup the pear to pear communication

  std::vector<std::thread*> threads;

  // these two queues are used together one has the chunks that need to be multiplied and the other the free chunks
  auto freeMultiplyQueue = allocateFreeQueue((size_t) 2 * std::max(omp_get_max_threads(), communicator->getNumNodes()));
  auto multiplyQueue = new chunkQueue();

  // these two queues are used together one has the chunks that need to be multiplied and the other the free chunks
  auto freeMultipliedQueue = allocateFreeQueue((size_t) 2 * std::max(omp_get_max_threads(), communicator->getNumNodes()));
  auto multipliedQueue = new chunkQueue();

  // these two queues are used together one is used to forward chunks to the sending thread
  auto freeSendingQueue = allocateFreeQueue((size_t) 2 * std::max(omp_get_max_threads(), communicator->getNumNodes()));
  auto sendingQueue = new chunkQueue();

  // these two queues are used together one is used to forward chunks to the sending thread
  auto freeReceivedQueue = allocateFreeQueue((size_t) 2 * std::max(omp_get_max_threads(), communicator->getNumNodes()));
  auto receivedQueue = new chunkQueue();

  // how many nodes are not done executing
  atomic_int32_t unfinishedJoinReceivers;
  unfinishedJoinReceivers = communicator->getNumNodes();

  atomic_int32_t unfinishedMultiplyThreads;
  unfinishedMultiplyThreads = resourceManager->getNumCores();

  atomic_int32_t unfinishedReceiverThreads;
  unfinishedReceiverThreads = communicator->getNumNodes();

  // this part is taking in
  for(int i = 0; i < communicator->getNumNodes(); ++i) {

    // create the threads
    auto *joinSenderStageThread = new std::thread(joinSenderStage, communicator, i, bValues, bReverseIndexed);
    auto *joinReceiveStageThread = new std::thread(joinReceiverStage, communicator, freeMultiplyQueue.first, multiplyQueue, i, &unfinishedJoinReceivers);

    // store it in the vector
    threads.push_back(joinSenderStageThread);
    threads.push_back(joinReceiveStageThread);
  }

  // for each core create a matrix processing thread this thread puts stuff in the multipliedQueue
  for(int i = 0; i < resourceManager->getNumCores(); ++i) {

    auto *joinMultiplyThread = new std::thread(multiplyStage,
                                               myNodeID,
                                               i,
                                               multiplyQueue,
                                               freeMultiplyQueue.first,
                                               aValues,
                                               aIndexed,
                                               aNodeOffset,
                                               &aIndices,
                                               &unfinishedJoinReceivers,
                                               &unfinishedMultiplyThreads,
                                               freeMultipliedQueue.first,
                                               multipliedQueue);

    // store it in the vector
    threads.push_back(joinMultiplyThread);
  }

  // for each core create an aggregation thread that is going to do a local aggregation or forward the chunk to a
  // sending queue that is going to forward it to the right node
  for(int i = 0; i < resourceManager->getNumCores(); ++i) {


    // init the aggregation thread
    auto *aggregationProcessingThread = new std::thread(aggregationProcessingStage,
                                                        &aggregateMatrix,
                                                        &aggregateLocks,
                                                        freeMultipliedQueue.first,
                                                        multipliedQueue,
                                                        sendingQueue,
                                                        freeSendingQueue.first,
                                                        &unfinishedMultiplyThreads);
    // store it in the vector
    threads.push_back(aggregationProcessingThread);
  }

  // go through each node
  for(int i = 0; i < communicator->getNumNodes(); ++i) {

    // init the aggregation sender thread
    auto *aggregationSenderThread = new std::thread(aggregationSender,
                                                    communicator,
                                                    sendingQueue,
                                                    freeSendingQueue.first,
                                                    &unfinishedMultiplyThreads);

    // init the receiver thread
    auto *aggregationReceiverThread = new std::thread(aggregationReceiver,
                                                      communicator,
                                                      i,
                                                      &sentMultiCounts,
                                                      freeReceivedQueue.first,
                                                      receivedQueue,
                                                      &unfinishedReceiverThreads);

    // store it in the vector
    threads.push_back(aggregationSenderThread);
    threads.push_back(aggregationReceiverThread);
  }

  // for each core create an aggregation thread that is going to do a local aggregation on chunks received from other nodes
  for(int i = 0; i < resourceManager->getNumCores(); ++i) {


    // init the aggregation thread
    auto *aggregationProcessingThread = new std::thread(recievedAggregationProcessingStage,
                                                        &aggregateMatrix,
                                                        &aggregateLocks,
                                                        freeReceivedQueue.first,
                                                        receivedQueue,
                                                        &unfinishedReceiverThreads);
    // store it in the vector
    threads.push_back(aggregationProcessingThread);
  }

  // go through each thread and wait for it to finish
  for(auto &i : threads) {

    // wait for it to finish
    i->join();

    // free the memory
    delete(i);
  }

  // end time
  auto end = std::chrono::steady_clock::now();

  // add the total time
  std::cout << "Finished : " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;

  // delete the queues
  delete freeMultiplyQueue.first;
  delete multiplyQueue;
  delete freeMultipliedQueue.first;
  delete multipliedQueue;
  delete freeSendingQueue.first;
  delete sendingQueue;
  delete freeReceivedQueue.first;
  delete receivedQueue;
  delete aIndexed;

  // free the queue memory
  free(freeMultiplyQueue.second);
  free(freeMultipliedQueue.second);
  free(freeSendingQueue.second);
  free(freeReceivedQueue.second);
  free(aValues);
  free(bValues);

  // delete the locks
  for(auto l : aggregateLocks) {
    delete l.second;
  }

#ifdef DEBUG_ME

  sleep(communicator->getNumNodes());

  for(auto &it : aggregateMatrix) {

    std::cout << it.first.first << ", " << it.first.second << ":" << std::endl;

    std::cout << "[";

    for(int i = 0; i < chunkSize; ++i) {

      std::cout << "[";

      for(int j = 0; j <  chunkSize; ++j) {
        std::cout << it.second[i * chunkSize + j] << " ";
      }

      std::cout << "]";
    }

    std::cout << "]" << std::endl;
  }

#endif

}