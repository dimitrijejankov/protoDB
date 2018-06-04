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
#include <mutex>

const int32_t ROW_IDX_TAG = 1;
const int32_t COL_IDX_TAG = 2;
const int32_t DOUBLE_TAG = 3;
const int32_t COUNTS_TAG = 4;

// dimensions of the matrices A and B
size_t size = 4;
size_t chunkSize = 2;

typedef protoDB::bwtree::BwTree<size_t, int> matrixIndexBTree;
typedef protoDB::bwtree::BwTree<int, size_t> matrixReverseIndexBTree;

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

void receiveRandomShuffled(CommunicatorPtr &communicator,
                           std::vector<size_t> &rowIDs,
                           std::vector<size_t> &colIDs,
                           std::vector<double> &values,
                           size_t chunkSize) {

  // grab the number of tuples
  auto num = communicator->recv<size_t>(0, COUNTS_TAG);

  // reserve the approximate amount of memory
  rowIDs.reserve(num);
  colIDs.reserve(num);
  values.reserve(chunkSize * chunkSize * num);

  // tmp chunk
  std::vector<double> tmp;

  // receive each value
  for(int i = 0; i < num; ++i) {

    // grab the rows
    auto rowID = communicator->recv<size_t>(0, ROW_IDX_TAG);

    // grab the columns
    auto colID = communicator->recv<size_t>(0, COL_IDX_TAG);

    // grab the double chunk
    communicator->recv(tmp, 0, DOUBLE_TAG);

    // copy the chunk
    rowIDs.push_back(rowID);
    colIDs.push_back(colID);
    values.insert(values.end(), tmp.begin(), tmp.end());
  }
}

void generateMatrix(size_t (*valueFunc)(size_t, size_t), size_t size, size_t chunkSize, CommunicatorPtr communicator) {

  // figure out the chunk size
  size_t chunksPerDimension = size / chunkSize;

  // grab the number of nodes
  auto numNodes = (size_t) communicator->getNumNodes();

  /// 1. Generate a random permutation for each node and store the counts

  // allocate the memory for the permutation
  std::vector<int32_t> permutation(chunkSize * chunkSize);

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
  std::vector<double> tmp = std::vector<double>(chunkSize * chunkSize);

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
          tmp[valueIndex] = valueFunc(c_i * chunkSize + i, c_j * chunkSize + j);
        }
      }

      // send the chunk
      communicator->send(c_i, permutation[c_i * chunksPerDimension + c_j], ROW_IDX_TAG);
      communicator->send(c_j, permutation[c_i * chunksPerDimension + c_j], COL_IDX_TAG);
      communicator->send(tmp.data(), tmp.size(), permutation[c_i * chunksPerDimension + c_j], DOUBLE_TAG);
    }
  }
}

void createMatrix(size_t (*valueFunc)(size_t, size_t),
                  CommunicatorPtr &communicator,
                  size_t &size,
                  size_t chunkSize,
                  std::vector<double> &values,
                  std::vector<size_t> &rowIDs,
                  std::vector<size_t> &colIDs) {

  if (communicator->isMaster()) {

    // create the matrix and random shuffle it
    std::thread shuffleReceiveAndThread(generateMatrix, valueFunc, size, chunkSize, communicator);

    // receive the shuffled data
    receiveRandomShuffled(communicator, rowIDs, colIDs, values, chunkSize);

    // wait for the shuffle to finish
    shuffleReceiveAndThread.join();
  } else {

    // receive the shuffled data
    receiveRandomShuffled(communicator, rowIDs, colIDs, values, chunkSize);
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

void joinSenderStage(CommunicatorPtr communicator,
                     int32_t node,
                     size_t chunkSize,
                     std::vector<size_t> *bRowIDs,
                     std::vector<size_t> *bColIDs,
                     std::vector<double> *bValues,
                     matrixReverseIndexBTree *reverseB) {

  // register node
  reverseB->AssignGCID(node);

  // get squared chunk size
  auto chunkSquared = chunkSize * chunkSize;

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
    auto chunkIndex = it;

    // send stuff
    communicator->send((*bRowIDs)[chunkIndex], node, ROW_IDX_TAG);
    communicator->send((*bColIDs)[chunkIndex], node, COL_IDX_TAG);
    communicator->send(&bValues->data()[chunkSquared * chunkIndex], chunkSquared, node, DOUBLE_TAG);
  }
}

void joinReceiverStage(CommunicatorPtr communicator,
                       int32_t node,
                       size_t chunkSize) {

  // get squared chunk size
  auto chunkSquared = chunkSize * chunkSize;
  std::vector<double> tmp(chunkSquared);

  auto counts = communicator->recv<size_t>(node, COUNTS_TAG);

  for(size_t i = 0; i < counts; ++i) {

    auto rowID = communicator->recv<size_t>(node, ROW_IDX_TAG);
    auto colID = communicator->recv<size_t>(node, COL_IDX_TAG);

    communicator->recv(tmp, node, DOUBLE_TAG);
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

  // chunks per dimension
  size_t chunksPerDimension = size / chunkSize;

  // this is where our stuff will be stored for matrix A
  std::vector<size_t> aRowIDs;
  std::vector<size_t> aColIDs;
  std::vector<double> aValues;

  // this is where our stuff will be stored for matrix B
  std::vector<size_t> bRowIDs;
  std::vector<size_t> bColIDs;
  std::vector<double> bValues;

  // create the identity lambda
  auto identityLambda = [](size_t i, size_t j) { return (size_t) (i == j); };

  // the matrix b will will fit a sequence of 0,1,2,3,4,5 row-wise
  auto sequenceLambda = [](size_t i, size_t j) { return i * size + j; };

  // create the matrices
  createMatrix(identityLambda, communicator, size, chunkSize, aValues, aRowIDs, aColIDs);
  createMatrix(sequenceLambda, communicator, size, chunkSize, bValues, bRowIDs, bColIDs);

  /// 1. Broadcast all the indices to every node

  // initialize the indices for A
  BroadcastedIndices aIndices((size_t) communicator->getNumNodes(), size);
  broadcastAllIndices(communicator, aRowIDs, aColIDs, aIndices);

  // initialize the indices for b
  BroadcastedIndices bIndices((size_t) communicator->getNumNodes(), size);
  broadcastAllIndices(communicator, bRowIDs, bColIDs, bIndices);

  /// 2. Join and Aggregate to get the indices of the final result

  // create a btree
  auto aIndexed = new matrixIndexBTree{true};

  // set the maximum number of threads
  aIndexed->UpdateThreadLocal((size_t) omp_get_max_threads());

  // do the parallel insert
  #pragma omp parallel
  {

    // register the thread
    aIndexed->AssignGCID(omp_get_thread_num());

    #pragma omp for
    for(int i = 0; i < aIndices.rowIDs.size(); ++i)
    {
      // store the thing into the btree
      aIndexed->Insert(aIndices.colIDs[i], i);
    }

    // unregister the thread
    aIndexed->UnregisterThread(omp_get_thread_num());
  }

  // aggregator map
  std::map<std::pair<size_t, size_t>, int> aggregator;

  // create a btree
  auto bReverseIndexed = new matrixReverseIndexBTree{true};

  // set the number of threads
  bReverseIndexed->UpdateThreadLocal((size_t) omp_get_max_threads());

  // grab my node id
  auto myNodeID = communicator->getNodeID();

  // calculate the node offset
  size_t bNodeOffset = 0;
  for(auto i = 0; i < myNodeID; ++i) {
    bNodeOffset += bIndices.nodeCounts[i];
  }

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

        // grab the row id
        auto rowID = aIndices.rowIDs[it->second];

        // we are joining two tuples
        auto key = std::make_pair(rowID, bIndices.colIDs[i]);

        // is this on my node
        if(bIndices.nodes[i] == myNodeID) {

          // insert the reverse index
          bReverseIndexed->Insert(bIndices.nodes[i], i - bNodeOffset);
        }

        // if we don't have it set it to 0
        localAggregator.try_emplace(key, 0);

        // increment the thing
        localAggregator.find(key)->second =+ 1;

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
        aggregator.try_emplace(it.first, 0);

        // sum it up
        aggregator.find(it.first)->second += it.second;
      }
    }
  }

  /// 3. Preallocate the memory for the aggregation

  // aggregator map
  std::map<std::pair<size_t, size_t>, std::vector<double>> aggregateMatrix;

  for(auto &it : aggregator) {

    // figure out on which node this belongs to
    auto onNode = (it.first.first + it.first.second) % communicator->getNumNodes();

    // is this our node
    if(onNode == communicator->getNodeID()) {

      // make a new vector
      aggregateMatrix.emplace(it.first, std::vector<double>());

      // allocate the right chunk size
      aggregateMatrix.find(it.first)->second.resize(chunkSize * chunkSize);
    }
  }

  /// 4. Setup the pear to pear communication

  std::vector<std::thread*> threads;

  // go through each node
  for(int i = 0; i < communicator->getNumNodes(); ++i) {

    // create the threads
    auto *joinSenderStageThread = new std::thread(joinSenderStage, communicator, i, chunkSize, &bRowIDs, &bColIDs, &bValues, bReverseIndexed);
    auto *joinReceiveStageThread = new std::thread(joinReceiverStage, communicator, i, chunkSize);

    // store it in the vector
    threads.push_back(joinSenderStageThread);
    threads.push_back(joinReceiveStageThread);
  }

  // go through each thread and wait for it to finish
  for(auto &i : threads) {

    // wait for it to finish
    i->join();

    // free the memory
    delete(i);
  }

}
