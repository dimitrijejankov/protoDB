#include <cstdlib>
#include "BlockManager.h"

BlockManager::BlockManager(size_t blockSize, size_t numBlocks) {

  // set the sizes
  this->numBlocks = numBlocks;
  this->blockSize = blockSize;

  // initialize the array that holds the blocks
  rawBlocks = (int8_t**)malloc(numBlocks);

  // reserve the space for the block objects
  freeList.reserve(numBlocks);

  // initialize the blocks
  for(auto i = 0; i < numBlocks; i++) {

    // initialize the i-th block
    rawBlocks[i] = (int8_t*)malloc(blockSize * sizeof(int8_t));

    // add the block to the free list
    freeList.emplace_back(Block(this->blockSize, rawBlocks[i]));
  }
}

BlockManager::~BlockManager() {

  // free the blocks
  for(auto i = 0; i < numBlocks; i++) {

    // free the i-th block
    free(rawBlocks[i]);
  }

  // free the block container
  free(rawBlocks);
}

Block &BlockManager::grabBlock() {

  // lock the access
  std::lock_guard<std::mutex> lck (mutex);

  // return the last block
  auto &block = freeList.back();

  // remove the last block
  freeList.pop_back();

  // return the block
  return block;
}

void BlockManager::freeBlock(Block &block) {

  // lock the access
  std::lock_guard<std::mutex> lck (mutex);

  // add the block back to the list
  freeList.push_back(block);
}
