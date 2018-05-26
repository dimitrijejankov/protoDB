//
// Created by dimitrije on 5/26/18.
//

#ifndef PROTODB_BLOCKMANAGER_H
#define PROTODB_BLOCKMANAGER_H

#include <cstdint>
#include <vector>
#include <mutex>
#include "Block.h"

class BlockManager {

public:

  /**
   * The block manager, basically this calls is used to grab some memory once we need it
   * @param blockSize - the size of one block
   * @param numBlocks - the number of blocks we have available
   */
  BlockManager(size_t blockSize, size_t numBlocks);

  /**
   * The destructor of the block manager
   */
  ~BlockManager();

  /**
   * Returns a new free block
   * @return the block
   */
  Block& grabBlock();

  /**
   * Returns the block back to the block manager
   */
  void freeBlock(Block& block);

protected:

  /**
   * The size of the block
   */
  size_t blockSize;

  /**
   * The number of blocks
   */
  size_t numBlocks;

  /**
   * The raw blocks in the block manager
   */
  int8_t** rawBlocks;

  /**
   * List with all the free blocks
   */
  std::vector<Block> freeList;

  /**
   * Mutex to synchronize the access to the blocks
   */
  std::mutex mutex;

};

#endif //PROTODB_BLOCKMANAGER_H
