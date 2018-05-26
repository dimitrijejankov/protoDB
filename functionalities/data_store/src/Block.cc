#include "Block.h"

Block::Block(size_t blockSize, int8_t *block) : blockSize(blockSize), block(block) {}

size_t Block::getBlockSize() {
  return blockSize;
}

