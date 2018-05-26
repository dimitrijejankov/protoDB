//
// Created by dimitrije on 5/26/18.
//

#ifndef PROTODB_BLOCK_H
#define PROTODB_BLOCK_H

#include <cstdint>
#include <cstdio>

class Block {

public:

  Block(size_t blockSize, int8_t *block);

  /**
   * This method gives us the block in the appropriate type
   * @tparam T - the type of the records stored in this block
   * @return - the raw block
   */
  template<typename T>
  T* getRawBlock() {

    // cast the thing to the appropriate type
    return (T*) block;
  }

  /**
   * Returns the size of the block
   * @return the block size
   */
  size_t getBlockSize();

private:

  /**
   * The block size
   */
  size_t blockSize;

  /**
   * A pointer to the blocks memory
   */
  int8_t *block;

};

#endif //PROTODB_BLOCK_H
