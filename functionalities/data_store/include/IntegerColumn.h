//
// Created by dimitrije on 5/24/18.
//

#ifndef PROTODB_INTEGERCOLUMN_H
#define PROTODB_INTEGERCOLUMN_H

#include <vector>
#include <string>
#include <cstdint>
#include "AbstractColumn.h"

class IntegerColumn : public AbstractColumn {

public:

  /**
   * Creates the integer column
   * @param name - the name of the column
   */
  explicit IntegerColumn(const std::string &name);

  /**
   * Creates the integer column with a name and a predefined capacity
   * @param name
   * @param capacity
   */
  IntegerColumn(const std::string &name, size_t capacity);

 private:

  /**
   * The values of this column
   */
  std::vector<int32_t> values;

};

#endif //PROTODB_INTEGERCOLUMN_H
