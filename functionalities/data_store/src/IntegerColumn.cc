//
// Created by dimitrije on 5/24/18.
//

#include "IntegerColumn.h"

IntegerColumn::IntegerColumn(const std::string &name) : AbstractColumn(name) {}

IntegerColumn::IntegerColumn(const std::string &name, size_t capacity) : AbstractColumn(name) {
  values.reserve(capacity);
}
