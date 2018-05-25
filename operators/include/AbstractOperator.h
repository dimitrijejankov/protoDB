//
// Created by dimitrije on 5/25/18.
//

#ifndef PROTODB_ABSTRACTOPERATOR_H
#define PROTODB_ABSTRACTOPERATOR_H

#include <vector>
#include "ColumnSpecification.h"

class AbstractOperator {

  /**
   * The specifications for the
   */
  std::vector<ColumnSpecification> outputColumns;


};

#endif //PROTODB_ABSTRACTOPERATOR_H
