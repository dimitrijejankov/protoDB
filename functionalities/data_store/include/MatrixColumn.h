//
// Created by dimitrije on 5/24/18.
//

#ifndef PROTODB_MATRIXCOLUMN_H
#define PROTODB_MATRIXCOLUMN_H

#include "AbstractColumn.h"

class MatrixColumn : public AbstractColumn  {

public:

  explicit MatrixColumn(const std::string &name);

};

#endif //PROTODB_MATRIXCOLUMN_H
