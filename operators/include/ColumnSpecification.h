//
// Created by dimitrije on 5/25/18.
//

#ifndef PROTODB_ATTRIBUTESPECIFICATION_H
#define PROTODB_ATTRIBUTESPECIFICATION_H

#include <string>
#include <AbstractColumn.h>

class ColumnSpecification {

  /**
   * The name of the set this attribute belongs to
   */
  std::string setIdentifier;

  /**
   * The name of the column
   */
  std::string columnIdentifier;

  /**
   * The type of the column
   */
  ColumnType columnType;

};

#endif //PROTODB_ATTRIBUTESPECIFICATION_H
