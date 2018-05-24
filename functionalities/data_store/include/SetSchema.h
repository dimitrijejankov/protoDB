#ifndef PROTODB_SETSCHEMA_H
#define PROTODB_SETSCHEMA_H

#include <unordered_map>

/**
 * The type of attribute
 */
enum AttributeType {
  INT_TYPE,
  MATRIX_TYPE
};

/**
 * Represents the set schema
 * currently supports only int and matrix types
 */
class SetSchema {

  /**
   * The attributes that are in the schema
   */
  std::unordered_map<std::string, AttributeType> attributes;

  /**
   * The name of the schema
   */
  std::string name;
};

#endif //PROTODB_SETSCHEMA_H
