//
// Created by dimitrije on 5/24/18.
//

#include "SetSchema.h"

std::string SetSchema::getSchemaIdentifier() {
  return databaseName + ":" + setName;
}

const std::unordered_map<std::string, AttributeType> &SetSchema::getAttributes() const {
  return attributes;
}
