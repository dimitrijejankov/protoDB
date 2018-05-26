//
// Created by dimitrije on 5/24/18.
//

#include "SetSchema.h"

std::string SetSchema::getSchemaIdentifier() {
  return databaseName + ":" + setName;
}
