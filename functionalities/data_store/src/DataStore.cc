//
// Created by dimitrije on 5/24/18.
//

#include "IntegerColumn.h"
#include "DataStore.h"

Columns DataStore::getColumns(const std::string &setName) {
  return Columns();
}

SetSchema DataStore::getSchema(const std::string &setName) {
  return SetSchema();
}

void DataStore::addSet(SetSchema schema) {

  // get the set identifier
  auto identifier = schema.getSchemaIdentifier();

  // store the schema
  schemas[identifier] = schema;

  // columns we need to add
  Columns columns;

  // go through each column and create the appropriate
  for(auto &it : schema.getAttributes()) {

    // ok so we have to check which attribute type we are dealing with
    switch (it.second) {
      MATRIX_TYPE :
        columns.push_back(std::make_shared<IntegerColumn>());
        break;

      INT_TYPE:
        break;
    }

  }
}

FunctionalityType DataStore::getType() {
  return DATA_STORE;
}
