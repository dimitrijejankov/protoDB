//
// Created by dimitrije on 5/24/18.
//


#ifndef PROTODB_DATASTORE_H
#define PROTODB_DATASTORE_H

#include <unordered_map>
#include <vector>
#include "AbstractColumn.h"
#include "SetSchema.h"

typedef std::vector<AbstractColumnPtr> Columns;

class DataStore {


  /**
   * This links the set with their respective columns
   */
  std::unordered_map<std::string, Columns> sets;

  /**
   * This links the sets with the respective schema
   */
  std::unordered_map<std::string, SetSchema> schema;

  /**
   * Returns the appropriate columns for a particular set
   * @return the columns
   */
  Columns getColumns(const std::string &setName);

  /**
   * Returns the schema for a particular set
   * @return the schema
   */
  SetSchema getSchema(const std::string &setName);

  /**
   * This adds a set to the data store and creates the columns for it
   * @param schema - the schema
   */
  void addSet(SetSchema schema);

};

#endif //PROTODB_DATASTORE_H
