#ifndef PROTODB_DATASTORE_H
#define PROTODB_DATASTORE_H

#include <unordered_map>
#include <vector>
#include "AbstractFunctionality.h"
#include "AbstractColumn.h"
#include "SetSchema.h"

typedef std::vector<AbstractColumnPtr> Columns;

/**
 * The shared pointer to the data store
 */
class DataStore;
typedef std::shared_ptr<DataStore> DataStorePtr;

class DataStore : public AbstractFunctionality {

public:

  /**
   * Just the default destructor
   */
  ~DataStore() override = default;

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

  /**
   * Returns the type of the DataStore functionality
   * @return the type
   */
  FunctionalityType getType() override;

private:

  /**
   * This links the set with their respective columns
   */
  std::unordered_map<std::string, Columns> sets;

  /**
   * This links the sets with the respective schema
   */
  std::unordered_map<std::string, SetSchema> schemas;

};

#endif //PROTODB_DATASTORE_H
