#ifndef PROTODB_PIPELINE_H
#define PROTODB_PIPELINE_H

#include <SetSchema.h>
#include <DataStore.h>

class Pipeline {

public:

  /**
   * Execute this pipeline
   */
  virtual void execute() = 0;

protected:

  /**
   * The output schema
   */
  SetSchemaPtr outputSchema;

  /**
   * The data store where our stuff is
   */
  DataStorePtr dataStore;
};

#endif //PROTODB_PIPELINE_H
