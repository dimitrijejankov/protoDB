#ifndef PROTODB_PIPELINE_H
#define PROTODB_PIPELINE_H

#include <SetSchema.h>
#include <DataStore.h>

class Pipeline {

public:

  /**
   * Initializes the pipeline with the output schema
   * @param outputSchema - the output schema
   */
  explicit Pipeline(const SetSchemaPtr &outputSchema);

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
