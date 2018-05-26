#ifndef PROTODB_STRAIGHTPIPELINE_H
#define PROTODB_STRAIGHTPIPELINE_H

#include "Pipeline.h"

/**
 * Straight pipeline is a pipeline that runs locally
 */
class StraightPipeline : public Pipeline {

public:

  explicit StraightPipeline(SetSchemaPtr outputSchema);

  /**
   * Executes this straight pipeline
   */
  void execute() override;

protected:

  /**
   * The input schema
   */
  SetSchemaPtr inputSchema;

};

#endif //PROTODB_STRAIGHTPIPELINE_H
