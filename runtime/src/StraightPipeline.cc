#include "StraightPipeline.h"

StraightPipeline::StraightPipeline(SetSchemaPtr outputSchema) : outputSchema(outputSchema) {}

void StraightPipeline::execute() {

  // grab the columns
  auto columns = dataStore->getColumns(inputSchema->getSchemaIdentifier());

}


