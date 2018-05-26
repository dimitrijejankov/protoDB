#include "StraightPipeline.h"

StraightPipeline::StraightPipeline(SetSchemaPtr outputSchema) : Pipeline(outputSchema) {}

void StraightPipeline::execute() {

  // grab the columns
  auto columns = dataStore->getColumns(inputSchema->getSchemaIdentifier());


  /// TODO this part is supposed to be templetized



  /// TODO end

}


