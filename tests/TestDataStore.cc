#include <mpi.h>
#include <stdio.h>
#include <Communicator.h>
#include <DataStore.h>
#include <AbstractLogger.h>
#include <StandardOutputLogger.h>
#include <Server.h>
#include <ResourceManager.h>

int main() {

  // create data store
  DataStorePtr dataStore = (new DataStore())->getHandle();

  // create a set
  //SetSchemaPtr setIdentifier = make_shared<>

}