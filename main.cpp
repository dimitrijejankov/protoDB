#include <mpi.h>
#include <stdio.h>
#include <Communicator.h>
#include <DataStore.h>
#include <AbstractLogger.h>
#include <StandardOutputLogger.h>
#include <Server.h>
#include <ResourceManager.h>

int main() {

  ServerPtr server = (new Server())->getHandle();

  // create the communicator
  CommunicatorPtr communicator = (new Communicator())->getHandle()->to<Communicator>();
  server->addFunctionality(communicator);

  // create data store
  AbstractFunctionalityPtr dataStore = (new DataStore())->getHandle();
  server->addFunctionality(dataStore);

  // create the resource manager
  AbstractFunctionalityPtr resourceManager = (new ResourceManager())->getHandle();
  server->addFunctionality(resourceManager);

  // if this is the master
  if(communicator->getNodeID() == 0) {

    // create the logger
    AbstractFunctionalityPtr logger = (new StandardOutputLogger("Master"))->getHandle();
    server->addFunctionality(logger);
  }
  else {

    // create the logger
    AbstractFunctionalityPtr logger = (new StandardOutputLogger("Worker"))->getHandle();
    server->addFunctionality(logger);
  }

  // run the server
  server->run();
}