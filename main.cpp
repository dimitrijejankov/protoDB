#include <mpi.h>
#include <stdio.h>
#include <Communicator.h>
#include <DataStore.h>
#include <AbstractLogger.h>
#include <StandardOutputLogger.h>
#include <Server.h>

int main() {

  Server server;

  // create the communicator
  CommunicatorPtr communicator = (new Communicator())->getHandle()->to<Communicator>();
  server.addFunctionality(communicator);

  // create data store
  AbstractFunctionalityPtr dataStore = (new DataStore())->getHandle();
  server.addFunctionality(dataStore);

  // if this is the master
  if(communicator->getNodeID() == 0) {

    // create the logger
    AbstractFunctionalityPtr logger = (new StandardOutputLogger("Master"))->getHandle();
    server.addFunctionality(logger);
  }
  else {

    // create the logger
    AbstractFunctionalityPtr logger = (new StandardOutputLogger("Worker"))->getHandle();
    server.addFunctionality(logger);
  }

  // run the server
  server.run();
}