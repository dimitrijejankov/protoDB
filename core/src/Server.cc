#include <Communicator.h>
#include <AbstractLogger.h>
#include "Server.h"

void Server::addFunctionality(AbstractFunctionalityPtr fun) {
  this->functionalities[fun->getType()] = fun;
}

AbstractFunctionalityPtr Server::getFunctionality(FunctionalityType type) {
  return this->functionalities[type];
}

void Server::run() {

  auto logger = getFunctionality(LOGGER)->to<AbstractLogger>();
  auto communicator = getFunctionality(COMMUNICATOR)->to<Communicator>();

  // print the hallo world form the logger
  logger->info() << "Hello world from processor " << communicator->getNodeName().c_str() << ", rank "
                 << communicator->getNodeID() << " out of " << communicator->getNumNodes() << " processors"
                 << logger->endl;
}

