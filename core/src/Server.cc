#include <Communicator.h>
#include <AbstractLogger.h>
#include <ResourceManager.h>
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
  auto resourceManager = getFunctionality(RESOURCE_MANAGER)->to<ResourceManager>();

  // print the hallo world form the logger
  logger->info() << "Hello world from processor " << communicator->getNodeName().c_str() << ", rank "
                 << communicator->getNodeID() << " out of " << communicator->getNumNodes() << " processors"
                 << logger->endl;

  // print out the stats about the machine
  logger->info() << "Memory on the machine " << resourceManager->getTotalMemory() << " and number of cores "
                 << resourceManager->getNumCores() << logger->endl;
}

