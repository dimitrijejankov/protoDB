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

  // print off a hello world message
  logger->info("Hello world from processor");
  logger->info(std::to_string(communicator->getNodeID()));
  logger->info(std::to_string(communicator->getNumNodes()));
  logger->info(communicator->getNodeName());
}

