#include <Communicator.h>
#include <AbstractLogger.h>
#include <ResourceManager.h>
#include "Server.h"

void Server::addFunctionality(AbstractFunctionalityPtr fun) {

  // register the functionality
  this->functionalities[fun->getType()] = fun;

  // give the functionality a weak handle of this server
  fun->assignToServer(getHandle());
}

AbstractFunctionalityPtr Server::getFunctionality(FunctionalityType type) {

  // return the requested functionality if exists
  return this->functionalities[type];
}

ServerPtr Server::getHandle() {

  // if we do not have a pointer set
  if(handle.use_count() == 0) {

    // we make a shared pointer out of this object
    auto it = std::shared_ptr<Server>(this);
    handle = it;

    // return the shared pointer
    return handle.lock();
  }

  // return the shared pointer because we already have it
  return handle.lock();
}

void Server::run() {

  auto logger = getFunctionality(LOGGER)->to<AbstractLogger>();
  auto communicator = getFunctionality(COMMUNICATOR)->to<Communicator>();
  auto resourceManager = getFunctionality(RESOURCE_MANAGER)->to<ResourceManager>();

  // print the hallo world form the logger
  logger->info() << "Hello world from processor " << communicator->getNodeName().c_str() << ", rank "
                 << communicator->getNodeID() << " out of " << communicator->getNumNodes() << " nodes"
                 << logger->endl;

  // print out the stats about the machine
  logger->info() << "Memory on the machine " << resourceManager->getTotalMemory() << " and number of cores "
                 << resourceManager->getNumCores() << logger->endl;
}



