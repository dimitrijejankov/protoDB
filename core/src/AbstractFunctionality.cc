#include "AbstractFunctionality.h"

#include <utility>

AbstractFunctionalityPtr AbstractFunctionality::getHandle() {

  // if we do not have a pointer set
  if(handle.use_count() == 0) {

    // we make a shared pointer out of this object
    auto it = std::shared_ptr<AbstractFunctionality>(this);
    handle = it;

    // return the shared pointer
    return handle.lock();
  }

  // return the shared pointer because we already have it
  return handle.lock();
}

void AbstractFunctionality::assignToServer(ServerWeakPtr server) {
  serverHandle = std::move(server);
}
