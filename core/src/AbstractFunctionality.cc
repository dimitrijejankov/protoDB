#include "AbstractFunctionality.h"

AbstractFunctionalityPtr AbstractFunctionality::getHandle() {

  // if we do not have a pointer set
  if(node.use_count() == 0) {

    // we make a shared pointer out of this object
    auto it = std::shared_ptr<AbstractFunctionality>(this);
    node = it;

    // return the shared pointer
    return node.lock();
  }

  // return the shared pointer because we already have it
  return node.lock();
}
