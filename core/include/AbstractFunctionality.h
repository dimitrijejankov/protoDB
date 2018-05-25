#ifndef PROTODB_ABSTRACTFUNCTIONALITY_H
#define PROTODB_ABSTRACTFUNCTIONALITY_H

#include <memory>

/**
 * The types of functionalities
 */
enum FunctionalityType {

  DATA_STORE,
  LOGGER,
  COMMUNICATOR
};


/**
 * A pointer to the shared
 */
class AbstractFunctionality;
typedef std::shared_ptr<AbstractFunctionality> AbstractFunctionalityPtr;
typedef std::weak_ptr<AbstractFunctionality> AbstractFunctionalityWeakPtr;

class AbstractFunctionality {

public:

  /**
   * Returns the type of the functionality
   * @return the type
   */
  virtual FunctionalityType getType() = 0;

  /**
   * Returns a handle to this functionality
   * @return the handle
   */
  AbstractFunctionalityPtr getHandle();

  /**
   * This method is used to cast the handle to cast the abstract functionality to different functionally based on the
   * template parameter T
   * @tparam T - the type we want to cast the handle into
   * @return a shared pointer handle of the provided type
   */
  template<typename T>
  std::shared_ptr<T> to() {
    return std::dynamic_pointer_cast<T>(getHandle());
  }

protected:

  /**
   * The weak pointer to the node
   */
  AbstractFunctionalityWeakPtr node;

};

#endif //PROTODB_ABSTRACTFUNCTIONALITY_H
