#ifndef PROTODB_ABSTRACTFUNCTIONALITY_H
#define PROTODB_ABSTRACTFUNCTIONALITY_H

#include <memory>

/**
 * The types of functionalities
 */
enum FunctionalityType {

  DATA_STORE,
  LOGGER,
  COMMUNICATOR,
  RESOURCE_MANAGER
};


/**
 * A pointer to the shared
 */
class AbstractFunctionality;
typedef std::shared_ptr<AbstractFunctionality> AbstractFunctionalityPtr;
typedef std::weak_ptr<AbstractFunctionality> AbstractFunctionalityWeakPtr;

/**
 * Weak pointer to the server
 */
class Server;
typedef std::shared_ptr<Server> ServerPtr;
typedef std::weak_ptr<Server> ServerWeakPtr;

class AbstractFunctionality {

public:

  /**
   * Virtual destructor - must be implemented by each functionality
   */
  virtual ~AbstractFunctionality() = default;

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

  /**
   * Assigns this functionality to a server
   */
  void assignToServer(ServerWeakPtr server);

protected:

  /**
   * The weak pointer to the handle
   */
  AbstractFunctionalityWeakPtr handle;

  /**
   * A weak pointer to the server
   */
  ServerWeakPtr serverHandle;

};

#endif //PROTODB_ABSTRACTFUNCTIONALITY_H
