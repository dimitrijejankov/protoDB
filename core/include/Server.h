#ifndef PROTODB_ABSTRACTSERVER_H
#define PROTODB_ABSTRACTSERVER_H

#include <AbstractFunctionality.h>
#include <map>



class Server {

public:
  /**
   * Adds a new functionality to the server
   * @param fun - the functionality to add
   */
  void addFunctionality(AbstractFunctionalityPtr fun);

  /**
   * Returns a particular functionality associated of the requested type
   * @param type - the type of the functionality
   * @return the functionality
   */
  AbstractFunctionalityPtr getFunctionality(FunctionalityType type);

  /**
   * Returns a handle to this server
   * @return the handle
   */
  ServerPtr getHandle();

  /**
   * Runs the server
   */
  void run();

private:

  /**
   * The functionalities this server has
   */
  std::map<FunctionalityType, AbstractFunctionalityPtr> functionalities;

  /**
   * A handle to this server when the object is instantiated with new this has to be called to grab the handle
   */
  ServerWeakPtr handle;
};

#endif //PROTODB_ABSTRACTSERVER_H
