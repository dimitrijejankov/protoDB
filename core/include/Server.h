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
   * Runs the server
   */
  void run();

private:

  /**
   * The functionalities this server has
   */
  std::map<FunctionalityType, AbstractFunctionalityPtr> functionalities;
};

#endif //PROTODB_ABSTRACTSERVER_H
