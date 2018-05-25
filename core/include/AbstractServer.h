#ifndef PROTODB_ABSTRACTSERVER_H
#define PROTODB_ABSTRACTSERVER_H

#include <AbstractFunctionality.h>
#include <map>



class AbstractServer {

  /**
   * The functionalities this server has
   */
  std::map<FunctionalityType, AbstractFunctionalityPtr> functionalities;

};

#endif //PROTODB_ABSTRACTSERVER_H
