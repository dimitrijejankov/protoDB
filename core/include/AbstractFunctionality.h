#ifndef PROTODB_ABSTRACTFUNCTIONALITY_H
#define PROTODB_ABSTRACTFUNCTIONALITY_H

#include <memory>

/**
 * The types of functionalities
 */
enum FunctionalityType {

  DATA_STORE,
  LOGGER
};


/**
 * A pointer to the shared
 */
class AbstractFunctionality;
typedef std::shared_ptr<AbstractFunctionality> AbstractFunctionalityPtr;

class AbstractFunctionality {


};

#endif //PROTODB_ABSTRACTFUNCTIONALITY_H
