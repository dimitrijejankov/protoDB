#ifndef PROTODB_COMMUNICATORFUNCTIONALITY_H
#define PROTODB_COMMUNICATORFUNCTIONALITY_H

#include "AbstractFunctionality.h"

class Communicator : public AbstractFunctionality {

 public:

  /**
   * The communicator constructor
   */
  Communicator();

  /**
   * The destructor this shuts down our communicator
   */
  virtual ~Communicator();

  /**
   * Returns the number of nodes in this cluster
   * @return the number of nodes
   */
  int32_t getNumNodes();


  /**
   * Returns the id of this node
   * @return the node id
   */
  int32_t getNodeID();

  /**
   * Returns the name of this node
   * @return - the name
   */
  std::string getNodeName();

};

#endif //PROTODB_COMMUNICATORFUNCTIONALITY_H
