#ifndef PROTODB_COMMUNICATORFUNCTIONALITY_H
#define PROTODB_COMMUNICATORFUNCTIONALITY_H

#include "AbstractFunctionality.h"

class Communicator;
typedef std::shared_ptr<Communicator> CommunicatorPtr;

class Communicator : public AbstractFunctionality {

 public:

  /**
   * The communicator constructor
   */
  Communicator();

  /**
   * The destructor this shuts down our communicator
   */
  ~Communicator() override;

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

  /**
   * Returns true if this communicator belongs to the master
   * @return
   */
  bool isMaster();

  /**
   * Returns the id of the master node
   * @return node id
   */
  int32_t masterID();

  /**
   * Returns the type of the communicator functionality
   * @return the type
   */
  FunctionalityType getType() override;

};

#endif //PROTODB_COMMUNICATORFUNCTIONALITY_H
