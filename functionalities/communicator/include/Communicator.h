#ifndef PROTODB_COMMUNICATORFUNCTIONALITY_H
#define PROTODB_COMMUNICATORFUNCTIONALITY_H

#include <vector>
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

  /**
   * Sends a double array to a particular node
   * @param values - a pointer to the first array of the array
   * @param n - the size of the array
   * @param node - the node we are sending it to
   * @param tag - the tag of the message we are sending
   */
  void send(double *values, size_t n, int32_t node, int32_t tag);

  /**
   * Sends an unsigned long array to a particular node
   * @param values - a pointer to the first array of the array
   * @param n - the size of the array
   * @param node - the node we are sending it to
   * @param tag - the tag of the message we are sending
   */
  void send(unsigned long *values, size_t n, int32_t node, int32_t tag);

  /**
   * This sends a single value
   * @tparam T
   * @param value
   * @param node
   * @param tag
   */
  template<typename T>
  void send(T value, int32_t node, int32_t tag);

  /**
   *
   * @param values
   * @param maxN
   * @param source
   * @return
   */
  void recv(std::vector<double> &values, int32_t source, int32_t tag);

  /**
   *
   * @param values
   * @param maxN
   * @param node
   * @return
   */
  void recv(std::vector<unsigned long> &values, int32_t source, int32_t tag);

};

#endif //PROTODB_COMMUNICATORFUNCTIONALITY_H
