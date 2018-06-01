#ifndef PROTODB_COMMUNICATORFUNCTIONALITY_H
#define PROTODB_COMMUNICATORFUNCTIONALITY_H

#include <vector>
#include <mpi.h>
#include <cassert>
#include <iostream>
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
   * This sends a single value to a particular node
   * @tparam T - the type of the value
   * @param value - the value we are sending
   * @param node - the node we are sending it to
   * @param tag - the tag of the message
   */
  template<typename T>
  void send(T value, int32_t node, int32_t tag){

    // send it
    send(&value, 1, node, tag);
  }

  /**
   * Sends an array of type T to a particular node
   * @param values - a pointer to the first array of the array
   * @param n - the size of the array
   * @param node - the node we are sending it to
   * @param tag - the tag of the message we are sending
   */
  template<typename T>
  void send(T *values, size_t n, int32_t node, int32_t tag) {

    // send the matrix
    MPI_Send(values, (int32_t) n, getDataType<T>(), node, tag, MPI_COMM_WORLD);
  }

  /**
   * Receives the data send with the @see send method
   * @param values - the vector where we are storing the values
   * @param source - the node which is sending the data
   * @param tag - the tag of the message
   */
  template<typename T>
  void recv(std::vector<T> &values, int32_t source, int32_t tag) {

    // just an empty status and message structure
    MPI_Status status{};
    MPI_Message msg{};

    // get the type of the data
    MPI_Datatype type = this->getDataType<T>();

    // probe the message
    MPI_Mprobe(source, tag, MPI_COMM_WORLD, &msg, &status);

    // grab the count
    int32_t number_amount;
    MPI_Get_count(&status, type, &number_amount);

    // resize the vector it will allocate enough memory if necessary
    values.resize((size_t) number_amount);

    // grab how many we got
    MPI_Mrecv(values.data(), number_amount, type, &msg, MPI_STATUS_IGNORE);
  }

  /**
   * This method does an all gather on a single value into a vector
   * @tparam T - the type we are gathering
   * @param value - the value on this node
   * @param values - the vector where are we storing the values
   */
  template<typename T>
  void allGather(T &value, std::vector<T> &values) {
    MPI_Allgather(&value, 1, getDataType<T>(), values.data(), 1, getDataType<T>(), MPI_COMM_WORLD);
  }

  /**
   * This method does an all gather on a single value into a vector
   * @tparam T - the type we are gathering
   * @param value - the value on this node
   * @param values - the vector where are we storing the values
   */
  template<typename T>
  void allGather(std::vector<T> &values, std::vector<T> &outputValues, std::vector<int32_t> &counts) {

    // form a displacement vector
    std::vector<int32_t> displacement(counts.size());
    for(auto i = 0; i < counts.size(); ++i) {
      for(int j = i + 1; j < counts.size(); ++j) {
        displacement[j] += counts[i];
      }
    }

    // do an all gather
    MPI_Allgatherv(values.data(), (int32_t)values.size(), getDataType<T>(),
                   outputValues.data(), counts.data(), displacement.data(),
                   getDataType<T>(), MPI_COMM_WORLD);
  }

 private:

  /**
   * Returns the MPI data type
   * @tparam T - the cpp type we want to get the mpi type
   * @return returns the type of the mpi data type
   */
  template<typename T>
  MPI_Datatype getDataType(){

    // if the type is integer
    if(typeid(T) == typeid(int) || typeid(T) == typeid(int32_t)) {
      return MPI_INT;
    }
    // if the type is double
    else if(typeid(T) == typeid(double)) {
      return MPI_DOUBLE;
    }
    else if(typeid(T) == typeid(size_t) || typeid(T) == typeid(unsigned long)) {
      return MPI_UNSIGNED_LONG;
    }

    // assert if we are here the type is unsupported
    std::cout << "Type unsupported : " << std::string(typeid(T).name()) << std::endl;
    exit(-1);
  }

};

#endif //PROTODB_COMMUNICATORFUNCTIONALITY_H
