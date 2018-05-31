#include <Communicator.h>
#include <mpi.h>

Communicator::Communicator() {

  // Initialize the MPI environment
  int provided;

  MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
  if (provided < MPI_THREAD_MULTIPLE) {
    printf("ERROR: The MPI library does not have full thread support\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

Communicator::~Communicator() {

  // finalize the MPI environment.
  MPI_Finalize();
}

int32_t Communicator::getNumNodes() {

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // returns the number of nodes
  return world_size;
}

int32_t Communicator::getNodeID() {

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // returns id of this node (rank)
  return world_rank;
}

std::string Communicator::getNodeName() {

  // get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // return the name of the node
  return std::string(processor_name);
}

FunctionalityType Communicator::getType() {
  return COMMUNICATOR;
}

bool Communicator::isMaster() {
  // the node id of the master is 0
  return getNodeID() == 0;
}

int32_t Communicator::masterID() {
  return 0;
}

void Communicator::send(double *values, size_t n, int32_t node, int32_t tag) {

  // send the matrix
  MPI_Send(values, (int32_t) n, MPI_DOUBLE, node, tag, MPI_COMM_WORLD);
}

void Communicator::send(unsigned long *values, size_t n, int32_t node, int32_t tag) {

  // send the matrix
  MPI_Send(values, (int32_t) n, MPI_UNSIGNED_LONG, node, tag, MPI_COMM_WORLD);
}

template<typename T>
void Communicator::send(T value, int32_t node, int32_t tag) {

  // send it
  send(&value, 1, node, tag);
}

void Communicator::recv(std::vector<double> &values, int32_t source, int32_t tag) {

  // just an empty status and message structure
  MPI_Status status{};
  MPI_Message msg{};

  // probe the message
  MPI_Mprobe(source, tag, MPI_COMM_WORLD, &msg, &status);

  // grab the count
  int32_t number_amount;
  MPI_Get_count(&status, MPI_DOUBLE, &number_amount);

  // resize the vector it will allocate enough memory if necessary
  values.resize((size_t) number_amount);

  // grab how many we got
  MPI_Mrecv(values.data(), number_amount, MPI_DOUBLE, &msg, MPI_STATUS_IGNORE);
}

void Communicator::recv(std::vector<unsigned long> &values, int32_t source, int32_t tag) {

  // just an empty status and message structure
  MPI_Status status{};
  MPI_Message msg{};

  // probe the message
  MPI_Mprobe(source, tag, MPI_COMM_WORLD, &msg, &status);

  // grab the count
  int32_t number_amount;
  MPI_Get_count(&status, MPI_UNSIGNED_LONG, &number_amount);

  // resize the vector it will allocate enough memory if necessary
  values.resize((size_t) number_amount);

  // grab how many we got
  MPI_Mrecv(values.data(), number_amount, MPI_UNSIGNED_LONG, &msg, MPI_STATUS_IGNORE);
}
