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
