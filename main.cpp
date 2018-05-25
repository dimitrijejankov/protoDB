#include <mpi.h>
#include <stdio.h>
#include <Communicator.h>

int main() {


  Communicator communicator;

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n", communicator.getNodeName().c_str(),
         communicator.getNodeID(), communicator.getNumNodes());
}