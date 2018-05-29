//
// Created by dimitrije on 5/26/18.
//

#ifndef PROTODB_RESOURCEMANAGER_H
#define PROTODB_RESOURCEMANAGER_H

#include <cstdio>
#include <AbstractFunctionality.h>

/**
 * The abstract pointer
 */
class ResourceManager;
typedef std::shared_ptr<ResourceManager> ResourceManagerPtr;

class ResourceManager : public AbstractFunctionality {

public:

  /**
   * Initializes the resource manager for this machine
   */
  ResourceManager();

  /**
   * Returns the type of the functionality
   * @return the type
   */
  FunctionalityType getType() override;

  /**
   * Returns the number of cores on this machine
   * @return the number of cores
   */
  size_t getNumCores() const;

  /**
   * Returns the total memory on this machine
   * @return the memory
   */
  size_t getTotalMemory() const;

  /**
   * Returns the ratio of partitons to the number of cores
   * @return the ratio
   */
  double getCoreToPartitionRatio() const;

 protected:

  /**
   * The number of cores available on this machine
   */
  size_t numCores;

  /**
   * Total memory on this machine
   */
  size_t totalMemory;

  /**
   * The ratio of partitions to the number of cores
   */
  double coreToPartitionRatio;

};

#endif //PROTODB_RESOURCEMANAGER_H
