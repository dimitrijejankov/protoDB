//
// Created by dimitrije on 5/26/18.
//

#include <sys/sysinfo.h>
#include <thread>

#include "ResourceManager.h"

ResourceManager::ResourceManager() {

  // grab the system info
  struct sysinfo sys_info{};
  sysinfo(&sys_info);

  // grab the number of cores
  numCores = std::thread::hardware_concurrency();;

  // grab the total memory on this machine
  totalMemory = sys_info.totalram;

  // core to partition ratio is 1
  coreToPartitionRatio = 1.0;
}

FunctionalityType ResourceManager::getType() {
  return RESOURCE_MANAGER;
}

size_t ResourceManager::getNumCores() const {
  return numCores;
}

size_t ResourceManager::getTotalMemory() const {
  return totalMemory;
}

double ResourceManager::getCoreToPartitionRatio() const {
  return coreToPartitionRatio;
}
