cmake_minimum_required(VERSION 3.9)
project(protoDB)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# check if we support native architecture
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

# find threads
find_package (Threads)

# find jemalloc
find_package(JEMALLOC REQUIRED)

# find the boost package
find_package(Boost COMPONENTS system thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# try to find the mpi package
find_package(MPI REQUIRED)

# if found set the include directories
include_directories(${MPI_INCLUDE_PATH})

### SETTING UP HEADERS AND SOURCE FILES
file(GLOB_RECURSE source_list "*.cc")
file(GLOB_RECURSE all_headers "*.h")

include_directories(${PROJECT_SOURCE_DIR}/functionalities/data_store/include)
include_directories(${PROJECT_SOURCE_DIR}/functionalities/logger/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/btree)
include_directories(${PROJECT_SOURCE_DIR}/core/include)

# define a target for our executable
add_executable(protoDB main.cpp ${source_list})

# link mpi to the protoDB
target_link_libraries(protoDB ${MPI_LIBRARIES})

# link the jmalloc
target_link_libraries(protoDB ${JEMALLOC_LIBRARY})

# link threads
target_link_libraries(protoDB ${CMAKE_THREAD_LIBS_INIT})

# link boost
target_link_libraries(protoDB ${Boost_LIBRARIES})

# set the mpi compile flags for the target
if(MPI_COMPILE_FLAGS)
    set_target_properties(protoDB PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

# set the mpi link flags for the target
if(MPI_LINK_FLAGS)
    set_target_properties(protoDB PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

# this target is used for running a cluster
add_custom_target(run-cluster mpirun -np 2 ${CMAKE_BINARY_DIR}/protoDB)
add_dependencies(run-cluster protoDB)

# add a header only target to enable autocompletion in IDEs
add_custom_target(header_only SOURCES ${all_headers})