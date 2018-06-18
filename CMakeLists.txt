cmake_minimum_required(VERSION 3.5)
project(protoDB)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# check if we support native architecture
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
if (COMPILER_OPT_ARCH_NATIVE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
endif()

# find threads
find_package (Threads)

# find libatomic
find_package(GccAtomic)

# find open mp
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# find gsl
find_package(GSL REQUIRED)

# find the boost package
find_package(Boost COMPONENTS system thread REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

# try to find the mpi package
find_package(MPI REQUIRED)

# if found set the include directories
include_directories(${MPI_INCLUDE_PATH})

### SETTING UP HEADERS AND SOURCE FILES
file(GLOB_RECURSE source_list "core/*.cc" "functionalities/*.cc" "runtime/*.cc")
file(GLOB_RECURSE all_headers "core/*.h" "functionalities/*.h" "runtime/*.h" third-party/*.hpp third-party/*.h)

include_directories(${PROJECT_SOURCE_DIR}/functionalities/data_store/include)
include_directories(${PROJECT_SOURCE_DIR}/functionalities/logger/include)
include_directories(${PROJECT_SOURCE_DIR}/functionalities/communicator/include)
include_directories(${PROJECT_SOURCE_DIR}/functionalities/resource_manager/include)
include_directories(${PROJECT_SOURCE_DIR}/runtime/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/lockfree)
include_directories(${PROJECT_SOURCE_DIR}/third-party/bwtree/include)
include_directories(${PROJECT_SOURCE_DIR}/third-party/queue)
include_directories(${PROJECT_SOURCE_DIR}/core/include)

# add the third party libraries
add_library(bwtree STATIC third-party/bwtree/source/bwtree.cc)
target_link_libraries(bwtree ${GCCLIBATOMIC_LIBRARY})

# adds one pdb integration test to CMAKE
function(add_protodb_app app-name app-file)

    # define a target for our executable
    add_executable(${app-name} ${app-file} ${source_list})

    # link the libs
    target_link_libraries(${app-name} bwtree ${MPI_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${GCCLIBATOMIC_LIBRARY} ${Boost_LIBRARIES} GSL::gsl)

    # set the mpi compile flags for the target
    if(MPI_COMPILE_FLAGS)
        set_target_properties(${app-name} PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    # set the mpi link flags for the target
    if(MPI_LINK_FLAGS)
        set_target_properties(${app-name} PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()

    # this target is used for running a cluster
    add_custom_target(run-cluster-${app-name} mpirun -np 2 ${CMAKE_BINARY_DIR}/${app-name})
    add_dependencies(run-cluster-${app-name} ${app-name})

endfunction(add_protodb_app)

# add the main app
add_protodb_app(protoDB main.cpp)

add_protodb_app(test-matrix-multiply tests/TestMatrixMultiply.cc)

# add the test data store test
add_protodb_app(test-data-store tests/TestDataStore.cc)

# add the test to test the communicator with a traditional relional database workload
add_protodb_app(test-comunicator-traditional tests/TestCommunicatorTraditional.cc)

# add the test to test the communicator with a traditional relional database workload
add_protodb_app(test-comunicator tests/TestCommunicatorNew.cc)

# add a header only target to enable autocompletion in IDEs
add_custom_target(header_only SOURCES ${all_headers})
