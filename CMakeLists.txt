set(TARGET "fracopt")

set(SOURCES
  main.cpp
  nonlocal.cpp
  utilities.cpp
  )

set(TARGET_SRC ${SOURCES})

cmake_minimum_required(VERSION 3.3.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fopenmp -march=native")

# Debug build flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Release build flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PLANAR "planar problem" OFF)
IF(PLANAR)
  ADD_DEFINITIONS(-DPLANAR)
ENDIF(PLANAR)

find_package(
  deal.II
  9.4.0
  HINTS
  ${deal.II_DIR}
  ${DEAL_II_DIR}
  ../
  ../../
  $ENV{DEAL_II_DIR})
if(NOT ${deal.II_FOUND})
  message(
    FATAL_ERROR
      "\n"
      "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
      "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
      "or set an environment variable \"DEAL_II_DIR\" that contains this path.")
endif()



include_directories($ENV{IPOPT_DIR}/include/coin-or)
link_directories($ENV{IPOPT_DIR}/lib)
set(IPOPT_INCLUDE_DIRS $ENV{IPOPT_DIR}/include/coin-or)

deal_ii_initialize_cached_variables()
project(${TARGET})
deal_ii_invoke_autopilot()

target_include_directories(${TARGET} PUBLIC $ENV{PAROPT_DIR}/src)

set(PAROPT_LD_CMD -L$ENV{PAROPT_DIR}/lib/ -Wl,-rpath,$ENV{PAROPT_DIR}/lib -lparopt)
target_link_libraries(${TARGET} ${PAROPT_LD_CMD})
