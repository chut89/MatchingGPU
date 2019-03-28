 
CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2)
PROJECT(3dFace)

#
# Setup our own cmake modules
#

message(STATUS "CMAKE_MODULE_PATH " ${CMAKE_MODULE_PATH})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake/)

message(STATUS "CMAKE_MODULE_PATH " ${CMAKE_MODULE_PATH})

find_package(Threads REQUIRED)
# boost
find_package( Boost 1.40 COMPONENTS program_options REQUIRED )
include_directories( ${Boost_INCLUDE_DIR} )

#Important packages
FIND_PACKAGE(OpenCV REQUIRED)


# Setup flycapture
FIND_PACKAGE(Fly REQUIRED)
# don't use SYSTEM, it make trouble on cross compile
INCLUDE_DIRECTORIES(${FLY_INCLUDE_DIR})

FILE(GLOB SOURCES "src/*.cu")
FILE(GLOB HEADER_FILES "include/*.h")
  
find_package(CUDA REQUIRED)
# architecture and code depend on particular gpu
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O5 -gencode arch=compute_50,code=sm_50)


INCLUDE_DIRECTORIES(include)

CUDA_ADD_EXECUTABLE(matching ${SOURCES} ${HEADER_FILES})

TARGET_LINK_LIBRARIES(matching ${OpenCV_LIBS} -lflycapture)
#message(STATUS "OpenCV Libs Hello" ${OpenCV_LIBS})


