# CMakeList.txt

cmake_minimum_required (VERSION 3.8)
project(VECTPR_ADD CUDA CXX)
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61")
find_package(CUDA  REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${CMAKE_SOURCE_DIR}/lib/)

set(SRC_FILES
	"vectorMult.cu" 
)

add_executable(emult_perf ${SRC_FILES})

# We need C++ 11 features in NVCC so we can use stoi
set_property(TARGET emult_perf PROPERTY CUDA_STANDARD 11)

