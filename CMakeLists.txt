cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "xydistance")
project(${PROJECT_NAME})

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

option (DEBUG "Print Debug Information" ON)
if(DEBUG)
    add_definitions(-DDEBUG)
    message(STATUS "Open DEBUG")
else()
	message(STATUS "Close DEBUG")
endif()

add_definitions(-std=c++11)
#add_compile_options(-mavx2)
add_definitions(-mpopcnt -mavx2)
#add_definitions(-mavx512f -mavx512dq)
#add_definitions(-mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mpopcnt -mavx -mfma -mavx2 -mfpmath=both)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIR})
#target_link_libraries(${PROJECT_NAME} OpenCL)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall ${OpenMP_CXX_FLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL ${OpenMP_CXX_FLAGS})

#set(OPENCL_VERSION 120)
set(OPENCL_VERSION 220)
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=${OPENCL_VERSION})

