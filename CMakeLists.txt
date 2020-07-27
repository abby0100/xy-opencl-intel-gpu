cmake_minimum_required(VERSION 3.10)

set(PROJECT_NAME "xydistance")
project(${PROJECT_NAME})
add_executable(${PROJECT_NAME} main.cpp)

find_package(OpenCL REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCL_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} OpenCL)

#set(OPENCL_VERSION 120)
set(OPENCL_VERSION 220)
target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=${OPENCL_VERSION})

