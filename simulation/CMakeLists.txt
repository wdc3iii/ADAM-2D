cmake_minimum_required(VERSION 3.10)

# Project name
project(mujoco_interface)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)  # Need c++ 17 for inline variables
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

# Add necessary packages
find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)

# Add the robot utilitity library
add_library(robot_utils SHARED IMPORTED GLOBAL)
set_property(TARGET robot_utils PROPERTY IMPORTED_LOCATION /home/wcompton/Repos/ADAM-2D/utils/build/librobot_utils.so)
target_include_directories(robot_utils INTERFACE /home/wcompton/Repos/ADAM-2D/utils/include)

# Include mujoco
include_directories($ENV{MUJOCO_DIR}/include)
include_directories($ENV{MUJOCO_DIR}/include/mujoco)
find_library(MUJOCO_LIB mujoco REQUIRED HINTS $ENV{MUJOCO_DIR}/lib)
add_library(mujoco SHARED IMPORTED GLOBAL)
set_property(TARGET mujoco PROPERTY IMPORTED_LOCATION ${CMAKE_INCLUDE_DIR}/lib/libmujoco.so)

# Include the header files
include_directories(include)

# Create the mujoco interface library
add_library(mujoco_interface SHARED
    src/mujoco_interface.cpp
)

# Create the sim executable
add_executable(sim_test
    test/sim_test.cpp
)

# Link the libraries to the mujoco_interface library
target_link_libraries(mujoco_interface PUBLIC Eigen3::Eigen)
target_link_libraries(mujoco_interface PUBLIC glfw)
target_link_libraries(mujoco_interface PUBLIC ${MUJOCO_LIB})
target_link_libraries(mujoco_interface PUBLIC robot_utils)


# Link the libraries to the sim example
target_link_libraries(sim_test PUBLIC glfw)
target_link_libraries(sim_test PUBLIC ${MUJOCO_LIB})
target_link_libraries(sim_test PUBLIC Eigen3::Eigen)
target_link_libraries(sim_test PUBLIC robot_utils)
target_link_libraries(sim_test PUBLIC mujoco_interface)