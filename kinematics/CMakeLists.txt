cmake_minimum_required(VERSION 3.10)

# Project name
project(adam_kinematics)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -pthread")

# Find the eigen package
find_package(Eigen3 REQUIRED)

# Find the pinocchio package
find_package(pinocchio REQUIRED HINTS $ENV{PINOCCHIO_DIR}/lib)

# Add the robot utilitity library
add_library(robot_utils SHARED IMPORTED GLOBAL)
set_property(TARGET robot_utils PROPERTY IMPORTED_LOCATION /home/wcompton/Repos/ADAM-2D/utils/build/librobot_utils.so)
target_include_directories(robot_utils INTERFACE /home/wcompton/Repos/ADAM-2D/utils/include)

# Include pinocchio
include_directories($ENV{PINOCCHIO_DIR}/include)

# Include the header files
include_directories(include)

# Create the kinematics library
add_library(adam_kinematics SHARED
    src/adam_kinematics.cpp
)

# Create a kinematics example executable
add_executable(kinematics_example
    test/main.cpp
)

add_executable(test_fk
    test/test_fk.cpp
)

# Link the libraries to the adam_kinematics library
target_link_libraries(adam_kinematics PUBLIC Eigen3::Eigen)
target_link_libraries(adam_kinematics PUBLIC ${PINOCCHIO_LIBRARIES})
target_link_libraries(adam_kinematics PUBLIC robot_utils)
target_link_directories(adam_kinematics PUBLIC ${PINOCCHIO_DIRS})

# Link the libraries to the main example
target_link_libraries(kinematics_example PUBLIC Eigen3::Eigen)
target_link_libraries(kinematics_example PUBLIC ${PINOCCHIO_LIBRARIES})
target_link_libraries(kinematics_example PUBLIC adam_kinematics)
target_link_libraries(kinematics_example PUBLIC robot_utils)
target_link_directories(kinematics_example PUBLIC ${PINOCCHIO_DIRS})

# Link the libraries to the main example
target_link_libraries(test_fk PUBLIC Eigen3::Eigen)
target_link_libraries(test_fk PUBLIC ${PINOCCHIO_LIBRARIES})
target_link_libraries(test_fk PUBLIC adam_kinematics)
target_link_libraries(test_fk PUBLIC robot_utils)
target_link_directories(test_fk PUBLIC ${PINOCCHIO_DIRS})
