# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wcompton/Repos/ADAM-2D/kinematics

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wcompton/Repos/ADAM-2D/kinematics/build

# Include any dependencies generated for this target.
include CMakeFiles/adam_kinematics.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/adam_kinematics.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/adam_kinematics.dir/flags.make

CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.o: CMakeFiles/adam_kinematics.dir/flags.make
CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.o: ../src/adam_kinematics.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wcompton/Repos/ADAM-2D/kinematics/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.o -c /home/wcompton/Repos/ADAM-2D/kinematics/src/adam_kinematics.cpp

CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wcompton/Repos/ADAM-2D/kinematics/src/adam_kinematics.cpp > CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.i

CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wcompton/Repos/ADAM-2D/kinematics/src/adam_kinematics.cpp -o CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.s

# Object files for target adam_kinematics
adam_kinematics_OBJECTS = \
"CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.o"

# External object files for target adam_kinematics
adam_kinematics_EXTERNAL_OBJECTS =

libadam_kinematics.so: CMakeFiles/adam_kinematics.dir/src/adam_kinematics.cpp.o
libadam_kinematics.so: CMakeFiles/adam_kinematics.dir/build.make
libadam_kinematics.so: /opt/openrobots/lib/libpinocchio.so
libadam_kinematics.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
libadam_kinematics.so: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
libadam_kinematics.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
libadam_kinematics.so: /home/wcompton/Repos/ADAM-2D/utils/build/librobot_utils.so
libadam_kinematics.so: CMakeFiles/adam_kinematics.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wcompton/Repos/ADAM-2D/kinematics/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libadam_kinematics.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/adam_kinematics.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/adam_kinematics.dir/build: libadam_kinematics.so

.PHONY : CMakeFiles/adam_kinematics.dir/build

CMakeFiles/adam_kinematics.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/adam_kinematics.dir/cmake_clean.cmake
.PHONY : CMakeFiles/adam_kinematics.dir/clean

CMakeFiles/adam_kinematics.dir/depend:
	cd /home/wcompton/Repos/ADAM-2D/kinematics/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wcompton/Repos/ADAM-2D/kinematics /home/wcompton/Repos/ADAM-2D/kinematics /home/wcompton/Repos/ADAM-2D/kinematics/build /home/wcompton/Repos/ADAM-2D/kinematics/build /home/wcompton/Repos/ADAM-2D/kinematics/build/CMakeFiles/adam_kinematics.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/adam_kinematics.dir/depend

