#include "stdio.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#include <Eigen/Dense>

#include <cmath>

#include<cstdlib>

#include "utility_math.h"
#include "utility_kinematics.h"
#include "utility_log.h"

#include "../include/kinematics_definitions.h"
#include "adam_kinematics.h"

// Load pinocchio stuff
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/multibody/data.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/multibody/joint/joints.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/math/rpy.hpp"



int main() {
    // Specify the URDF path 
    std::string urdf_file_location = "/home/wcompton/Repos/ADAM-2D/rsc/models/adam.urdf";

    // Set the print precision
    std::cout << std::fixed << std::setprecision(5);

    // Create an instance of the kinematics class
    Kinematics kinematics;

    // Initialize the kinemtics class
    kinematics.Initialize(urdf_file_location, true);

    // Create the vector to use for the initial guess in the IK solver
    GenPosVec q = GenPosVec::Zero();

    // Update the frame positions
    kinematics.UpdateFramePlacements(q);

    // Get the left foot position
    Vector3d left_foot_pos = kinematics.GetFramePos("left_foot");

    // Calculate the center of mass world pos
    Vector3d com_pos_world = kinematics.GetCoMPos(q);
    
    Vector3d com_pos_rel_foot_pos = com_pos_world - left_foot_pos;

    std::cout << "CoM:      " << com_pos_world.transpose() << std::endl;
    std::cout << "Foot:     " << left_foot_pos.transpose() << std::endl;
    std::cout << "Foot Rel: " << com_pos_rel_foot_pos.transpose() << std::endl;
    std::cout << std::endl;

    return 0;
}