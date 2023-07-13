#include "stdio.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#include <Eigen/Dense>

#include <cmath>

#include <cstdlib>

#include "utility_math.h"
#include "utility_kinematics.h"
#include "utility_log.h"

#include "../include/kinematics_definitions.h"
#include "adam_kinematics.h"

#include "mujoco_interface.h"

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
    std::string urdf_file_location = "../../rsc/models/adam.urdf";
    const char model_file_location[] = "../../rsc/models/adam.xml";

    // Set the randomizer seed to get repeatable behavior
    std::srand(0);

    // Set the print precision
    std::cout << std::fixed << std::setprecision(5);

    // Create an instance of the kinematics class
    Kinematics kinematics;
    kinematics.Initialize(urdf_file_location, true);

    // Create an instance of the class used to run the mujoco simulation
    MujocoInterface mujoco_interface;
    mujoco_interface.MujocoSetup(model_file_location);


    // Generate an initial configuration
    GenPosVec q_pos = GenPosVec::Zero();

    // Fix position states
    q_pos(GenPosID::P_X) = 0;
    q_pos(GenPosID::P_Z) = 0;

    // Orientation
    double pitch = (float) rand() / RAND_MAX * 0.4 - 0.2;
    Vector3d eulerZYX;
    eulerZYX << 0, pitch, 0;
    Vector4d quatXYZW = EulerZYXToQuatXYZW(eulerZYX);
    q_pos.block<4, 1>(GenPosID::Q_X, 0) = quatXYZW;

    // Randomize the hip pitch angles
    q_pos(GenPosID::P_LHP) = (float) rand() / RAND_MAX * (M_PI / 2.0) - M_PI / 4.0;
    q_pos(GenPosID::P_RHP) = (float) rand() / RAND_MAX * (M_PI / 2.0) - M_PI / 4.0;

    // Randomize the knee pitch angles
    q_pos(GenPosID::P_LKP) = (float) rand() / RAND_MAX * (M_PI / 2);
    q_pos(GenPosID::P_RKP) = (float) rand() / RAND_MAX * (M_PI / 2);


    // Calculate the outputs from the position state, using pinocchio
    y_sol = kinematics.CalculateOutputs(q_pos, stance_foot);
    
    // Compute the same outputs, using MuJoCo
    GenPosVecMJC q_pos_mj = ConvertGenPosFromPinocchioToMujoco(q_pos); 
    mujoco_interface.SetState(q_pos_mj);
    
    return 0;
}