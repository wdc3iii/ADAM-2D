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
    std::string urdf_file_location = "../../rsc/models/adam.urdf";

    // The number of configuration tests to perform
    const int number_of_tests = 10000;

    // Set the randomizer seed to get repeatable behavior
    std::srand(0);

    // Set the print precision
    std::cout << std::fixed << std::setprecision(5);

    // Create an instance of the kinematics class
    Kinematics kinematics;

    // Initialize the kinemtics class
    kinematics.Initialize(urdf_file_location, true);
    
    // Create the vector to use for the initial guess in the IK solver
    GenPosVec q_init = GenPosVec::Zero();
    q_init(GenPosID::Q_W) = 1.0;

    // The vector to store the state solution
    GenPosVec q_sol = GenPosVec::Zero();

    // The vector to store the IK solution
    GenPosVec q_ik = GenPosVec::Zero();
    
    // The correct outputs 
    OutputVec y_sol = OutputVec::Zero();

    // The outputs given by the IK solver
    OutputVec y_ik = OutputVec::Zero();

    // Define a vector to initialize the torso euler angle orientation
    Vector3d euler_zyx = Vector3d::Zero();

    // Define a vector to initialize the torso orientation
    Vector4d quat_xyzw = Vector4d::Zero();

    // Define an error vector
    GenPosVec q_error = GenPosVec::Zero(); 

    bool solution_found = true;

    OutputVec y_error = OutputVec::Zero();

    Foot stance_foot = Foot::LEFT;
    
    for (int i = 0; i < number_of_tests; i++) {
        // Print the current iteration
        std::cout << i << ":\t" << std::endl;


        // Generate a solution
        q_sol = GenPosVec::Zero();

        // Fix position states
        q_sol(GenPosID::P_X) = 0;
        q_sol(GenPosID::P_Z) = 0;

        // Orientation
        double pitch = (float) rand() / RAND_MAX * 0.4 - 0.2;
        Vector3d eulerZYX;
        eulerZYX << 0, pitch, 0;
        Vector4d quatXYZW = EulerZYXToQuatXYZW(eulerZYX);
        q_sol.block<4, 1>(GenPosID::Q_X, 0) = quatXYZW;

        // Randomize the hip pitch angles
        q_sol(GenPosID::P_LHP) = (float) rand() / RAND_MAX * (M_PI / 2.0) - M_PI / 4.0;
        q_sol(GenPosID::P_RHP) = (float) rand() / RAND_MAX * (M_PI / 2.0) - M_PI / 4.0;

        // Randomize the knee pitch angles
        q_sol(GenPosID::P_LKP) = (float) rand() / RAND_MAX * (M_PI / 2);
        q_sol(GenPosID::P_RKP) = (float) rand() / RAND_MAX * (M_PI / 2);

        // Calculate the outputs from the position state
        y_sol = kinematics.CalculateOutputs(q_sol, stance_foot);

        std::cout << "y_sol: " << y_sol.transpose() << std::endl;
        
        // Solve the IK problem

        // Set the IK initial guess
        q_ik = q_init;
        double d = 0.08;
        // Set a better IK guess
        
        q_ik(GenPosID::P_LHP) = q_sol(GenPosID::P_LHP) + (float) rand() / RAND_MAX * 2.0 * d - d;
        q_ik(GenPosID::P_LKP) = q_sol(GenPosID::P_LKP) + (float) rand() / RAND_MAX * 2.0 * d - d;

        q_ik(GenPosID::P_RHP) = q_sol(GenPosID::P_RHP) + (float) rand() / RAND_MAX * 2.0 * d - d;
        q_ik(GenPosID::P_RKP) = q_sol(GenPosID::P_RKP) + (float) rand() / RAND_MAX * 2.0 * d - d;
        
        if (i == 6727) {
            std::cout << "q_ik_init: " << q_ik.transpose() << std::endl;
        }

        // Try to solve the IK using the initial guess
        //q_ik = q_sol;
        solution_found = kinematics.SolveIK(q_ik, y_sol, stance_foot);
        
        // Calculate the outputs based on the IK solution
        y_ik = kinematics.CalculateOutputs(q_ik, stance_foot);

        // Compare the actual solution to the IK solution

        // Print the true solution and the IK solution
        std::cout << "q_sol: " << q_sol.transpose() << std::endl;
        std::cout << "q_ik:  " << q_ik.transpose() << std::endl;

        std::cout << "y_sol: " << y_sol.transpose() << std::endl;
        std::cout << "y_ik:  " << y_ik.transpose() << std::endl;

        std::cout << std::endl;

        // Calculate the error in all states except global position
        q_error.block<5, 1>(GenPosID::Q_X, 0) = q_sol.block<5, 1>(GenPosID::Q_X, 0) - q_ik.block<5, 1>(GenPosID::Q_X, 0);

        // Check if the solution is correct
        if ((solution_found == false) || ((y_sol - y_ik).norm() > 0.001)) {
            std::cout << "Error in solution" << std::endl;
            while (true);
        }
    }

    return 0;
}