#include "stdio.h"
#include <iostream>
#include <iomanip>

#include <Eigen/Dense>

#include <cmath>

#include "mujoco_interface.h"

#include "utility_kinematics.h"
#include "utility_math.h"
#include "utility_log.h"

int main() {
    // Specify the freqency you want the mujoco visuals to update at (should be lower than the simulation frequency)
    const double mujcoco_visual_update_rate = 60.0;

    // Specify the file name of the mdoel you want to simulate
    const char model_file_location[] = "../../rsc/models/adam.xml";

    // Specify the file location of the logger file
    const char logger_file_location[] = "../../plot/log.csv";

    // Create an instance of the class used to run the mujoco simulation
    MujocoInterface mujoco_interface;

    // Initialize the mujoco simulation
    mujoco_interface.MujocoSetup(model_file_location);

    // Define the initial pose
    Eigen::Vector<double, 3> q_base_initial;
    q_base_initial << 0.0, 0.0, 0.0;
    
    // Define the initial joint configuration
    Eigen::Vector<double, 4> q_joint_initial;
    q_joint_initial << 0, 0, 0, 0;

    // Define the initial velocity
    Eigen::Vector<double, 3> q_d_base_initial;
    q_d_base_initial << 0.0, 0.0, 0.0;

    // Define the initial joint velocities
    Eigen::Vector<double, 4> q_d_joint_initial;
    q_d_joint_initial << 0, 0, 0, 0;

    // Set the initial state
    mujoco_interface.SetState(q_base_initial, q_d_base_initial, q_joint_initial, q_d_joint_initial);

    // Base states
    Eigen::Vector<double, 3> q_base;
    Eigen::Vector<double, 3> q_d_base;
    
    // Joint states
    Eigen::Vector<double, 4> q_joint;
    Eigen::Vector<double, 4> q_d_joint;

    // Control references
    Eigen::Vector<double, 4> q_joint_ref = q_joint_initial;
    Eigen::Vector<double, 4> q_d_joint_ref = q_d_joint_initial;
    Eigen::Vector<double, 4> ff_joint_torque = Eigen::Vector<double, 4>::Zero();

    // Control inputs
    Eigen::Vector<double, 4> u = Eigen::Vector<double, 4>::Zero();

    // Simulation time
    double t;
 
    Logger logger(logger_file_location);
    
    // Add data descriptions
    logger.AddLabels("t,"
                     "pos_x, pos_y, pos_z, roll, pitch, yaw,"
                     "x_vel, y_vel, z_vel, roll_rate, pitch_rate, yaw_rate,"
                     "left_hip_yaw_pos, left_hip_roll_pos, left_hip_pitch_pos, left_knee_pitch_pos,"
                     "right_hip_yaw_pos, right_hip_roll_pos, right_hip_pitch_pos, right_knee_pitch_pos,"
                     "left_shoulder_yaw_pos, left_shoulder_pitch_pos, left_elbow_pitch_pos," 
                     "right_shoulder_yaw_pos, right_shoulder_pitch_pos, right_elbow_pitch_pos," 
                     "left_hip_yaw_vel, left_hip_roll_vel, left_hip_pitch_vel, left_knee_pitch_vel,"
                     "right_hip_yaw_vel, right_hip_roll_vel, right_hip_pitch_vel, right_knee_pitch_vel,"
                     "left_shoulder_yaw_vel, left_shoulder_pitch_vel, left_elbow_pitch_vel," 
                     "right_shoulder_yaw_vel, right_shoulder_pitch_vel, right_elbow_pitch_vel,"  
                     "left_hip_yaw_pos_ref, left_hip_roll_pos_ref, left_hip_pitch_pos_ref, left_knee_pitch_pos_ref,"
                     "right_hip_yaw_pos_ref, right_hip_roll_pos_ref, right_hip_pitch_pos_ref, right_knee_pitch_pos_ref,"
                     "left_shoulder_yaw_pos_ref, left_shoulder_pitch_pos_ref, left_elbow_pitch_pos_ref," 
                     "right_shoulder_yaw_pos_ref, right_shoulder_pitch_pos_ref, right_elbow_pitch_pos_ref," 
                     "left_hip_yaw_vel_ref, left_hip_roll_vel_ref, left_hip_pitch_vel_ref, left_knee_pitch_vel_ref,"
                     "right_hip_yaw_vel_ref, right_hip_roll_vel_ref, right_hip_pitch_vel_ref, right_knee_pitch_vel_ref,"
                     "left_shoulder_yaw_vel_ref, left_shoulder_pitch_vel_ref, left_elbow_pitch_vel_ref," 
                     "right_shoulder_yaw_vel_ref, right_shoulder_pitch_vel_ref, right_elbow_pitch_vel_ref,"
                     "left_hip_yaw_tor_ref, left_hip_roll_tor_ref, left_hip_pitch_tor_ref, left_knee_pitch_tor_ref,"
                     "right_hip_yaw_tor_ref, right_hip_roll_tor_ref, right_hip_pitch_tor_ref, right_knee_pitch_tor_ref,"
                     "left_shoulder_yaw_tor_ref, left_shoulder_pitch_tor_ref, left_elbow_pitch_tor_ref," 
                     "right_shoulder_yaw_tor_ref, right_shoulder_pitch_tor_ref, right_elbow_pitch_tor_ref,");

    // Create a vector to store the log data;
    Eigen::VectorXd log_data(1 + q_base.rows() +  q_d_base.rows() + q_joint.rows() + q_d_joint.rows() + q_joint_ref.rows() + q_d_joint.rows() + ff_joint_torque.rows());

    // The simulation loop
    while (!glfwWindowShouldClose(mujoco_interface.window)) {
        // Get the current simulation time
        mjtNum simstart = MJ_DATA_PTR->time;

        // This loop ensures that the mujoco visuals updates at the rate specified by mujcoco_visual_update_rate
        while (MJ_DATA_PTR->time - simstart < 1.0 / mujcoco_visual_update_rate) {      
            // Base states
            q_base = mujoco_interface.GetBasePositions();
            q_d_base = mujoco_interface.GetBaseVelocities();
            
            // Joint states
            q_joint = mujoco_interface.GetJointPositions();
            q_d_joint = mujoco_interface.GetJointVelocities();

            // Get the current simulation time
            t = MJ_DATA_PTR->time;            

            // Add a controller to update the joint references
            //controller.UpdateReferences(q_joint_ref, q_d_joint_ref, ff_joint_torque, q_joint, q_d_joint, q_base, q_d_base, t);

            // Set the joint references
            mujoco_interface.JointPosCmd(q_joint_ref);

            // Set the joint velocities
            mujoco_interface.JointVelCmd(q_d_joint_ref);

            // Comment out to remove fixed pose
            //mujoco_interface.SetState(q_base_initial, q_d_base_initial, q_joint_initial, q_d_joint_initial);

            // Store the simulation data in a vector
            log_data << t, q_base,  q_d_base, q_joint, q_d_joint, q_joint_ref, q_d_joint_ref, ff_joint_torque;

            // Write the simulation data to the logger file
            logger.Write(log_data);

            std::map<std::string, ContactData> contact_pairs = mujoco_interface.GetContactData();

            std::cout << contact_pairs["right_foot_to_plane"] << std::endl;

            //Eigen::Vector<double, 9> contact_frame;
            //for(int k = 0; k < 9; k++)
            //{
            //    contact_frame(k) = MJ_DATA_PTR->contact[0].frame[k];
            //}
            //std::cout << contact_frame.block<3, 1>(0, 0).transpose() << std::endl;
            //std::cout << contact_frame.block<3, 1>(3, 0).transpose() << std::endl;
            //std::cout << contact_frame.block<3, 1>(6, 0).transpose() << std::endl;
            //std::cout << std::endl;
            
            

            //std::cout << MJ_DATA_PTR->contact[0].frame << std::endl;

            // Propagate the dynamics
            mj_step(MJ_MODEL_PTR, MJ_DATA_PTR);       
        }

        // Update the visuals
        mujoco_interface.UpdateScene();
    }

    // Close the mujoco simulation
    mujoco_interface.MujocoShutdown();

    return 0;
}