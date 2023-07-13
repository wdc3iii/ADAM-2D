#ifndef mujoco_interface_h
#define mujoco_interface_h

#include "stdio.h"
#include <map>
#include <iostream>

#include "mujoco.h"
#include "GLFW/glfw3.h"

#include <Eigen/Dense>

#include <cmath>

#include "utility_kinematics.h"
#include "utility_math.h"

#include "mujoco_types.h"

// Mujoco pointers

// Mujoco model pointer
inline mjModel *MJ_MODEL_PTR = NULL;

// Mujoco data pointer
inline mjData *MJ_DATA_PTR = NULL;

// Mujoco contact pointer
inline mjContact *MJ_CONTACT_PTR = NULL;


// Mujoco data structures

// Mujoco camera
inline mjvCamera MJ_CAMERA;

// Mujoco visualization options 
inline mjvOption MJ_OPTIONS;

// Mujoco scene
inline mjvScene MJ_SCENE;

// Mujoco GPU options
inline mjrContext MJ_CONTEXT;


// Mujoco mouse interactions

// Left mouse click
inline bool BUTTON_LEFT = false;

// Middle mouse click
inline bool BUTTON_MIDDLE = false;

// Right mouse click
inline bool BUTTON_RIGHT =  false;

// X position of last mouse click
inline double BUTTON_LAST_X = 0;

// Y position of last mouse click
inline double BUTTON_LAST_Y = 0;


// Mujoco interface class
class MujocoInterface {
    // Generic Mujoco functions and variables

    /// \brief Constructor
    public: MujocoInterface();
    
    /// \brief Destructor
    public: virtual ~MujocoInterface();
    
    /// \brief Pointer to the Mujoco graphics window
    public: GLFWwindow* window;

    /// \brief The MujocoSetup function initializes the simulator
    /// by loading the model and setting up the data interface, 
    /// physics, scene, graphics, camera, and user interaction. 
    public: void MujocoSetup(const char file_name[]);

    /// \brief The MujocoShutdown function shuts down the simulator
    /// and releases all the resources allocated by it. 
    public: void MujocoShutdown();

    /// \brief The UpdateScene function updates the visuals of
    /// of the mujoco simulator
    public: void UpdateScene();


    // Robot specific functions

    /// \brief The GetBasePositions function gets the latest base pose
    /// of the robot in the world frame from the simulator
    /// \return Returns a vector containing the base pose of the robot
    /// in the world frame where the orientation is given in ZYX Euler coordinates
    /// (x_w, y_w, z_w, roll, pitch, yaw)
    public: Eigen::Vector<double, 3> GetBasePositions();

    /// \brief The GetJointPositions function gets the latest joint positions (rad)
    /// of the robot's joints
    /// \return Returns a vector containing the joint positions of all the robot joints
    /// (left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee_pitch,
    ///  right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee_pitch,
    ///  left_shoulder_yaw, left_shoulder_pitch, left_elbow_pitch,
    ///  right_shoulder_yaw, right_shoulder_pitch, right_elbow_pitch)
    public: Eigen::Vector<double, 4> GetJointPositions();

    /// \brief The GetBaseVelocities function gets the latest base twist
    /// of the robot relative to the world frame from the simulator
    /// \return Returns a vector containing the base twist of the robot
    /// in the world frame where the oriention rate is given in ZYX Euler angle rates
    /// (x_w, y_w, z_w, roll_rate, pitch_rate, yaw_rate)
    public: Eigen::Vector<double, 3> GetBaseVelocities();

    /// \brief The GetJointVelocities function gets the latest joint velocities (rad/s)
    /// of the robot's joints
    /// \return Returns a vector containing the joint velocities of all the robot joints
    /// (left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee_pitch,
    ///  right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee_pitch,
    ///  left_shoulder_yaw, left_shoulder_pitch, left_elbow_pitch,
    ///  right_shoulder_yaw, right_shoulder_pitch, right_elbow_pitch)
    public: Eigen::Vector<double, 4> GetJointVelocities();

    /// \brief The SetState function initializes all of the position and velocity states
    /// of the robot in the simulation environment to the ones specified by the inputs
    /// \param[in] q_base The desired base pose of the robot 
    /// \param[in] q_d_base The desired base twist of the robot
    /// \param[in] q_joint The desired joint position of the robot
    /// \param[in] q_d_joint The desired joint velocities of the robot
    public: void SetState(Eigen::Vector<double, 3> q_base, Eigen::Vector<double, 3> q_d_base, Eigen::Vector<double, 4> q_joint, Eigen::Vector<double, 4> q_d_joint);

    public: void SetState(Eigen::Vector<double, 7> q_pos, Eigen::Vector<double, 7> q_vel);

    /// \brief Send a joint position reference to the robots actuators
    /// \param[in] joint_pos_ref The desired joint position
    public: void JointPosCmd(Eigen::Vector<double, 4> joint_pos_ref);

    /// \brief Send a joint velocity reference to the robots actuators
    /// \param[in] joint_vel_ref The desired joint velocity
    public: void JointVelCmd(Eigen::Vector<double, 4> joint_vel_ref);

    /// \brief Send a joint torque reference to the robots actuators
    /// \param[in] joint_torque_ref The desired joint torque
    public: void JointTorCmd(Eigen::Vector<double, 4> joint_torque_ref);

    public: Eigen::Vector<double, 7> GetGeneralizedPos();

    public: Eigen::Vector<double, 7> GetGeneralizedVel();

    public: void PrintContactForce();

    public: void PropagateDynamics();

    /// \brief A map that maps geom indices to geom names
    public: std::map<int, std::string> geom_map;

    /// \brief A map that maps contact pair names to structs containing contact and force/torque states
    public: std::map<std::string, ContactData> contact_pair_map;

    /// \brief The GetContactData checks which of the contact pairs that are active and not,
    /// and calculates the forces and torques between the geoms
    /// \return A map containing the contact data for all the contact pairs
    public: std::map<std::string, ContactData> GetContactData();

    public: Eigen::Vector<double, 2> GetComPos();
    public: Eigen::Vector<double, 2> GetSWFootPos();
};

// Simulation interaction callback functions (DO NOT CHANGE)

/// \brief The keyboard function is the callback function for keyboard presses
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);

/// \brief The mouse_button function is the callback function for mouse button presses
void mouse_button(GLFWwindow* window, int button, int act, int mods);

/// \brief The mouse_move function is the callback function for mouse movements
void mouse_move(GLFWwindow* window, double xpos, double ypos);

/// \brief The scroll function is the callback function for mouse scrolling presses
void scroll(GLFWwindow* window, double xoffset, double yoffset);

#endif