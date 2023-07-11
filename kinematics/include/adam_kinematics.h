#ifndef adam_kinematics_h
#define adam_kinematics_h

#include "stdio.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#include <Eigen/Dense>

#include <cmath>

#include "utility_math.h"
#include "utility_kinematics.h"
#include "utility_log.h"

#include "kinematics_definitions.h"

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

class Kinematics {
    public: Kinematics();

    public: Kinematics(std::string urdf_file_location, bool useStaticCom);

    public: virtual ~Kinematics();

    public: void Initialize(std::string urdf_file_location, bool useStaticCom);

    public: OutputVec CalculateOutputs(GenPosVec q, Foot stance_foot);

    public: Vector3d ComApprox(GenPosVec q);
    public: Vector3d ComPinocchio(GenPosVec q);

    public: bool SolveIK(GenPosVec &q, OutputVec y_out_ref, Foot stance_foot);

    // Foot pos is relative to the hip roll joint
    public: Vector3d HipRollToFootIK(Vector3d foot_pos, Foot foot);

    Vector3d GetCoMPos(GenPosVec q_pos);

    Vector3d GetFramePos(std::string frame_name);

    void UpdateFramePlacements(GenPosVec q);

    // Variables 

    // The pinocchio model
    private: pinocchio::Model model;

    // The pinocchio data
    private: pinocchio::Data data;

    // COM Reference
    private: bool use_static_com = true;

    // Frame IDs
    private: int TORSO_FRAME_ID; 

    private: int LEFT_HIP_YAW_FRAME_ID;
    private: int LEFT_HIP_ROLL_FRAME_ID;
    private: int LEFT_HIP_PITCH_FRAME_ID;
    private: int LEFT_SHIN_FRAME_ID;
    private: int LEFT_FOOT_FRAME_ID; 

    private: int RIGHT_HIP_YAW_FRAME_ID;
    private: int RIGHT_HIP_ROLL_FRAME_ID;
    private: int RIGHT_HIP_PITCH_FRAME_ID;
    private: int RIGHT_SHIN_FRAME_ID;
    private: int RIGHT_FOOT_FRAME_ID; 

    private: int STATIC_COM_FRAME_ID;

    private: Vector3d r_torso_to_hip;

    private: Vector3d r_lhr_to_lhp;
    private: Vector3d r_lhp_to_lkp;
    private: Vector3d r_lkp_to_lf;

    private: Vector3d r_rhr_to_rhp;
    private: Vector3d r_rhp_to_rkp;
    private: Vector3d r_rkp_to_rf;

    // IK solver parameters
    private: double eps = 1e-4;
    private: double damping_factor = 1e-6;
    private: double alpha = 0.2;
    private: double max_number_of_iterations = 300;

    private: double l_t = 0.25;
    private: double l_s = 0.25;

    // Debug function
    public: void PrintOutputs(OutputVec q_out);

    //
    public: int GetNumberOfFrames() {
        return model.frames.size();
    }
};

GenPosVec ConvertGenPosFromMujocoToPinocchio(GenPosVec q_mj);

GenPosVec ConvertGenPosFromPinocchioToMujoco(GenPosVec q_pin);

GenVelVec ConvertGenVelFromMujocoToPinocchio(GenVelVec q_vel_mj);

GenVelVec ConvertGenVelFromPinocchioToMujoco(GenVelVec q_vel_pin);

JointVec ConvertJointVecFromPinocchioToMujoco(JointVec q_joint_pin);

GenPosVec GetControlFramePosState(GenPosVec q_pin);

GenVelVec GetControlFrameVelState(GenPosVec q_pos_world, GenVelVec q_vel_world);

void PrintGenPos(GenPosVec q);

void PrintOutput(OutputVec y);

#endif