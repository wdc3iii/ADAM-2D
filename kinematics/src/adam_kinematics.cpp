#include "../include/adam_kinematics.h"
#include <iostream>

Kinematics::Kinematics(){}

Kinematics::Kinematics(std::string urdf_file_location, bool useStaticCom) {
    this->Initialize(urdf_file_location, useStaticCom);
}

Kinematics::~Kinematics(){}

void Kinematics::Initialize(std::string urdf_file_location, bool useStaticCom) {
    this->model = pinocchio::Model();

    pinocchio::urdf::buildModel(urdf_file_location, pinocchio::JointModelFreeFlyer(), this->model);

    this->data = pinocchio::Data(this->model);
    this->use_static_com = useStaticCom;

    // Get frame ID's
    TORSO_FRAME_ID = model.getFrameId("torso");
    LEFT_HIP_YAW_FRAME_ID = model.getFrameId("left_hip_yaw");
    RIGHT_HIP_YAW_FRAME_ID = model.getFrameId("right_hip_yaw");
    LEFT_FOOT_FRAME_ID = model.getFrameId("left_foot");
    RIGHT_FOOT_FRAME_ID = model.getFrameId("right_foot");
    LEFT_HIP_ROLL_FRAME_ID = model.getFrameId("left_hip_roll");
    RIGHT_HIP_ROLL_FRAME_ID = model.getFrameId("right_hip_roll");
    LEFT_HIP_PITCH_FRAME_ID = model.getFrameId("left_hip_pitch");
    RIGHT_HIP_PITCH_FRAME_ID = model.getFrameId("right_hip_pitch");
    LEFT_SHIN_FRAME_ID = model.getFrameId("left_shin");
    RIGHT_SHIN_FRAME_ID = model.getFrameId("right_shin");
    STATIC_COM_FRAME_ID = model.getFrameId("static_com");

    // Calculate joint offsets
    GenPosVec q_nom = GenPosVec::Zero();

    pinocchio::forwardKinematics(this->model, this->data, q_nom);

    pinocchio::updateFramePlacements(this->model, this->data);

    r_torso_to_hip = (data.oMf[LEFT_HIP_ROLL_FRAME_ID].translation() + data.oMf[RIGHT_HIP_ROLL_FRAME_ID].translation()) / 2.0
                   - data.oMf[TORSO_FRAME_ID].translation();

    r_lhr_to_lhp = data.oMf[LEFT_HIP_PITCH_FRAME_ID].translation() - data.oMf[LEFT_HIP_ROLL_FRAME_ID].translation();
    r_lhp_to_lkp = data.oMf[LEFT_SHIN_FRAME_ID].translation() - data.oMf[LEFT_HIP_PITCH_FRAME_ID].translation();
    r_lkp_to_lf = data.oMf[LEFT_FOOT_FRAME_ID].translation() - data.oMf[LEFT_SHIN_FRAME_ID].translation();

    r_rhr_to_rhp = data.oMf[RIGHT_HIP_PITCH_FRAME_ID].translation() - data.oMf[RIGHT_HIP_ROLL_FRAME_ID].translation();
    r_rhp_to_rkp = data.oMf[RIGHT_SHIN_FRAME_ID].translation() - data.oMf[RIGHT_HIP_PITCH_FRAME_ID].translation();
    r_rkp_to_rf = data.oMf[RIGHT_FOOT_FRAME_ID].translation() - data.oMf[RIGHT_SHIN_FRAME_ID].translation();

    std::cout << "r_torso_to_hip: " << r_torso_to_hip.transpose() << std::endl;

    std::cout << "r_lhr_to_lhp: " << r_lhr_to_lhp.transpose() << std::endl;
    std::cout << "r_lhp_to_lkp: " << r_lhp_to_lkp.transpose() << std::endl;
    std::cout << "r_lkp_to_lf: " << r_lkp_to_lf.transpose() << std::endl;

    std::cout << "r_rhr_to_rhp: " << r_rhr_to_rhp.transpose() << std::endl;
    std::cout << "r_rhp_to_rkp: " << r_rhp_to_rkp.transpose() << std::endl;
    std::cout << "r_rkp_to_rf: " << r_rkp_to_rf.transpose() << std::endl;
}


OutputVec Kinematics::CalculateOutputs(GenPosVec q, Foot stance_foot) {
    // Set indices corresponding to the stance and swing foot 
    int stf_frame_id;
    int swf_frame_id;
    int sthy_frame_id;
    int swhy_frame_id;

    if (stance_foot == Foot::LEFT) {
        stf_frame_id = this->LEFT_FOOT_FRAME_ID;
        swf_frame_id = this->RIGHT_FOOT_FRAME_ID;
    } else if (stance_foot == Foot::RIGHT) {
        stf_frame_id = this->RIGHT_FOOT_FRAME_ID;
        swf_frame_id = this->LEFT_FOOT_FRAME_ID;
    } else {
        std::cout << "ERROR: Kinematics::CalculateOutputs: Invalid stance foot chosen" << std::endl;
        while(true);
    }
    
    // Define variables
    // The CoM position in the local world aligned frame
    Vector3d com_pos_world = Vector3d::Zero();
    // The stance foot position in the world frame
    Vector3d stf_pos_world = Vector3d::Zero();
    // The swing foot position in the world frame
    Vector3d swf_pos_world = Vector3d::Zero();
    // The CoM position relative to the stance foot position
    Vector3d com_pos_body = Vector3d::Zero();
    // The swing foot positon relative to the stance foot position
    Vector3d swf_pos_body = Vector3d::Zero();
    // The orientation of the torso in the world frame given by zyx euler angles
    Vector3d torso_euler_zyx = Vector3d::Zero();
    // The output vector to be returned
    OutputVec y_out = OutputVec::Zero();

    // Compute the outputs
    // Calculate the forward kinematics for the model
    pinocchio::forwardKinematics(this->model, this->data, q);
    // Update the forward kinematics of all the frames
    pinocchio::updateFramePlacements(this->model, this->data);

    // Calculate the CoM positions in the world frame
    if (this->use_static_com) {
        // Use an approximated fixed CoM position
        com_pos_world = data.oMf[STATIC_COM_FRAME_ID].translation();
    } else {
        // Calculate the current CoM position
        com_pos_world = pinocchio::centerOfMass(this->model, this->data, q, true);
    }

    // Get the ZYX euler angle orientation of the torso in the world frame
    torso_euler_zyx = QuatXYZWToEulerZYX(q.block<4, 1>(GenPosID::Q_X, 0));
    
    // Zero initialize the torso orientaiton (error is handled below)
    y_out(OutID::PITCH) = torso_euler_zyx(1);

    // Calculate the stance and swing foot positions in the world frame
    stf_pos_world = data.oMf[stf_frame_id].translation();
    swf_pos_world = data.oMf[swf_frame_id].translation();

    // Calculate frame positions relative to the stance foot
    swf_pos_body = swf_pos_world - stf_pos_world;
    com_pos_body = com_pos_world - stf_pos_world; 

    // Swing foot position relative to the stance foot position
    y_out(OutID::SWF_POS_X) = swf_pos_body(0);
    y_out(OutID::SWF_POS_Z) = swf_pos_body(2);

    // Torso pos relative to stance foot
    y_out(OutID::COM_POS_X) = com_pos_body(0);
    y_out(OutID::COM_POS_Z) = com_pos_body(2);

    return y_out;
}

Vector3d Kinematics::ComPinocchio(GenPosVec q) {
    return pinocchio::centerOfMass(this->model, this->data, q, true);
}

Vector3d Kinematics::ComApprox(GenPosVec q) {
    // Calculate the forward kinematics for the model
    pinocchio::forwardKinematics(this->model, this->data, q);

    // Update the forward kinematics of all the frames
    pinocchio::updateFramePlacements(this->model, this->data);
    return data.oMf[STATIC_COM_FRAME_ID].translation();
}

Vector3d Kinematics::HipRollToFootIK(Vector3d foot_pos, Foot foot) {
    double x = foot_pos(0);
    double y = foot_pos(1);
    double z = foot_pos(2);

    double r_z;
    double r_y;
    double r_x;

    if (foot == Foot::LEFT) {
        r_z = r_lhr_to_lhp(2);
        r_y = r_lhr_to_lhp(1) + r_lhp_to_lkp(1) + r_lkp_to_lf(1);
        r_x = r_lhr_to_lhp(0);
    } else if (foot == Foot::RIGHT) {
        r_z = r_rhr_to_rhp(2);
        r_y = r_rhr_to_rhp(1) + r_rhp_to_rkp(1) + r_rkp_to_rf(1);
        r_x = r_rhr_to_rhp(0);
    } else {
        std::cout << "ERROR in Kinematics::HipRollToFootIK. Invalid foot chosen" << std::endl;
        while(true);
    }

    double x_prime = x - r_x;

    double L_prime = sqrt(y * y + z * z);

    double phi_prime = atan2(y, -z);

    double L_yz = sqrt(L_prime * L_prime - r_y * r_y) + r_z;

    double gamma = atan2(-r_y, -r_z + L_yz);

    double q_hr = phi_prime + gamma;

    double alpha = atan2(-x_prime, L_yz);

    double L = sqrt(x_prime * x_prime + L_yz * L_yz);

    double beta = CalculateCosineRuleAngle(this->l_t, this->l_s, L);

    double eta = CalculateSineRuleAngle(L, beta, this->l_s);

    double q_hp = alpha - eta;

    double q_kp = M_PI - beta;

    Vector3d q(q_hr, q_hp, q_kp);

    return q;
}

bool Kinematics::SolveIK(GenPosVec &q, OutputVec y_out_ref, Foot stance_foot) {
    std::cout << "q0: " << q.transpose() << std::endl;
    // Set indices corresponding to the stance and swing foot 

    int stf_frame_id;
    int swf_frame_id;
    int sthy_frame_id;
    int swhy_frame_id;

    int stsy_gen_pos_id;
    int swsy_gen_pos_id;

    if (stance_foot == Foot::LEFT) {
        stf_frame_id = this->LEFT_FOOT_FRAME_ID;
        swf_frame_id = this->RIGHT_FOOT_FRAME_ID;
    } else if (stance_foot == Foot::RIGHT) {
        stf_frame_id = this->RIGHT_FOOT_FRAME_ID;
        swf_frame_id = this->LEFT_FOOT_FRAME_ID;
    } else {
        std::cout << "ERROR: Kinematics::SolveIK: Invalid stance foot chosen" << std::endl;
        while (true);
    }

    // Define variables

    // The desired torso rotation relative to the world frame
    Eigen::Matrix3d torso_rot_world_ref = Eigen::Matrix3d::Zero();
    torso_rot_world_ref = REulerZYX(0, y_out_ref(OutID::PITCH), 0);
    // The IK solver torso rotation relative to the world frame
    Eigen::Matrix3d torso_rot_world = Eigen::Matrix3d::Zero();
    // The torso rotation error matrix between the desired and IK solver 
    Eigen::Matrix3d torso_rot_world_error = Eigen::Matrix3d::Zero();

    // Define Jacobians to be used by the IK solver
    // Jacobians of frames defined in the local world aligned coordinate frames
    Eigen::Matrix<double, 6, N_VEL_STATES> J_torso_world  = Eigen::Matrix<double, 6, N_VEL_STATES>::Zero();
    Eigen::Matrix<double, 6, N_VEL_STATES> J_stf_world  = Eigen::Matrix<double, 6, N_VEL_STATES>::Zero();
    Eigen::Matrix<double, 6, N_VEL_STATES> J_swf_world  = Eigen::Matrix<double, 6, N_VEL_STATES>::Zero();
    Eigen::Matrix<double, 6, N_VEL_STATES> J_com_world = Eigen::Matrix<double, 6, N_VEL_STATES>::Zero();

    // Jacobains of the frames defined relative to the stance foot world aligned coordinate frame
    Eigen::Matrix<double, 6, N_VEL_STATES> J_swf_rel  = Eigen::Matrix<double, 6, N_VEL_STATES>::Zero(); 
    Eigen::Matrix<double, 6, N_VEL_STATES> J_com_rel = Eigen::Matrix<double, 6, N_VEL_STATES>::Zero();
    Eigen::Matrix<double, 6, N_VEL_STATES> J_torso_rel = Eigen::Matrix<double, 6, N_VEL_STATES>::Zero();

    // A Jacobian for the outputs
    Eigen::Matrix<double, N_OUTPUTS, N_VEL_STATES> J  = Eigen::Matrix<double, N_OUTPUTS, N_VEL_STATES>::Zero(); 

    // A Jacobian given by JJt = J * J^T
    Eigen::Matrix<double, N_OUTPUTS, N_OUTPUTS> JJt = Eigen::Matrix<double, N_OUTPUTS, N_OUTPUTS>::Zero();

    // Rate of state change
    Eigen::Matrix<double, N_VEL_STATES, 1> v = Eigen::Matrix<double, N_VEL_STATES, 1>::Zero();

    // The current outputs
    OutputVec y_out = OutputVec::Zero();

    // The output error
    OutputVec y_out_error = OutputVec::Zero();

    // Keep track of the number of iterations
    int i = 0;

    while (true) {
        // Calculate the current outputs
        y_out = this->CalculateOutputs(q, stance_foot);

        // Calculate the output error
        y_out_error = y_out_ref - y_out;
        
        // Check if the outputs have converged
        if (y_out_error.norm() < this->eps) {
            std::cout << "IK converged. Iteration: " << i << std::endl;
            break;
        }

        // Check if the maximum number of iterations have been reached
        if (i >= this->max_number_of_iterations) {
            //std::cout << "q_out: " << q.transpose() << std::endl;
            //std::cout << "y_out: " << y_out.transpose() << std::endl;
            std::cout << "Max number of iterations reached" << std::endl;
            
            break;
        }

        // Compute the Jacobians for the torso and feet in the world aligned local frames
        pinocchio::computeJointJacobians(this->model, this->data, q);
        pinocchio::getFrameJacobian(this->model, this->data, TORSO_FRAME_ID, pinocchio::LOCAL_WORLD_ALIGNED, J_torso_world);
        pinocchio::getFrameJacobian(this->model, this->data, stf_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_stf_world);
        pinocchio::getFrameJacobian(this->model, this->data, swf_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_swf_world);
        
        // Compute the Jacobian of the CoM in the world frame
        if (this->use_static_com) {
            pinocchio::getFrameJacobian(model, data, STATIC_COM_FRAME_ID, pinocchio::LOCAL_WORLD_ALIGNED, J_com_world);
        } else {
            J_com_world.block<3, N_VEL_STATES>(0, 0) = pinocchio::jacobianCenterOfMass(this->model, this->data, q, false);
        }

        //std::cout << "J_com_world:\n" << J_com_world << std::endl;

        // Compute the Jacobian of the swing foot relative to the stance foot
        J_swf_rel = J_swf_world - J_stf_world;

        J_com_rel = J_com_world - J_stf_world;

        J_torso_rel = J_torso_world - J_stf_world;
            
        // Torso rotation
        // J(OutID::PITCH, 0) = J_torso_world(3, 0);
        J.block<1, N_VEL_STATES>(OutID::PITCH, 0) = J_torso_world.block<1, N_VEL_STATES>(3, 0);
            
        // Swing foot pos relative to stance foot position
        J.block<3, N_VEL_STATES>(OutID::SWF_POS_X, 0) = J_swf_rel.block<3, N_VEL_STATES>(0, 0);
            
        // Torso position relative to stance foot position
        J.block<1, N_VEL_STATES>(OutID::COM_POS_X, 0) = J_com_rel.block<1, N_VEL_STATES>(0, 0);
        J.block<1, N_VEL_STATES>(OutID::COM_POS_Z, 0) = J_com_rel.block<1, N_VEL_STATES>(2, 0);            

        // Compute J * J^T
        JJt.noalias() = J * J.transpose();

        // Add damping terms to the matrix so that it is always invertible
        JJt.diagonal().array() += this->damping_factor;

        // Compute the state update vector
        v.noalias() = J.transpose() * JJt.ldlt().solve(y_out_error);

        // Update the state vector
        q = pinocchio::integrate(this->model, q, this->alpha * v);
            
        // Update the number of iterations
        i++;
    }

    // Wrap the joint angles to be in the interval +-pi
    for (int k = N_POS_STATES_MJC - N_JOINTS + 1; k < N_POS_STATES_MJC; k++) {
        q(k) = WrapAngle(q(k));
    }

    GenPosVec q_nom = GenPosVec::Zero();

    q_nom(GenPosID::P_LHP) = q(GenPosID::P_LHP);
    q_nom(GenPosID::P_LKP) = q(GenPosID::P_LKP);

    q_nom(GenPosID::P_RHP) = q(GenPosID::P_RHP);
    q_nom(GenPosID::P_RKP) = q(GenPosID::P_RKP);

    pinocchio::forwardKinematics(this->model, this->data, q_nom);

    updateFramePlacements(this->model, this->data);

    Vector3d left_foot_pos_rel = data.oMf[LEFT_FOOT_FRAME_ID].translation() - data.oMf[LEFT_HIP_ROLL_FRAME_ID].translation();
    Vector3d right_foot_pos_rel = data.oMf[RIGHT_FOOT_FRAME_ID].translation() - data.oMf[RIGHT_HIP_ROLL_FRAME_ID].translation();

    Vector3d left_leg_joints = this->HipRollToFootIK(left_foot_pos_rel, Foot::LEFT);
    Vector3d right_leg_joints = this->HipRollToFootIK(right_foot_pos_rel, Foot::RIGHT);
    
    q(GenPosID::P_LHP) = left_leg_joints(1);
    q(GenPosID::P_LKP) = left_leg_joints(2);

    q(GenPosID::P_RHP) = right_leg_joints(1);
    q(GenPosID::P_RKP) = right_leg_joints(2);

    return i < this->max_number_of_iterations;
}

void Kinematics::PrintOutputs(OutputVec q_out) {
    std::cout << "y_out:" << std::endl;

    std::cout << "pitch: " << q_out(OutID::PITCH) << std::endl;

    std::cout << "swf pos x: " << q_out(OutID::SWF_POS_X) << std::endl;
    std::cout << "swf pos z: " << q_out(OutID::SWF_POS_Z) << std::endl;

    std::cout << "torso pos x: " << q_out(OutID::COM_POS_X) << std::endl;
    std::cout << "torso pos z: " << q_out(OutID::COM_POS_Z) << std::endl;
}

GenPosVec ConvertGenPosFromMujocoToPinocchio(GenPosVecMJC q_pos_mj) {
    GenPosVec q_pos_pin;

    q_pos_pin(GenPosID::P_X) = q_pos_mj(MujocoGenPosID::P_X);
    q_pos_pin(GenPosID::P_Z) = q_pos_mj(MujocoGenPosID::P_Z);

    Vector3d eulerXYZ;
    eulerXYZ << 0, q_pos_mj(MujocoGenPosID::R_Y), 0;
    Vector4d quatXYZ = EulerZYXToQuatXYZW(eulerXYZ);
    q_pos_pin(GenPosID::Q_X) = quatXYZ(0);
    q_pos_pin(GenPosID::Q_Y) = quatXYZ(1);
    q_pos_pin(GenPosID::Q_Z) = quatXYZ(2);
    q_pos_pin(GenPosID::Q_W) = quatXYZ(3);

    q_pos_pin(GenPosID::P_LHP) = q_pos_mj(MujocoGenPosID::P_LHP);
    q_pos_pin(GenPosID::P_LKP) = q_pos_mj(MujocoGenPosID::P_LKP);

    q_pos_pin(GenPosID::P_RHP) = q_pos_mj(MujocoGenPosID::P_RHP);
    q_pos_pin(GenPosID::P_RKP) = q_pos_mj(MujocoGenPosID::P_RKP);

    return q_pos_pin;
}

GenPosVecMJC ConvertGenPosFromPinocchioToMujoco(GenPosVec q_pos_pin) {
    GenPosVecMJC q_pos_mj = GenPosVecMJC::Zero();
    
    q_pos_mj(MujocoGenPosID::P_X) = q_pos_pin(GenPosID::P_X);
    q_pos_mj(MujocoGenPosID::P_Z) = q_pos_pin(GenPosID::P_Z);

    Vector4d quatXYZ = q_pos_pin.block<4, 1>(GenPosID::Q_X, 0);
    Vector3d eulerXYZ = QuatXYZWToEulerZYX(quatXYZ);
    q_pos_mj(MujocoGenPosID::R_Y) = eulerXYZ(1);

    q_pos_mj(MujocoGenPosID::P_LHP) = q_pos_pin(GenPosID::P_LHP);
    q_pos_mj(MujocoGenPosID::P_LKP) = q_pos_pin(GenPosID::P_LKP);

    q_pos_mj(MujocoGenPosID::P_RHP) = q_pos_pin(GenPosID::P_RHP);
    q_pos_mj(MujocoGenPosID::P_RKP) = q_pos_pin(GenPosID::P_RKP);

    return q_pos_mj;
}

GenVelVec ConvertGenVelFromMujocoToPinocchio(GenVelVec q_vel_mj) {
    GenVelVec q_vel_pin;

    q_vel_pin(GenVelID::V_X) = q_vel_mj(MujocoGenVelID::V_X);
    q_vel_pin(GenVelID::V_Z) = q_vel_mj(MujocoGenVelID::V_Z);

    q_vel_pin(GenVelID::W_Y) = q_vel_mj(MujocoGenVelID::W_Y);

    q_vel_pin(GenVelID::V_LHP) = q_vel_mj(MujocoGenVelID::V_LHP);
    q_vel_pin(GenVelID::V_LKP) = q_vel_mj(MujocoGenVelID::V_LKP);

    q_vel_pin(GenVelID::V_RHP) = q_vel_mj(MujocoGenVelID::V_RHP);
    q_vel_pin(GenVelID::V_RKP) = q_vel_mj(MujocoGenVelID::V_RKP);

    return q_vel_pin;
}

GenVelVec ConvertGenVelFromPinocchioToMujoco(GenVelVec q_vel_pin) {
    GenVelVec q_vel_mj;

    q_vel_mj(MujocoGenVelID::V_X) = q_vel_pin(GenVelID::V_X);
    q_vel_mj(MujocoGenVelID::V_Z) = q_vel_pin(GenVelID::V_Z);

    q_vel_mj(MujocoGenVelID::W_Y) = q_vel_pin(GenVelID::W_Y);

    q_vel_mj(MujocoGenVelID::V_LHP) = q_vel_pin(GenVelID::V_LHP);
    q_vel_mj(MujocoGenVelID::V_LKP) = q_vel_pin(GenVelID::V_LKP);

    q_vel_mj(MujocoGenVelID::V_RHP) = q_vel_pin(GenVelID::V_RHP);
    q_vel_mj(MujocoGenVelID::V_RKP) = q_vel_pin(GenVelID::V_RKP);

    return q_vel_mj;
}

JointVec ConvertJointVecFromPinocchioToMujoco(JointVec q_joint_pin) {
    JointVec q_joint_mj = JointVec::Zero();

    q_joint_mj(MujocoJointID::P_LHP) = q_joint_pin(JointID::P_LHP);
    q_joint_mj(MujocoJointID::P_LKP) = q_joint_pin(JointID::P_LKP);

    q_joint_mj(MujocoJointID::P_RHP) = q_joint_pin(JointID::P_RHP);
    q_joint_mj(MujocoJointID::P_RKP) = q_joint_pin(JointID::P_RKP);

    return q_joint_mj;
}

GenPosVec GetControlFramePosState(GenPosVec q) {
    // Get the quaternion orientation
    Vector4d quat_xyzw = q.block<4, 1>(GenPosID::Q_X, 0);

    // Get the zyx euler angle orientation
    Vector3d euler_zyx = QuatXYZWToEulerZYX(quat_xyzw);

    // Set the yaw to zero
    euler_zyx(2) = 0.0;

    // Convert the euler angles back to a quaternion
    quat_xyzw = EulerZYXToQuatXYZW(euler_zyx);

    // Add the modified quaternion back to the original state
    q.block<4, 1>(GenPosID::Q_X, 0) = quat_xyzw;

    return q;
}

GenVelVec GetControlFrameVelState(GenPosVec q_pos_world, GenVelVec q_vel_world) {
    // Get the quaternion orientation
    Vector4d quat_xyzw = q_pos_world.block<4, 1>(GenPosID::Q_X, 0);

    // Get the zyx euler angle orientation
    Vector3d euler_zyx = QuatXYZWToEulerZYX(quat_xyzw);

    // Get the yaw rotation
    double yaw = euler_zyx(2);

    // Linear velocity in the world frame
    Vector3d lin_vel_world = q_vel_world.block<3, 1>(GenVelID::V_X, 0);

    // Angular velocity in the world frame
    Vector3d ang_vel_world = q_vel_world.block<3, 1>(GenVelID::W_X, 0);

    // Roation matrix from the world frame to the control frame
    Eigen::Matrix3d R = R_z(yaw);

    // Calculate the linear velocity in the control frame
    Vector3d lin_vel_control = R.transpose() * lin_vel_world;

    // Calculate the angular velocity in the control frame
    Vector3d ang_vel_control = R.transpose() * ang_vel_world;

    // Initialize the velocity state vector in the control frame
    GenVelVec q_vel_control = GenVelVec::Zero();

    q_vel_control.block<3, 1>(GenVelID::V_X, 0) = lin_vel_control;

    q_vel_control.block<3, 1>(GenVelID::W_X, 0) = ang_vel_control;

    q_vel_control.block<N_JOINTS, 1>(GenVelID::V_LHP, 0) = q_vel_world.block<N_JOINTS, 1>(GenVelID::V_LHP, 0);

    return q_vel_control;
}

void PrintGenPos(GenPosVec q) {
    std::cout << "p: " << (q.block<3, 1>(GenPosID::P_X, 0)).transpose()
              << "\tq: " << (q.block<4, 1>(GenPosID::Q_X, 0)).transpose()
              << std::endl;
}

void PrintOutput(OutputVec y) {
    std::cout << "\tswf: " << (y.block<3, 1>(OutID::SWF_POS_X, 0)).transpose()
              << "\thip: " << (y.block<3, 1>(OutID::COM_POS_X, 0)).transpose()
              << std::endl;
}

Vector3d Kinematics::GetCoMPos(GenPosVec q_pos) {
    return pinocchio::centerOfMass(this->model, this->data, q_pos, false);
}

Vector3d Kinematics::GetFramePos(std::string frame_name) {
    int frame_id = model.getFrameId(frame_name);
    std::cout << "Frame: " << frame_id << std::endl;
    return data.oMf[frame_id].translation();
}

void Kinematics::UpdateFramePlacements(GenPosVec q) {
    // Calculate the forward kinematics for the model
    pinocchio::forwardKinematics(this->model, this->data, q);

    pinocchio::updateFramePlacements(this->model, this->data);
}