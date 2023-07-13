#ifndef kinematics_definitions_h
#define kinematics_definitions_h

#include <Eigen/Dense>

const int N_JOINTS = 4;
const int N_POS_STATES_MJC = 7;
const int N_POS_STATES = 11;
const int N_VEL_STATES_MJC = 7;
const int N_VEL_STATES = 10;
const int N_OUTPUTS = 5;

typedef Eigen::Matrix<double, 2, 1> Vector2d;
typedef Eigen::Matrix<double, 3, 1> Vector3d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 5, 1> Vector5d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;

typedef Eigen::Vector<double, N_JOINTS> JointVec;
typedef Eigen::Vector<double, N_POS_STATES> GenPosVec;
typedef Eigen::Vector<double, N_POS_STATES_MJC> GenPosVecMJC;
typedef Eigen::Vector<double, N_VEL_STATES> GenVelVec;
typedef Eigen::Vector<double, N_OUTPUTS> OutputVec;

namespace GenPosID {
    enum eGenPosID {
        P_X = 0, P_Y = 1, P_Z = 2,
        Q_X = 3, Q_Y = 4, Q_Z = 5, Q_W = 6,
        P_LHP = 7, P_LKP = 8,
        P_RHP = 9, P_RKP = 10,
    }; 
}

namespace MujocoGenPosID {
    enum eMujocoGenPosID  {
        P_X = 0, P_Z = 1, R_Y = 2,   // Base Pose
        P_LHP = 3, P_LKP = 4,        // Left Leg
        P_RHP = 5, P_RKP = 6         // Right Leg
    }; 
}

namespace GenVelID {
    enum eGenVelID {
        V_X = 0, V_Y = 1, V_Z = 2,
        W_X = 3, W_Y = 4, W_Z = 5,
        V_LHP = 6, V_LKP = 7,
        V_RHP = 8, V_RKP = 9,
    }; 
}

namespace MujocoGenVelID {
    enum eMujocoGenVelID {
        V_X = 0, V_Z = 1, W_Y = 2,
        V_LHP = 3, V_LKP = 4,
        V_RHP = 5, V_RKP = 6,
    }; 
}

namespace OutID {
    enum eOutID {
        PITCH = 0,
        SWF_POS_X = 1, SWF_POS_Z = 2,
        COM_POS_X = 3, COM_POS_Z = 4,
    };
}

namespace JointID {
    enum eJointID {
        P_LHP = 0, P_LKP = 1,
        P_RHP = 2, P_RKP = 3,
    }; 
}

namespace MujocoJointID {
    enum eMujocoJointID {
        P_LHP = 0, P_LKP = 1,
        P_RHP = 2, P_RKP = 3
    }; 
}

enum Foot {LEFT = 0, RIGHT = 1};

namespace IKOutputs {
    enum IKOutputs {
        TORSO_PITCH,
        TORSO_POS_X,
        TORSO_POS_Z,
        COM_POS_X,
        COM_POS_Z,
        STF_POS_X,
        STF_POS_Z,
        SWF_POS_X,
        SWF_POS_Z,
    };
}

#endif