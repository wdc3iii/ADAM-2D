import numpy as np
import pinocchio as pin

class Kinematics:

    
    OUT_ID = {
        "PITCH": 0, "SWF_POS_X": 1, "SWF_POS_Z": 2,
        "COM_POS_X": 3, "COM_POS_Z": 4
    }
    GEN_POS_ID = {
        "P_X": 0, "P_Y": 1, "P_Z": 2, "Q_X": 3, "Q_Y": 4, "Q_Z": 5, "Q_W": 6,
        "P_LHP": 7, "P_LKP": 8, "P_RHP": 9, "P_RKP": 10,
    }
    GEN_POS_ID_MJC = {
        "P_X": 0, "P_Z": 1, "R_Y": 2,
        "P_LHP": 3, "P_LKP": 4, "P_RHP": 5, "P_RKP": 6
    }
    GEN_VEL_ID = {
        "V_X": 0, "V_Y": 1, "V_Z": 2, "W_X": 3, "W_Y": 4, "W_Z": 5,
        "V_LHP": 6, "V_LKP": 7, "V_RHP": 8, "V_RKP": 9,
    }
    GEN_VEL_ID_MJC = {
        "V_X": 0, "V_Z": 1, "W_Y": 2,
        "V_LHP": 3, "V_LKP": 4, "V_RHP": 5, "V_RKP": 6,
    }
    JOINT_ID = {"P_LHP": 0, "P_LKP": 1, "P_RHP": 2, "P_RKP": 3}

    N_JOINTS = 4
    N_POS_STATES_MJC = 7
    N_POS_STATES = 11
    N_VEL_STATES_MJC = 7
    N_VEL_STATES = 10
    N_OUTPUTS = 5

    def __init__(self, urdf_path: str, mesh_path:str, use_static_com: bool=False, eps:float=1e-4, damping_factor:float=1e-6, alpha:float=0.2, max_iter:int=300):
        self.eps = eps
        self.alpha = alpha
        self.damping_factor = damping_factor
        self.max_iter = max_iter

        self.use_static_com = use_static_com

        self.pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path, "/home/wcompton/Repos/ADAM-2D/rsc/models/", pin.JointModelFreeFlyer())
        # self.pin_model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_path, pin.JointModelPlanar())

        self.pin_data = self.pin_model.createData()

        self.TORSO_FRAME_ID = self.pin_model.getFrameId("torso")
        self.LEFT_HIP_YAW_FRAME_ID = self.pin_model.getFrameId("left_hip_yaw")
        self.RIGHT_HIP_YAW_FRAME_ID = self.pin_model.getFrameId("right_hip_yaw")
        self.LEFT_FOOT_FRAME_ID = self.pin_model.getFrameId("left_foot")
        self.RIGHT_FOOT_FRAME_ID = self.pin_model.getFrameId("right_foot")
        self.LEFT_HIP_ROLL_FRAME_ID = self.pin_model.getFrameId("left_hip_roll")
        self.RIGHT_HIP_ROLL_FRAME_ID = self.pin_model.getFrameId("right_hip_roll")
        self.LEFT_HIP_PITCH_FRAME_ID = self.pin_model.getFrameId("left_hip_pitch")
        self.RIGHT_HIP_PITCH_FRAME_ID = self.pin_model.getFrameId("right_hip_pitch")
        self.LEFT_SHIN_FRAME_ID = self.pin_model.getFrameId("left_shin")
        self.RIGHT_SHIN_FRAME_ID = self.pin_model.getFrameId("right_shin")
        self.STATIC_COM_FRAME_ID = self.pin_model.getFrameId("static_com")

        q_nom = np.zeros((self.pin_data.nq,))
        self.updateFramePlacements(q_nom)

    def calcOutputs(self, q: np.ndarray, stanceFoot: bool):
        if stanceFoot:
            stf_fid = self.LEFT_FOOT_FID
            swf_fid = self.RIGHT_FOOT_FID
        else:
            stf_fid = self.RIGHT_FOOT_FID
            swf_fid = self.LEFT_FOOT_FID

        self.updateFramePlacements()

        if self.use_static_com:
            com_pos_world = self.pin_data.oMf[self.STATIC_COM_FID].translation()
        else:
            com_pos_world = self.pin.centerOfMass(self.pin_model, self.pin_data)
        
        stf_pos_world = self.pin_data.oMf[stf_fid].translation()
        swf_pos_world = self.pin_data.oMf[swf_fid].translation()

        y_out = np.zeros((5,))
        torso_euler_zyx = pin.QuatXYZWToEulerZYX(q[self.GEN_POS_ID["Q_X"]:self.GEN_POS_ID["Q_W"] + 1])
        y_out[Kinematics.OUT_ID["PITCH"]] = torso_euler_zyx(1)
        y_out[Kinematics.OUT_ID["SWF_POS_X"]] = swf_pos_world(0) - stf_pos_world(0)
        y_out[Kinematics.OUT_ID["SWF_POS_Z"]] = swf_pos_world(2) - stf_pos_world(2)
        y_out[Kinematics.OUT_ID["COM_POS_X"]] = com_pos_world(0) - stf_pos_world(0)
        y_out[Kinematics.OUT_ID["COM_POS_Z"]] = com_pos_world(2) - stf_pos_world(2)

        return y_out

    def fk_CoM(self, q: np.ndarray):
        return self.pin.centerOfMass(self.pin_model, self.pin_data)

    def fk_StaticCom(self, q: np.ndarray):
        self.pin_data.oMf[self.STATIC_COM_FID].translation()

    def solveIK(self, q: np.ndarray, y_des: np.ndarray, stanceFoot: bool):
        if stanceFoot:
            stf_fid = self.LEFT_FOOT_FID
            swf_fid = self.RIGHT_FOOT_FID
        else:
            stf_fid = self.RIGHT_FOOT_FID
            swf_fid = self.LEFT_FOOT_FID

        ii = 0
        while ii < self.max_iter:
            y_out = self.calcOutputs(q, stanceFoot)
            y_err = y_des - y_out

            if np.linalg.norm(y_err) < self.eps:
                break

            pin.computeJointJacobian(self.pin_model, self.pin_data, q)
            J_torso_world = pin.getFrameJacobian(self.pin_model, self.pin_data, self.TORSO_FRAME_ID, pin.LOCAL_WORLD_ALIGNED)
            J_stf_world = pin.getFrameJacobian(self.pin_model, self.pin_data, stf_fid, pin.LOCAL_WORLD_ALIGNED)
            J_swf_world = pin.getFrameJacobian(self.pin_model, self.pin_data, swf_fid, pin.LOCAL_WORLD_ALIGNED)

            if self.use_static_com:
                J_com_world = pin.getFrameJacobian(self.pin_model, self.pin_Data, self.STATIC_COM_FRAME_ID, pin.LOCAL_WORLD_ALIGNED)
            else:
                J_com_world = pin.jacobianCenterOfMass(self.pin_model, self.data, q, False)

            J_swf_rel = J_swf_world - J_stf_world
            J_com_rel = J_com_world - J_stf_world

            J_out = np.zeros((Kinematics.N_OUTPUTS, Kinematics.N_STATES))
            J_out[Kinematics.OUT_ID["PITCH"], :] = J_torso_world[3, :]
            J_out[Kinematics.OUT_ID["SWF_POS_X"], :] = J_swf_rel[0, :]
            J_out[Kinematics.OUT_ID["SWF_POS_Z"], :] = J_swf_rel[2, :]
            J_out[Kinematics.OUT_ID["COM_POS_X"], :] = J_com_rel[0, :]
            J_out[Kinematics.OUT_ID["COM_POS_Z"], :] = J_com_rel[2, :]

            JJt = J_out * J_out.transpose()
            JJt += np.eye(Kinematics.N_OUTPUTS) * self.damping_factor

            v = J_out.transpose * JJt
            
            q = pin.integrate(self.pin_model, q, self.alpha * v)

            ii += 1

        q = Kinematics.WrapAngle(q)

        return ii < self.max_iter


    def fk_Frame(self, frame_name: str):
        return self.pin_data.oMf[self.pin_model.getFrameId(frame_name)].translation()

    def updateFramePlacements(self, q:np.ndarray) -> None:
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    @staticmethod
    def convertGenPosMJCtoPIN(q_mjc:np.ndarray):
        quat = pin.EulerZYXToQuatXYZW(q_mjc[Kinematics.GEN_POS_ID["Q_X"]:Kinematics.GEN_POS_ID["Q_W"]+1])
        q = np.zeros((Kinematics.N_POS_STATES,))
        q[Kinematics.GEN_POS_ID["Q_X"]:Kinematics.GEN_POS_ID["Q_W"]+1] = quat
        q[Kinematics.GEN_POS_ID["P_LHP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_LHP"]]
        q[Kinematics.GEN_POS_ID["P_RHP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["R_LHP"]]
        q[Kinematics.GEN_POS_ID["P_LKP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_LKP"]]
        q[Kinematics.GEN_POS_ID["P_RKP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_RKP"]]
        q[Kinematics.GEN_POS_ID["P_X"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_X"]]
        q[Kinematics.GEN_POS_ID["P_Z"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_Z"]]
        return q

    @staticmethod 
    def convertGenPosPINtoMJC(q_pin:np.ndarray):
        eulerXYZ = pin.EulerZYXToQuatXYZW(q_pin[Kinematics.GEN_POS_ID["Q_X"]:Kinematics.GEN_POS_ID["Q_W"]+1])
        q = np.zeros((Kinematics.N_POS_STATES_MJC,))
        q[Kinematics.GEN_POS_ID_MJC["PITCH"]] = eulerXYZ[1]
        q[Kinematics.GEN_POS_ID_MJC["P_LHP"]] = q_pin[Kinematics.GEN_POS_ID["P_LHP"]]
        q[Kinematics.GEN_POS_ID_MJC["P_RHP"]] = q_pin[Kinematics.GEN_POS_ID["R_LHP"]]
        q[Kinematics.GEN_POS_ID_MJC["P_LKP"]] = q_pin[Kinematics.GEN_POS_ID["P_LKP"]]
        q[Kinematics.GEN_POS_ID_MJC["P_RKP"]] = q_pin[Kinematics.GEN_POS_ID["P_RKP"]]
        q[Kinematics.GEN_POS_ID_MJC["P_X"]] = q_pin[Kinematics.GEN_POS_ID["P_X"]]
        q[Kinematics.GEN_POS_ID_MJC["P_Z"]] = q_pin[Kinematics.GEN_POS_ID["P_Z"]]
        return q
        

    @staticmethod
    def convertGenVelMJCtoPin(qd_mjc:np.ndarray):
        qd = np.zeros((Kinematics.N_VEL_STATES))
        qd[Kinematics.GEN_VEL_ID["V_X"]] = qd_mjc[Kinematics.GEN_VEL_ID_MJC["V_X"]]
        qd[Kinematics.GEN_VEL_ID["V_Z"]] = qd_mjc[Kinematics.GEN_VEL_ID_MJC["V_Z"]]
        qd[Kinematics.GEN_VEL_ID["W_Y"]] = qd_mjc[Kinematics.GEN_VEL_ID_MJC["W_Y"]]
        qd[Kinematics.GEN_VEL_ID["V_LHP"]] = qd_mjc[Kinematics.GEN_VEL_ID_MJC["V_LHP"]]
        qd[Kinematics.GEN_VEL_ID["V_LKP"]] = qd_mjc[Kinematics.GEN_VEL_ID_MJC["V_LKP"]]
        qd[Kinematics.GEN_VEL_ID["V_RHP"]] = qd_mjc[Kinematics.GEN_VEL_ID_MJC["V_RHP"]]
        qd[Kinematics.GEN_VEL_ID["V_RKP"]] = qd_mjc[Kinematics.GEN_VEL_ID_MJC["V_RKP"]]
        return qd

    @staticmethod
    def convertGenVelPintoMJC(qd_pin:np.ndarray):
        qd = np.zeros((Kinematics.N_VEL_STATES))
        qd[Kinematics.GEN_VEL_ID["V_X"]] = qd_pin[Kinematics.GEN_VEL_ID_MJC["V_X"]]
        qd[Kinematics.GEN_VEL_ID["V_Z"]] = qd_pin[Kinematics.GEN_VEL_ID_MJC["V_Z"]]
        qd[Kinematics.GEN_VEL_ID["W_Y"]] = qd_pin[Kinematics.GEN_VEL_ID_MJC["W_Y"]]
        qd[Kinematics.GEN_VEL_ID["V_LHP"]] = qd_pin[Kinematics.GEN_VEL_ID_MJC["V_LHP"]]
        qd[Kinematics.GEN_VEL_ID["V_LKP"]] = qd_pin[Kinematics.GEN_VEL_ID_MJC["V_LKP"]]
        qd[Kinematics.GEN_VEL_ID["V_RHP"]] = qd_pin[Kinematics.GEN_VEL_ID_MJC["V_RHP"]]
        qd[Kinematics.GEN_VEL_ID["V_RKP"]] = qd_pin[Kinematics.GEN_VEL_ID_MJC["V_RKP"]]
        return qd

    @staticmethod
    def wrapAngle(q: np.ndarray):
        return np.mod((q + np.pi), 2 * np.pi) - np.pi


if __name__ == "__main__":
    adamKin = Kinematics("/home/wcompton/Repos/ADAM-2D/rsc/models/adam.urdf", "/home/wcompton/Repos/ADAM-2D/rsc/models/")

