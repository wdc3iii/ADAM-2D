import numpy as np
import pinocchio as pin

class Kinematics:

    
    OUT_ID = {
        "PITCH": 0, "SWF_POS_X": 1, "SWF_POS_Z": 2,
        "COM_POS_X": 3, "COM_POS_Z": 4
    }
    OUT_IK_ID = {
        "R_X": 0, "R_Y": 1, "R_Z": 2, "SWF_POS_X": 3, "SWF_POS_Z": 4,
        "COM_POS_X": 5, "COM_POS_Z": 6
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
    N_JAC_STATES = 10
    N_VEL_STATES_MJC = 7
    N_VEL_STATES = 10
    N_OUTPUTS = 5
    N_OUTPUTS_IK = 7

    def __init__(self, urdf_path: str, mesh_path:str, use_static_com: bool=False, eps:float=1e-4, damping_factor:float=1e-6, alpha:float=0.2, max_iter:int=300):
        self.eps = eps
        self.alpha = alpha
        self.damping_factor = damping_factor
        self.max_iter = max_iter

        self.use_static_com = use_static_com

        self.pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path, "/home/wcompton/Repos/ADAM-2D/rsc/models/", pin.JointModelFreeFlyer())
        # self.pin_model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_path, pin.JointModelPlanar())

        self.pin_data = self.pin_model.createData()

        self.TORSO_FID = self.pin_model.getFrameId("torso")
        self.LEFT_HIP_YAW_FID = self.pin_model.getFrameId("left_hip_yaw")
        self.RIGHT_HIP_YAW_FID = self.pin_model.getFrameId("right_hip_yaw")
        self.LEFT_FOOT_FID = self.pin_model.getFrameId("left_foot")
        self.RIGHT_FOOT_FID = self.pin_model.getFrameId("right_foot")
        self.LEFT_HIP_ROLL_FID = self.pin_model.getFrameId("left_hip_roll")
        self.RIGHT_HIP_ROLL_FID = self.pin_model.getFrameId("right_hip_roll")
        self.LEFT_HIP_PITCH_FID = self.pin_model.getFrameId("left_hip_pitch")
        self.RIGHT_HIP_PITCH_FID = self.pin_model.getFrameId("right_hip_pitch")
        self.LEFT_SHIN_FID = self.pin_model.getFrameId("left_shin")
        self.RIGHT_SHIN_FID = self.pin_model.getFrameId("right_shin")
        self.STATIC_COM_FID = self.pin_model.getFrameId("static_com")

        q_nom = self.getZeroPos()
        self.updateFramePlacements(q_nom)

    def calcOutputs(self, q: np.ndarray, stanceFoot: bool):
        self.updateFramePlacements(q)
        
        if stanceFoot:
            stf_fid = self.LEFT_FOOT_FID
            swf_fid = self.RIGHT_FOOT_FID
        else:
            stf_fid = self.RIGHT_FOOT_FID
            swf_fid = self.LEFT_FOOT_FID

        

        if self.use_static_com:
            com_pos_world = self.pin_data.oMf[self.STATIC_COM_FID].translation
        else:
            com_pos_world = pin.centerOfMass(self.pin_model, self.pin_data)
        
        stf_pos_world = self.pin_data.oMf[stf_fid].translation
        swf_pos_world = self.pin_data.oMf[swf_fid].translation

        y_out = np.zeros((5,))
        torso_euler_zyx = Kinematics.QuatXYZWToEulerZYX(q[self.GEN_POS_ID["Q_X"]:self.GEN_POS_ID["Q_W"] + 1])
        y_out[Kinematics.OUT_ID["PITCH"]] = torso_euler_zyx[1]
        y_out[Kinematics.OUT_ID["SWF_POS_X"]] = swf_pos_world[0] - stf_pos_world[0]
        y_out[Kinematics.OUT_ID["SWF_POS_Z"]] = swf_pos_world[2] - stf_pos_world[2]
        y_out[Kinematics.OUT_ID["COM_POS_X"]] = com_pos_world[0] - stf_pos_world[0]
        y_out[Kinematics.OUT_ID["COM_POS_Z"]] = com_pos_world[2] - stf_pos_world[2]

        return y_out

    def fk_CoM(self, q: np.ndarray):
        return self.pin.centerOfMass(self.pin_model, self.pin_data)

    def fk_StaticCom(self, q: np.ndarray):
        self.pin_data.oMf[self.STATIC_COM_FID].translation

    def v_CoM(self, q:np.ndarray, qd:np.ndarray) -> np.ndarray:
        pass

    def v_StaticCom(self, q:np.ndarray, qd:np.array) -> np.ndarray:
        pass 

    def getVCom(self, q:np.ndarray, qd:np.ndarray) -> np.ndarray:
        if self.use_static_com:
            return self.v_StaticCom(q, qd)
        return self.v_CoM(q, qd)

    def solveIK(self, q: np.ndarray, y_des: np.ndarray, stanceFoot: bool) -> tuple:
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

            pin.computeJointJacobians(self.pin_model, self.pin_data, q)
            J_torso_world = pin.getFrameJacobian(self.pin_model, self.pin_data, self.TORSO_FID, pin.LOCAL_WORLD_ALIGNED)
            J_stf_world = pin.getFrameJacobian(self.pin_model, self.pin_data, stf_fid, pin.LOCAL_WORLD_ALIGNED)
            J_swf_world = pin.getFrameJacobian(self.pin_model, self.pin_data, swf_fid, pin.LOCAL_WORLD_ALIGNED)

            if self.use_static_com:
                J_com_world = pin.getFrameJacobian(self.pin_model, self.pin_data, self.STATIC_COM_FID, pin.LOCAL_WORLD_ALIGNED)
            else:
                J_lin = pin.jacobianCenterOfMass(self.pin_model, self.pin_data, q, False)
                J_com_world = np.vstack((J_lin, np.zeros_like(J_lin)))


            J_swf_rel = J_swf_world - J_stf_world
            J_com_rel = J_com_world - J_stf_world

            J_out = np.zeros((Kinematics.N_OUTPUTS_IK, Kinematics.N_JAC_STATES))
            J_out[Kinematics.OUT_IK_ID["R_X"]:Kinematics.OUT_IK_ID['R_Z'] + 1, :] = J_torso_world[3:6, :]
            J_out[Kinematics.OUT_IK_ID["SWF_POS_X"], :] = J_swf_rel[0, :]
            J_out[Kinematics.OUT_IK_ID["SWF_POS_Z"], :] = J_swf_rel[2, :]
            J_out[Kinematics.OUT_IK_ID["COM_POS_X"], :] = J_com_rel[0, :]
            J_out[Kinematics.OUT_IK_ID["COM_POS_Z"], :] = J_com_rel[2, :]

            JJt = J_out @ J_out.transpose()
            JJt += np.eye(Kinematics.N_OUTPUTS_IK) * self.damping_factor

            v = J_out.transpose() @ np.linalg.solve(JJt, np.hstack((0, y_err[0], 0, y_err[1:])))
            
            q = pin.integrate(self.pin_model, q, self.alpha * v)

            ii += 1

        q = Kinematics.wrapAngle(q)

        return q, ii < self.max_iter


    def fk_Frame(self, frame_name: str):
        return self.pin_data.oMf[self.pin_model.getFrameId(frame_name)].translation

    def updateFramePlacements(self, q:np.ndarray) -> None:
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    @staticmethod
    def convertGenPosMJCtoPIN(q_mjc:np.ndarray):
        quat = Kinematics.EulerZYXToQuatXYZW(np.array([0, q_mjc[Kinematics.GEN_POS_ID_MJC["R_Y"]], 0]))
        q = np.zeros((Kinematics.N_POS_STATES,))
        q[Kinematics.GEN_POS_ID["Q_X"]:Kinematics.GEN_POS_ID["Q_W"]+1] = quat
        q[Kinematics.GEN_POS_ID["P_LHP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_LHP"]]
        q[Kinematics.GEN_POS_ID["P_RHP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_RHP"]]
        q[Kinematics.GEN_POS_ID["P_LKP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_LKP"]]
        q[Kinematics.GEN_POS_ID["P_RKP"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_RKP"]]
        q[Kinematics.GEN_POS_ID["P_X"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_X"]]
        q[Kinematics.GEN_POS_ID["P_Z"]] = q_mjc[Kinematics.GEN_POS_ID_MJC["P_Z"]]
        return q

    @staticmethod 
    def convertGenPosPINtoMJC(q_pin:np.ndarray):
        eulerXYZ = Kinematics.QuatXYZWToEulerZYX(q_pin[Kinematics.GEN_POS_ID["Q_X"]:Kinematics.GEN_POS_ID["Q_W"]+1])
        q = np.zeros((Kinematics.N_POS_STATES_MJC,))
        q[Kinematics.GEN_POS_ID_MJC["R_Y"]] = eulerXYZ[1]
        q[Kinematics.GEN_POS_ID_MJC["P_LHP"]] = q_pin[Kinematics.GEN_POS_ID["P_LHP"]]
        q[Kinematics.GEN_POS_ID_MJC["P_RHP"]] = q_pin[Kinematics.GEN_POS_ID["P_RHP"]]
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
        qd = np.zeros((Kinematics.N_VEL_STATES_MJC))
        qd[Kinematics.GEN_VEL_ID_MJC["V_X"]] = qd_pin[Kinematics.GEN_VEL_ID["V_X"]]
        qd[Kinematics.GEN_VEL_ID_MJC["V_Z"]] = qd_pin[Kinematics.GEN_VEL_ID["V_Z"]]
        qd[Kinematics.GEN_VEL_ID_MJC["W_Y"]] = qd_pin[Kinematics.GEN_VEL_ID["W_Y"]]
        qd[Kinematics.GEN_VEL_ID_MJC["V_LHP"]] = qd_pin[Kinematics.GEN_VEL_ID["V_LHP"]]
        qd[Kinematics.GEN_VEL_ID_MJC["V_LKP"]] = qd_pin[Kinematics.GEN_VEL_ID["V_LKP"]]
        qd[Kinematics.GEN_VEL_ID_MJC["V_RHP"]] = qd_pin[Kinematics.GEN_VEL_ID["V_RHP"]]
        qd[Kinematics.GEN_VEL_ID_MJC["V_RKP"]] = qd_pin[Kinematics.GEN_VEL_ID["V_RKP"]]
        return qd

    @staticmethod
    def wrapAngle(q: np.ndarray) -> np.ndarray:
        return np.mod((q + np.pi), 2 * np.pi) - np.pi

    @staticmethod
    def QuatXYZWToEulerZYX(quat_xyzw: np.ndarray) -> np.ndarray:
        return np.array([
            np.arctan2(2 * (quat_xyzw[3] * quat_xyzw[0] + quat_xyzw[1] * quat_xyzw[2]), 1 - 2 * (quat_xyzw[0] * quat_xyzw[0] + quat_xyzw[1] * quat_xyzw[1])),
            -np.pi / 2 + 2*np.arctan2(np.sqrt(1 + 2 *(quat_xyzw[3] * quat_xyzw[1] - quat_xyzw[0] * quat_xyzw[2])), np.sqrt(1 - 2 * (quat_xyzw[3] * quat_xyzw[1] - quat_xyzw[0] * quat_xyzw[2]))),
            np.arctan2(2*(quat_xyzw[3] * quat_xyzw[2] + quat_xyzw[0] * quat_xyzw[1]), 1 - 2*(quat_xyzw[1] * quat_xyzw[1] + quat_xyzw[2] * quat_xyzw[2]))
        ])
    
    @staticmethod
    def EulerZYXToQuatXYZW(euler_zyx:np.ndarray) -> np.ndarray:
        cr = np.cos(euler_zyx[0] / 2)
        sr = np.sin(euler_zyx[0] / 2)
        cp = np.cos(euler_zyx[1] / 2)
        sp = np.sin(euler_zyx[1] / 2)
        cy = np.cos(euler_zyx[2] / 2)
        sy = np.sin(euler_zyx[2] / 2)
        return np.array([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy
        ])

    def getZeroPos(self) -> np.ndarray:
        q_zero = np.zeros((self.pin_model.nq,))
        q_zero[Kinematics.GEN_POS_ID["Q_W"]] = 1
        return q_zero


if __name__ == "__main__":
    from simulation_py.mujoco_interface import MujocoInterface

    adamKin = Kinematics("rsc/models/adam.urdf", "rsc/models/")
    mjInt = MujocoInterface("rsc/models/adam.xml", vis_enabled=False)

    # First, check the Forward kinematics in the zero (base) position
    q_zero = adamKin.getZeroPos()
    stanceFoot = True
    y_pin = adamKin.calcOutputs(q_zero, stanceFoot)
    
    mj_q = Kinematics.convertGenPosPINtoMJC(q_zero)
    mj_qvel = np.zeros_like(mjInt.getGenVelocity())
    mjInt.setState(mj_q, mj_qvel)
    mjInt.forward()

    mj_com = mjInt.getCoMPosition()
    mj_feet = mjInt.getFootPos()
    mj_stance = mj_feet[int(stanceFoot)]
    mj_swing = mj_feet[int(not stanceFoot)]
    y_mj = np.hstack(
        (mj_q[Kinematics.GEN_POS_ID_MJC["R_Y"]], mj_swing - mj_stance, mj_com - mj_swing)
    )

    print("Check the Forward kinematics in the zero (base) position")
    print("Y (pin): ", y_pin)
    print("Y (mj) : ", y_mj) 
    print("error  : ", y_pin - y_mj)

    # Now, check the pinocchio inverse kinematics
    max_error = 0
    delta = 0.1
    for ii in range(1000):
        print("\n\nIK Test ", ii)
        q_sol = adamKin.getZeroPos()

        pitch = np.random.random() * 0.4 - 0.2
        q_sol[Kinematics.GEN_POS_ID["Q_X"]:Kinematics.GEN_POS_ID["Q_W"] + 1] = Kinematics.EulerZYXToQuatXYZW(np.array([0, pitch, 0]))
        q_sol[Kinematics.GEN_POS_ID["P_LHP"]] = np.random.random() * np.pi / 2 - np.pi / 4
        q_sol[Kinematics.GEN_POS_ID["P_RHP"]] = np.random.random() * np.pi / 2 - np.pi / 4
        q_sol[Kinematics.GEN_POS_ID["P_LKP"]] = np.random.random() * np.pi / 2
        q_sol[Kinematics.GEN_POS_ID["P_RKP"]] = np.random.random() * np.pi / 2

        y_sol = adamKin.calcOutputs(q_sol, stanceFoot)

        q_ik = np.copy(q_zero)
        q_ik[Kinematics.GEN_POS_ID["P_LHP"]] = q_sol[Kinematics.GEN_POS_ID["P_LHP"]] + np.random.random() * delta * 2 - delta
        q_ik[Kinematics.GEN_POS_ID["P_RHP"]] = q_sol[Kinematics.GEN_POS_ID["P_RHP"]] + np.random.random() * delta * 2 - delta
        q_ik[Kinematics.GEN_POS_ID["P_LKP"]] = q_sol[Kinematics.GEN_POS_ID["P_LKP"]] + np.random.random() * delta * 2 - delta
        q_ik[Kinematics.GEN_POS_ID["P_RKP"]] = q_sol[Kinematics.GEN_POS_ID["P_RKP"]] + np.random.random() * delta * 2 - delta

        q_ik, sol_found = adamKin.solveIK(q_ik, y_sol, stanceFoot)

        y_ik = adamKin.calcOutputs(q_ik, stanceFoot)

        print("\tq_sol: ", q_sol)
        print("\tq_ik : ", q_ik)
        print("\ty_sol: ", y_sol)
        print("\ty_ik : ", y_ik)

        if not sol_found:
            print("Error in solution")
            exit(0)

        mj_q = Kinematics.convertGenPosPINtoMJC(q_sol)
        mj_qvel = np.zeros_like(mjInt.getGenVelocity())
        mjInt.setState(mj_q, mj_qvel)
        mjInt.forward()

        mj_com = mjInt.getCoMPosition()
        mj_feet = mjInt.getFootPos()
        mj_stance = mj_feet[int(stanceFoot)]
        mj_swing = mj_feet[int(not stanceFoot)]
        y_mj = np.hstack(
            (mj_q[Kinematics.GEN_POS_ID_MJC["R_Y"]], mj_swing - mj_stance, mj_com - mj_stance)
        )

        print("\n\tCheck the Forward kinematics in the IK position vs Mujoco")
        print("\tY (pin): ", y_sol)
        print("\tY (mj) : ", y_mj) 
        print("\terror  : ", y_sol - y_mj)

        if np.linalg.norm(y_sol - y_mj) > max_error:
            max_error = np.linalg.norm(y_sol - y_mj)
            max_error_pin_y = y_sol
            max_error_mjc_y = y_mj
            max_error_pin_q = q_sol
            max_error_mjc_q = mj_q
    
    print("\n\nMaximum Error between Pinocchio and Mujoco Outputs: ", max_error, "\n\tPin Y: ", max_error_pin_y, "\n\tMjc Y: ", max_error_mjc_y, "\n\tPin Q: ", max_error_pin_q, "\n\tMjc Q: ", max_error_mjc_q)

        


