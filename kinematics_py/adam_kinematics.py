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
        "P_X": 0, "P_Z": 1, "R_Y": 2, 
        "P_LHP": 3, "P_LKP": 4, "P_RHP": 5, "P_RKP": 6,
    }
    GEN_VEL_ID = {
        "V_X": 0, "V_Z": 1, "W_Y": 2,
        "V_LHP": 3, "V_LKP": 4, "V_RHP": 5, "V_RKP": 6
    }
    JOINT_ID = {"P_LHP": 0, "P_LKP": 1, "P_RHP": 2, "P_RKP": 3}

    N_JOINTS = 4
    N_POS_STATES = 7
    N_VEL_STATES = 7
    N_OUTPUTS = 5

    def __init__(self, urdf_path: str, mesh_path:str, use_static_com: bool=False, eps:float=1e-4, damping_factor:float=1e-6, alpha:float=0.2, max_iter:int=300):
        self.eps = eps
        self.alpha = alpha
        self.damping_factor = damping_factor
        self.max_iter = max_iter

        self.use_static_com = use_static_com

        self.pin_model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(urdf_path, mesh_path)

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

    def calcOutputs(self, q: np.ndarray, stanceFoot: bool) -> np.ndarray:
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

        y_out = np.array([
            q[Kinematics.GEN_POS_ID["R_Y"]],
            swf_pos_world[0] - stf_pos_world[0],
            swf_pos_world[2] - stf_pos_world[2],
            com_pos_world[0] - stf_pos_world[0],
            com_pos_world[2] - stf_pos_world[2]
        ])

        return y_out

    def calcGravityCompensation(self, q:np.ndarray, stanceFoot:bool) -> np.ndarray:
        if stanceFoot:
            stf_fid = self.LEFT_FOOT_FID
        else:
            stf_fid = self.RIGHT_FOOT_FID
        g = pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        Jc = pin.getFrameJacobian(self.pin_model, self.pin_data, stf_fid, pin.LOCAL_WORLD_ALIGNED)[0:3:2, :]

        Q, R = np.linalg.qr(Jc.transpose(), mode='complete')
        Su = np.hstack((np.zeros((5, 2)), np.eye(5)))
        S = np.vstack((np.zeros((3, 4)), np.eye(4))).transpose()
        return np.linalg.pinv(Su @ np.transpose(Q) @ np.transpose(S)) @ Su @ np.transpose(Q) @ g

        # Attempt with phantom ankle actuation for gravity comp
        # Su = np.hstack((np.zeros((5, 2)), np.eye(5)))
        # S = np.vstack((np.zeros((2, 5)), np.eye(5))).transpose()
        # tau = np.linalg.solve(Su @ np.transpose(Q) @ np.transpose(S), Su @ np.transpose(Q) @ g)
        # return tau[-4:]

        # Lol there are contact dynamics this doesn't consider
        # return g[-4:]

    def fk_CoM(self) -> np.ndarray:
        return self.pin.centerOfMass(self.pin_model, self.pin_data)

    def fk_StaticCom(self) -> np.ndarray:
        self.pin_data.oMf[self.STATIC_COM_FID].translation

    def v_CoM(self, q:np.ndarray, qd:np.ndarray) -> np.ndarray:
        Jcom = pin.jacobianCenterOfMass(self.pin_model, self.pin_data, q)
        return Jcom @ qd

    def v_StaticCom(self, q:np.ndarray, qd:np.array) -> np.ndarray:
        self.updateFramePlacements(q)
        Jstatic = pin.getFrameJacobian(self.pin_model, self.pin_data, self.STATIC_COM_FID, pin.LOCAL_WORLD_ALIGNED)
        return Jstatic @ qd

    def getVCom(self, q:np.ndarray, qd:np.ndarray) -> np.ndarray:
        if self.use_static_com:
            return self.v_StaticCom(q, qd)
        return self.v_CoM(q, qd)
    
    def getComMomentum(self, q, qd) -> np.ndarray:
        return pin.computeCentroidalMomentum(self.pin_model, self.pin_data, q, qd)
    
    def getSTFAngularMomentum(self, q, qd, stf) -> np.ndarray:
        comMom = self.getComMomentum(q, qd)
        L_com = comMom.linear[0:3:2]
        angMomCom = comMom.angular[1]
        y_out = self.calcOutputs(q, stf)
        return angMomCom + L_com[0] * y_out[Kinematics.OUT_ID["COM_POS_Z"]] - L_com[1] * y_out[Kinematics.OUT_ID["COM_POS_X"]]

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

            J_out = np.vstack([
                J_torso_world[4, :],
                J_swf_rel[0, :],
                J_swf_rel[2, :],
                J_com_rel[0, :],
                J_com_rel[2, :]
            ])

            JJt = J_out @ J_out.transpose()
            JJt += np.eye(Kinematics.N_OUTPUTS) * self.damping_factor

            v = J_out.transpose() @ np.linalg.solve(JJt, y_err)
            
            q = pin.integrate(self.pin_model, q, self.alpha * v)

            ii += 1

        q = Kinematics.wrapAngle(q)

        return q, ii < self.max_iter


    def fk_Frame(self, frame_name: str) -> np.ndarray:
        return self.pin_data.oMf[self.pin_model.getFrameId(frame_name)].translation

    def updateFramePlacements(self, q:np.ndarray) -> None:
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    @staticmethod
    def wrapAngle(q: np.ndarray) -> np.ndarray:
        return np.mod((q + np.pi), 2 * np.pi) - np.pi

    def getZeroPos(self) -> np.ndarray:
        q_zero = np.zeros((self.pin_model.nq,))
        return q_zero


if __name__ == "__main__":
    adamKin = Kinematics("rsc/models/adam2d.urdf", "rsc/models/")

    q_zero = adamKin.getZeroPos()
    stanceFoot = True
    y_pin = adamKin.calcOutputs(q_zero, stanceFoot)

    tau = adamKin.calcGravityCompensation(q_zero, stanceFoot)

    print(tau)


# if __name__ == "__main__":
#     from simulation_py.mujoco_interface import MujocoInterface

#     adamKin = Kinematics("rsc/models/adam2d.urdf", "rsc/models/")
#     mjInt = MujocoInterface("rsc/models/adam2d.xml", vis_enabled=False)

#     # First, check the Forward kinematics in the zero (base) position
#     q_zero = adamKin.getZeroPos()
#     stanceFoot = True
#     y_pin = adamKin.calcOutputs(q_zero, stanceFoot)
    
#     q_zerovel = np.zeros_like(mjInt.getGenVelocity())
#     mjInt.setState(q_zero, q_zerovel)
#     mjInt.forward()

#     mj_com = mjInt.getCoMPosition()
#     mj_feet = mjInt.getFootPos()
#     mj_stance = mj_feet[int(stanceFoot)]
#     mj_swing = mj_feet[int(not stanceFoot)]
#     y_mj = np.hstack(
#         (q_zero[Kinematics.GEN_POS_ID["R_Y"]], mj_swing - mj_stance, mj_com - mj_swing)
#     )

#     print("Check the Forward kinematics in the zero (base) position")
#     print("Y (pin): ", y_pin)
#     print("Y (mj) : ", y_mj) 
#     print("error  : ", y_pin - y_mj)

#     # Now, check the pinocchio inverse kinematics
#     max_error = 0
#     delta = 0.1
#     for ii in range(1000):
#         print("\n\nIK Test ", ii)
#         q_sol = adamKin.getZeroPos()

#         q_sol[Kinematics.GEN_POS_ID["R_Y"]] = np.random.random() * 0.4 - 0.2
#         q_sol[Kinematics.GEN_POS_ID["P_LHP"]] = np.random.random() * np.pi / 2 - np.pi / 4
#         q_sol[Kinematics.GEN_POS_ID["P_RHP"]] = np.random.random() * np.pi / 2 - np.pi / 4
#         q_sol[Kinematics.GEN_POS_ID["P_LKP"]] = np.random.random() * np.pi / 2
#         q_sol[Kinematics.GEN_POS_ID["P_RKP"]] = np.random.random() * np.pi / 2

#         y_sol = adamKin.calcOutputs(q_sol, stanceFoot)

#         q_ik = np.copy(q_zero)
#         q_ik[Kinematics.GEN_POS_ID["P_LHP"]] = q_sol[Kinematics.GEN_POS_ID["P_LHP"]] + np.random.random() * delta * 2 - delta
#         q_ik[Kinematics.GEN_POS_ID["P_RHP"]] = q_sol[Kinematics.GEN_POS_ID["P_RHP"]] + np.random.random() * delta * 2 - delta
#         q_ik[Kinematics.GEN_POS_ID["P_LKP"]] = q_sol[Kinematics.GEN_POS_ID["P_LKP"]] + np.random.random() * delta * 2 - delta
#         q_ik[Kinematics.GEN_POS_ID["P_RKP"]] = q_sol[Kinematics.GEN_POS_ID["P_RKP"]] + np.random.random() * delta * 2 - delta

#         q_ik, sol_found = adamKin.solveIK(q_ik, y_sol, stanceFoot)

#         y_ik = adamKin.calcOutputs(q_ik, stanceFoot)

#         print("\tq_sol: ", q_sol)
#         print("\tq_ik : ", q_ik)
#         print("\ty_sol: ", y_sol)
#         print("\ty_ik : ", y_ik)

#         if not sol_found:
#             print("Error in solution")
#             exit(0)

#         mjInt.setState(q_sol, q_zerovel)
#         mjInt.forward()

#         mj_com = mjInt.getCoMPosition()
#         mj_feet = mjInt.getFootPos()
#         mj_stance = mj_feet[int(stanceFoot)]
#         mj_swing = mj_feet[int(not stanceFoot)]
#         y_mj = np.hstack(
#             (q_sol[Kinematics.GEN_POS_ID["R_Y"]], mj_swing - mj_stance, mj_com - mj_stance)
#         )

#         print("\n\tCheck the Forward kinematics in the IK position vs Mujoco")
#         print("\tY (pin): ", y_sol)
#         print("\tY (mj) : ", y_mj) 
#         print("\terror  : ", y_sol - y_mj)

#         if np.linalg.norm(y_sol - y_mj) > max_error:
#             max_error = np.linalg.norm(y_sol - y_mj)
#             max_error_pin_y = y_sol
#             max_error_mjc_y = y_mj
#             max_error_q = q_sol
    
#     print("\n\nMaximum Error between Pinocchio and Mujoco Outputs: ", max_error, "\n\tPin Y: ", max_error_pin_y, "\n\tMjc Y: ", max_error_mjc_y, "\n\tQ: ", max_error_q)


# if __name__ == "__main__":

#     model, _, _ = pin.buildModelsFromUrdf("/home/wcompton/Repos/ADAM-2D/rsc/models/adam_planar.urdf", "/home/wcompton/Repos/ADAM-2D/rsc/models/")

#     data = model.createData()
#     pin.updateFramePlacements(model, data)

#     for ii in range(model.njoints):
#         print(model.names[ii], ": ", data.oMi[ii].translation)


#     zero_config = np.zeros_like(pin.randomConfiguration(model))

#     for jj in range(zero_config.size):
#         print(f"\n\n{jj}\n")
#         zero_config = np.zeros_like(pin.randomConfiguration(model))
#         zero_config[jj] = 0.1
#         # rng_config = pin.randomConfiguration(model)
#         # print(rng_config)

#         pin.forwardKinematics(model, data, zero_config)
#         pin.updateFramePlacements(model, data)

#         for ii in range(model.njoints):
#             print(model.names[ii], ": ", data.oMi[ii].translation, "\n", data.oMi[ii].rotation, "\n")
