import numpy as np
from kinematics_py.adam_kinematics import Kinematics
from control_py.bezier import Bezier
from control_py.poly import Poly
from plot.logger import Logger

class HLIPController:

    def __init__(
            self, T_SSP:float, z_ref:float, urdf_path:str, mesh_path:str, v_ref:float=0, pitch_ref:float=0.025,
            v_max_change:float=0.1, z_max_change:float=0.02, x_bez:np.ndarray=np.array([0,0,1,1,1]),
            vswf_tof:float=0.05, vswf_imp:float=-0.05, zswf_max:float=0.1, pswf_max:float=0.7,
            use_static_com:bool=False, T_DSP:float=0, log_path:str="plot/log_ctrl.csv"
        ):
        self.T_SSP = T_SSP
        self.T_DSP = T_DSP
        self.T_SSP_goal = T_SSP
        self.g = 9.81

        self.z_ref_goal = z_ref
        self.z_ref = z_ref
        self.calcLambda()
        self.calcSigma1()
        self.calcSigma2()
        self.pitch_ref = pitch_ref

        self.K_deadbeat = np.array([1, self.T_DSP + 1 / (self.lmbd * np.tanh(self.T_SSP * self.lmbd))])

        self.cur_stf = True
        self.cur_swf = not self.cur_stf

        self.t_phase_start = -2 * self.T_SSP

        self.v_ref = v_ref
        self.v_ref_goal = v_ref

        self.pos_swf_imp = 0
        self.v_swf_tof = vswf_tof
        self.v_swf_imp = vswf_imp
        self.z_swf_max = zswf_max
        self.t_swf_max_height = pswf_max

        self.adamKin = Kinematics(urdf_path, mesh_path, use_static_com)

        self.vel_max_change = v_max_change
        self.pos_z_max_change = z_max_change

        self.swf_x_bez = Bezier(x_bez)

        x_swf_pos_z = np.array([0, self.T_SSP, self.t_swf_max_height * self.T_SSP, 0, self.T_SSP])
        y_swf_pos_z = np.array([0, self.pos_swf_imp, self.z_swf_max, self.v_swf_tof, self.v_swf_imp])
        d_swf_pos_z = np.array([0, 0, 0, 1, 1])
        self.swf_pos_z_poly = Poly(x_swf_pos_z, y_swf_pos_z, d_swf_pos_z)

        self.logger = Logger(
            log_path,
            "t,tphase,tscaled,x,z,pitch,q1,q2,q3,q4,xdot,zdot,pitchdot,q1dot,q2dot,q3dot,q4dot,xkin,zkin,pitchkin,q1kin,q2kin,q3kin,q4kin,vrefgoal,vref,zrefgoal,zref,x_ssp_curr,v_ssp_curr,x_ssp_impact,v_ssp_impact,x_ssp_impact_ref,v_ssp_impact_ref,unom,u,bht,pitchref,swfx,swfz,comx,comz,xref,z_ref,pitchref,q1ref,q2ref,q3ref,q4ref\n"
        )

    def calcPreImpactStateRef(self, v_ref:float) -> np.ndarray:
        sigma_1 = self.lmbd / np.tanh(self.T_SSP * self.lmbd / 2)
        p_pre_ref = -v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * sigma_1)
        v_pre_ref = sigma_1 * v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * sigma_1)
        return np.array([p_pre_ref, v_pre_ref])
    
    def calcSSPStateRef(self, x0:np.ndarray, t:float) -> np.ndarray:
        V = np.array([[1, 1], [self.lmbd, -self.lmbd]])
        S = np.array([[np.exp(self.lmbd * t), 0], [0, np.exp(-self.lmbd * t)]])

        return V @ S @ np.linalg.inv(V) @ x0


    def getU(self) -> np.ndarray:
        return self.u
    
    def calcLambda(self) -> None:
        self.lmbd = np.sqrt(self.g / self.z_ref)
    
    def calcSigma1(self) -> None:
        self.sigma1 = self.lmbd / np.tanh(self.T_SSP * self.lmbd / 2)
    
    def calcSigma2(self) -> None:
        self.sigma2 = self.lmbd * np.tanh(self.T_SSP * self.lmbd / 2)
    
    def calcD2(lmbd:float, T_SSP:float, T_DSP:float, v_ref:float) -> float:
        return (lmbd * lmbd / np.cosh(lmbd * T_SSP / 2) * (T_SSP + T_DSP) * v_ref) / (lmbd * lmbd * T_DSP + 2 * HLIPController.calcSigma2(lmbd, T_SSP))
    
    def setT_SSP(self, T_SSP:float) -> None:
        self.T_SSP = T_SSP

    def setV_ref(self, v_ref):
        self.v_ref_goal = v_ref
    
    def setZ_ref(self, z_ref):
        self.z_ref_goal = z_ref

    def setPitchRef(self, pitch_ref):
        self.pitch_ref = pitch_ref

    def gaitController(self, q:np.ndarray, q_pos_ctrl:np.ndarray, q_vel_ctrl:np.ndarray, t:float) -> tuple:
        t_phase = t - self.t_phase_start

        t_scaled = t_phase / self.T_SSP

        y_out = self.adamKin.calcOutputs(q_pos_ctrl, self.cur_stf)
        swf_height = y_out[Kinematics.OUT_ID["SWF_POS_Z"]]

        if t_scaled >= 1 or (t_scaled > 0.5 and swf_height < 0.001):
            t_scaled = 0
            t_phase = 0
            self.t_phase_start = t

            self.cur_stf = not self.cur_stf
            self.cur_swf = not self.cur_swf

            # Recompute outputs with relabeled stance/swing feet
            y_out = self.adamKin.calcOutputs(q_pos_ctrl, self.cur_stf)

            delta_v_ref = self.v_ref_goal - self.v_ref
            if abs(delta_v_ref) > self.vel_max_change:
                self.v_ref += np.sign(delta_v_ref) * self.vel_max_change
            else:
                self.v_ref = self.v_ref_goal

            
            delta_z_ref = self.z_ref_goal - self.z_ref
            if abs(delta_z_ref) > self.pos_z_max_change:
                self.z_ref += np.sign(delta_z_ref) * self.pos_z_max_change
            else:
                self.z_ref = self.z_ref_goal

            self.calcLambda()
            self.calcSigma1()
            self.calcSigma2()

        # X-Dynamics
        self.u_nom = self.v_ref * (self.T_SSP + self.T_DSP)
        x_ssp_impact_ref = np.array([
            self.v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * self.sigma1),
            self.sigma1 * self.v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * self.sigma1)
        ])

        v_com = self.adamKin.getVCom(q_pos_ctrl, q_vel_ctrl)
        # x_ssp_curr = np.array([
        #     y_out[Kinematics.OUT_ID["COM_POS_X"]],
        #     v_com[0]
        # ])

        # Way this is done on other one
        x_ssp_curr = np.array([
            y_out[Kinematics.OUT_ID["COM_POS_X"]],
            q_vel_ctrl[Kinematics.GEN_VEL_ID["V_X"]]
        ])

        x_ssp_impact = self.calcSSPStateRef(x_ssp_curr, self.T_SSP - t_phase)

        self.u = self.u_nom + self.K_deadbeat @ (x_ssp_impact - x_ssp_impact_ref)

        swf_pos_x_curr = y_out[Kinematics.OUT_ID["SWF_POS_X"]]
        bht = self.swf_x_bez.eval(t_scaled)
        # dbht = self.swf_x_bez.deval(t_scaled)

        swf_pos_x_ref = swf_pos_x_curr * (1 - bht) + self.u * bht

        # Z-pos
        swf_pos_z_ref = self.swf_pos_z_poly.evalPoly(t_phase, 0)

        y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
        y_out_ref[Kinematics.OUT_ID["PITCH"]] = self.pitch_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = swf_pos_x_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = swf_pos_z_ref
        y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = y_out[Kinematics.OUT_ID["COM_POS_X"]] # No control authority over x position
        y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = self.z_ref

        q_gen_ref, sol_found = self.adamKin.solveIK(q, y_out_ref, self.cur_stf)
        q_ref = q_gen_ref[-4:]
        # Set desired joint velocities/torques
        qd_ref = np.zeros((Kinematics.N_JOINTS,))
        q_ff_ref = np.zeros((Kinematics.N_JOINTS,))

        self.logger.write(np.hstack((
            t, t_phase, t_scaled, q_pos_ctrl, q_vel_ctrl, q, self.v_ref_goal, self.v_ref, self.z_ref_goal, self.z_ref, 
            x_ssp_curr, x_ssp_impact, x_ssp_impact_ref, self.u_nom, self.u, bht, y_out, q_gen_ref
        )))

        return q_ref, qd_ref, q_ff_ref