import numpy as np
import scipy.sparse
import osqp
from kinematics_py.adam_kinematics import Kinematics
from control_py.bezier import Bezier
from control_py.poly import Poly
from plot.logger import Logger

class HLIPController:

    def __init__(
            self, T_SSP:float, z_ref:float, urdf_path:str, mesh_path:str, mass:float, grav_comp:bool=True, angMomState:bool=False, use_task_space_ctrl:bool=False,
            v_ref:float=0, pitch_ref:float=0.025, v_max_change:float=0.1, z_max_change:float=0.02, x_bez:np.ndarray=np.array([0,0,0,1,1]),
            vswf_tof:float=0.05, vswf_imp:float=-0.05, zswf_max:float=0.075, pswf_max:float=0.7,
            use_static_com:bool=False, T_DSP:float=0, mu_friction:float=1, tau_max:float=50, log_path:str="plot/log_ctrl.csv"
        ):
        self.T_SSP = T_SSP
        self.T_DSP = T_DSP
        self.T_SSP_goal = T_SSP
        self.g = 9.81
        self.gravity_comp = grav_comp
        self.mass = mass
        self.angMomState = angMomState

        self.z_ref_goal = z_ref
        self.z_ref = z_ref
        self.calcLambda()
        self.calcSigma1()
        self.calcSigma2()
        self.pitch_ref = pitch_ref
        self.use_task_space_ctrl = use_task_space_ctrl
        self.mu_friction = mu_friction
        self.tau_max = tau_max

        self.computeGain()
        self.K_L = np.array([0.9999, 0.0305])

        self.Kp = 100 * np.eye(4)
        self.Kd = 2 * np.sqrt(self.Kp) * np.eye(4)

        self.cur_stf = False
        self.cur_swf = not self.cur_stf

        self.swf_x_start = 0

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
        z_bez = np.array([0, 0.25 * self.z_swf_max, 0.5 * self.z_swf_max, self.z_swf_max, 0])
        self.swf_pos_z_bez = Bezier(z_bez)

        self.logger = Logger(
            log_path,
            "t,tphase,tscaled,x,z,pitch,q1,q2,q3,q4,xdot,zdot,pitchdot,q1dot,q2dot,q3dot,q4dot,xkin,zkin,pitchkin,q1kin,q2kin,q3kin,q4kin,vrefgoal,vref,zrefgoal,zref,x_ssp_curr,v_ssp_curr,x_ssp_impact,v_ssp_impact,x_ssp_impact_ref,v_ssp_impact_ref,unom,u,bht,ypitch,yswfx,yswfz,ycomx,ycomz,ypitchref,yswfxref,yswfzref,ycomxref,ycomzref,xref,z_ref,pitchref,q1ref,q2ref,q3ref,q4ref,tau1,tau2,tau3,tau4,vcom,vstatic,vbody,impact,stf_ang_mom_pin,x_ssp_curr_L,L_ssp_curr_L,x_ssp_impact_L,L_ssp_impact_L,x_ssp_impact_ref_L,L_ssp_impact_ref_L,tau1_gc,tau2_gc,tau3_gc,tau4_gc,tau1_tsc,tau2_tsc,tau3_tsc,tau4_tsc,ddx,ddz,ddtheta,ddq1,ddq2,ddq3,ddq4,grfx,grfz,deltaepitch,deltaeswfx,deltaeswfz,deltaecomz,obj_val,m11,m12,m13,m14,m15,m16,m17,m21,m22,m23,m24,m25,m26,m27,m31,m32,m33,m34,m35,m36,m37,m41,m42,m43,m44,m45,m46,m47,m51,m52,m53,m54,m55,m56,m57,m61,m62,m63,m64,m65,m66,m67,m71,m72,m73,m74,m75,m76,m77,h1,h2,h3,h4,h5,h6,h7,Jh11,Jh12,Jh13,Jh14,Jh15,Jh16,Jh17,Jh21,Jh22,Jh23,Jh24,Jh25,Jh26,Jh27\n"
        )

    def calcPreImpactStateRef_HLIP(self, v_ref:float) -> np.ndarray:
        sigma_1 = self.lmbd / np.tanh(self.T_SSP * self.lmbd / 2)
        p_pre_ref = -v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * sigma_1)
        v_pre_ref = sigma_1 * v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * sigma_1)
        return np.array([p_pre_ref, v_pre_ref])
    
    def calcSSPStateRef_HLIP(self, x0:np.ndarray, t:float) -> np.ndarray:
        V = np.array([[1, 1], [self.lmbd, -self.lmbd]])
        S = np.array([[np.exp(self.lmbd * t), 0], [0, np.exp(-self.lmbd * t)]])

        return V @ S @ np.linalg.inv(V) @ x0
    
    def calcSSPStateRef_LLIP(self, x_L0:np.ndarray, t:float) -> np.ndarray:
        return np.array([
            [np.cosh(self.lmbd * t), 1 / (self.mass * self.z_ref * self.lmbd) * np.sinh(self.lmbd * t)],
            [self.mass * self.z_ref * self.lmbd * np.sinh(self.lmbd * t), np.cosh(self.lmbd * t)]
        ]) @ x_L0


    def getU(self) -> np.ndarray:
        return self.u
    
    def calcLambda(self) -> None:
        self.lmbd = np.sqrt(self.g / self.z_ref)

    def computeGain(self) -> None:
        self.K_deadbeat = np.array([1, self.T_DSP + 1 / (self.lmbd * np.tanh(self.T_SSP * self.lmbd))])
    
    def calcSigma1(self) -> None:
        self.sigma1 = self.lmbd / np.tanh(self.T_SSP * self.lmbd / 2)
    
    def calcSigma2(self) -> None:
        self.sigma2 = self.lmbd * np.tanh(self.T_SSP * self.lmbd / 2)
    
    def calcD2(self, lmbd:float, T_SSP:float, T_DSP:float, v_ref:float) -> float:
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
            impact = True
        # if t_scaled > 1.5 or (t_scaled > 0.5 and ((self.cur_stf and right_cont) or (not self.cur_stf and left_cont))):
            # print("impact")
            # print(f"SWF Z at 'contact' {swf_height}, t_scaled at 'contact' {t_scaled}")
            t_scaled = 0
            t_phase = 0
            self.t_phase_start = t

            self.cur_stf = not self.cur_stf
            self.cur_swf = not self.cur_swf

            # Recompute outputs with relabeled stance/swing feet
            y_out = self.adamKin.calcOutputs(q_pos_ctrl, self.cur_stf)
            self.swf_x_start = y_out[Kinematics.OUT_ID["SWF_POS_X"]]

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
        else:
            impact = False


        ang_mom = self.adamKin.getSTFAngularMomentum(q_pos_ctrl, q_vel_ctrl, self.cur_stf)

        # X-Dynamics
        self.u_nom = self.v_ref * (self.T_SSP + self.T_DSP)
        x_ssp_impact_ref = np.array([
            self.v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * self.sigma1),
            self.sigma1 * self.v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * self.sigma1)
        ])

        x_ssp_impact_ref_L = np.array([
            self.v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * self.sigma1),
            (self.sigma1 * self.v_ref * (self.T_SSP + self.T_DSP) / (2 + self.T_DSP * self.sigma1)) * self.mass * self.z_ref
        ])


        # v_com = self.adamKin.v_CoM(q_pos_ctrl, q_vel_ctrl)
        # v_static = self.adamKin.v_StaticCom(q_pos_ctrl, q_vel_ctrl)

        v_com_use = self.adamKin.getVCom(q_pos_ctrl, q_vel_ctrl)
        x_ssp_curr = np.array([
            y_out[Kinematics.OUT_ID["COM_POS_X"]],
            v_com_use[0]
        ])

        x_ssp_curr_L = np.array([
            y_out[Kinematics.OUT_ID["COM_POS_X"]],
            ang_mom
        ])

        x_ssp_impact = self.calcSSPStateRef_HLIP(x_ssp_curr, self.T_SSP - t_phase)
        x_ssp_impact_L = self.calcSSPStateRef_LLIP(x_ssp_curr_L, self.T_SSP - t_phase)

        if self.angMomState:
            self.u = self.u_nom + self.K_L @ (x_ssp_impact_L - x_ssp_impact_ref_L)
        else:
            self.u = self.u_nom + self.K_deadbeat @ (x_ssp_impact - x_ssp_impact_ref)

        if self.u > 0.5:
            print(f"Large Step {self.u} Requested")

        swf_pos_x_curr = y_out[Kinematics.OUT_ID["SWF_POS_X"]]
        bht = self.swf_x_bez.eval(t_scaled)

        # Old Method, relative to current swing foot position
        # swf_pos_x_ref = swf_pos_x_curr * (1 - bht) + self.u * bht
        # New method, relative to swing foot position at beginning of stride
        swf_pos_x_ref = self.swf_x_start * (1 - bht) + self.u * bht

        # Z-pos
        swf_pos_z_ref = self.swf_pos_z_poly.evalPoly(t_phase, 0)
        # swf_pos_z_ref = self.swf_pos_z_bez.eval(t_scaled)

        y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
        y_out_ref[Kinematics.OUT_ID["PITCH"]] = self.pitch_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = swf_pos_x_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = swf_pos_z_ref
        y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = y_out[Kinematics.OUT_ID["COM_POS_X"]] # No control authority over x position
        y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = self.z_ref

        q_gen_ref, sol_found = self.adamKin.solveIK(q, y_out_ref, self.cur_stf)

        if not sol_found:
            print('No solution found for IK', y_out_ref)

        q_ref = q_gen_ref[-4:]
        # Set desired joint velocities/torques
        qd_ref = np.zeros((Kinematics.N_JOINTS,))

        # # Compute Task space controller
        # # Compute ya, yd, e
        # ya = y_out                                          # actual output
        # yd = y_out_ref                                      # desired output
        # e = np.hstack((ya[0:3] - yd[0:3], ya[4] - yd[4]))   # output error
        # # Compute dya, dyd, de
        # dbht = self.swf_x_bez.deval(t_scaled) * self.T_SSP                                # time derivative of swf x bezier polynomial
        # dpoly = self.swf_pos_z_poly.evalPoly(t_phase, 1)                                  # time deriviative of swf z polynomial
        # # dpoly = self.swf_pos_z_bez.deval(t_scaled) * self.T_SSP                           # time deriviative of swf z polynomial
        # Jcomz = self.adamKin.getCoMJacobian(q_pos_ctrl)[2, :]                                                # Jcom  center of mass jacobian
        # dJcomz = self.adamKin.getCoMJacobianTimeDerivative(q_pos_ctrl, q_vel_ctrl)[2, :]                     # dot Jcom, time variation of com jacobian
        # Jswf = self.adamKin.getSWFJacobian(q_pos_ctrl, self.cur_stf)[0:3:2, :]                               # Jswf, swing foot jacobian
        # Jswfx = Jswf[0, :]
        # Jswfz = Jswf[1, :]
        # dJswf = self.adamKin.getSWFJacobianTimeDerivative(q_pos_ctrl, q_vel_ctrl, self.cur_stf)[0:3:2, :]    # dot Jswf, swf time variation of jacobian
        # dJswfx = dJswf[0, :]
        # dJswfz = dJswf[1, :]
        # ddbht = self.swf_x_bez.ddeval(t_scaled) * self.T_SSP * self.T_SSP                                  # ddot swf x bezier curve
        # ddpoly = self.swf_pos_z_poly.evalPoly(t_phase, 2)                                                  # ddot swf z polynomial
        # # ddpoly = self.swf_pos_z_bez.ddeval(t_scaled) * self.T_SSP * self.T_SSP                             # ddot swf z polynomial

        # # Compute de
        # de = np.array([
        #     q_vel_ctrl[2],
        #     Jswfx @ q_vel_ctrl - (self.u - self.swf_x_start) * dbht,
        #     Jswfz @ q_vel_ctrl - dpoly,
        #     Jcomz @ q_vel_ctrl
        # ])
        # dded = -self.Kp @ e - self.Kd @ de                                                                      # Desired ddot error is ddoot e_d = -Kp e -Kd dot e
        
        # # Construct cost function ( dde = R @ ddq + g )
        # R = np.vstack((np.array([0, 0, 1, 0, 0, 0, 0]), Jswfx, Jswfz, Jcomz))
        # g = np.array([
        #     0,
        #     dJswfx @ q_vel_ctrl - (self.u - self.swf_x_start) * ddbht,
        #     dJswfz @ q_vel_ctrl - ddpoly,
        #     dJcomz @ q_vel_ctrl
        # ])
        # P1 = R.T @ R
        # Q1 = R.T @ (g - dded)
        # P = scipy.sparse.block_diag([P1, np.zeros((6, 6))])
        # Q = np.hstack((Q1, np.zeros((6,))))

        # # Construct contraints
        # M, H, B, Jh, dJh = self.adamKin.getDynamics(q_pos_ctrl, q_vel_ctrl, self.cur_stf)                       # Get the dynamics 

        # # Dynamics Constraint
        # A1 = np.hstack((M, -Jh.T, -B))
        # UB1 = -H
        # LB1 = -H
        # # Friction cone constraint (lambda_z >= 0, |lambda_x| <= mu lambda_z)
        # A2 = np.hstack((
        #     np.zeros((3, 7)), 
        #     np.array([
        #         [0, 1],
        #         [1, -self.mu_friction],
        #         [-1, -self.mu_friction]
        #     ]),
        #     np.zeros((3, 4))
        # ))
        # UB2 = np.array([np.inf, 0, 0])
        # LB2 = np.array([0, -np.inf, -np.inf])
        # # Input constraints
        # A3 = np.hstack((np.zeros((4, 7)), np.zeros((4, 2)), np.eye(4)))
        # UB3 = self.tau_max * np.ones((4,))
        # LB3 = -self.tau_max * np.ones((4,))
        # # Holonomic constraint
        # A4 = np.hstack((Jh, np.zeros((2, 2)), np.zeros((2, 4))))
        # UB4 = -dJh @ q_vel_ctrl
        # LB4 = -dJh @ q_vel_ctrl
        # A = scipy.sparse.csc_matrix(np.vstack((A1, A2, A3, A4)))
        # UB = np.hstack((UB1, UB2, UB3, UB4))
        # LB = np.hstack((LB1, LB2, LB3, LB4))
        
        # # Construct QP
        # tsc_qp = osqp.OSQP()
        # tsc_qp.setup(P=P, q=Q, A=A, u=UB, l=LB, eps_rel=1e-6, verbose=False)

        # # Solve QP
        # res = tsc_qp.solve()

        # ddq = res.x[:7]
        # tau = res.x[-4:]
        # grf = res.x[7:-4]

        q_ff_ref_gravcomp = self.adamKin.calcGravityCompensation(q_pos_ctrl, self.cur_stf)
        q_ff_ref_pd = np.zeros((Kinematics.N_JOINTS,))
        
        # dde = R @ ddq + g

        # # Check if solution is correct
        # obj = 0.5*np.inner(dde - dded, dde - dded)
        # sol_obj = res.info.obj_val
        # sol_obj_expected = 0.5 * ddq.T @ R.T @ R @ ddq + (g - dded).T @ R @ ddq
        # obj_expected = sol_obj_expected + 0.5 * np.inner(dded - g, dded - g)

        # if self.use_task_space_ctrl:
        #     q_ff_ref = tau
        #     q_ref = q_pos_ctrl[-4:]
        #     qd_ref = q_vel_ctrl[-4:]
        # el
        if self.gravity_comp:
            q_ff_ref = q_ff_ref_gravcomp
        else:
            q_ff_ref = q_ff_ref_pd

        # self.logger.write(np.hstack((
        #     t, t_phase, t_scaled, q_pos_ctrl, q_vel_ctrl, q, self.v_ref_goal, self.v_ref, self.z_ref_goal, self.z_ref, 
        #     x_ssp_curr, x_ssp_impact, x_ssp_impact_ref, self.u_nom, self.u, bht, y_out, y_out_ref, q_gen_ref, q_ff_ref, v_com[0], v_static[0], q_vel_ctrl[Kinematics.GEN_VEL_ID["V_X"]],
        #     impact, ang_mom, x_ssp_curr_L, x_ssp_impact_L, x_ssp_impact_ref_L, q_ff_ref_gravcomp, tau, ddq, grf, dde - dded, 0.5*np.inner(dde - dded, dde - dded), M.reshape((-1,)), H, Jh.reshape((-1))
        # )))

        # dyn_err = np.linalg.norm(M @ ddq + H - (B @ tau + Jh.T @ grf))
        # if dyn_err > 3e-4:
        #     print("uhoh", dyn_err)

        return q_ref, qd_ref, q_ff_ref
    
    def reset(self):
        self.cur_stf = False
        self.cur_swf = not self.cur_stf

        self.swf_x_start = 0

        self.t_phase_start = -2 * self.T_SSP
