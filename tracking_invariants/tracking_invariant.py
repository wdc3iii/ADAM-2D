import yaml
from tqdm import trange
import numpy as np
import time
from plot.logger import Logger
from kinematics_py.adam_kinematics import Kinematics
from control_py.hlip_controller import HLIPController
from simulation_py.mujoco_interface import MujocoInterface
from scipy.linalg import sqrtm

mesh_path = "rsc/models/"
log_path = "plot/log_tracking_invariant.csv"
urdf_path = "rsc/models/adam2d.urdf"
xml_path = "rsc/models/adam2d.xml"

class TrackingInvariant:
    
    def __init__(self, v_ref, z_ref, pitch_ref, T_SSP, approximationMethod, Nsamples, Nextreme=1, groupingSpace="Output", useAngMomState=False, use_static_com=False, gravity_comp=True, use_task_space_ctrl=False):
        self.v_ref = v_ref
        self.z_ref = z_ref
        self.T_SSP = T_SSP
        self.pitch_ref = pitch_ref
        self.approxMethod = approximationMethod
        if self.approxMethod not in ["InftyNorm", "Ellipsoid", "Polytope", "Extreme Points"]:
            raise ValueError(f"Set Approximation Method '{self.approxMethod}' not allowed")
        self.Nsamples = Nsamples
        self.groupingSpace = groupingSpace
        if self.groupingSpace not in ["Output", "State"]:
            raise ValueError(f"Grouping Space '{self.groupingSpace}' not allowed")
    
        self.Nextreme = Nextreme

        self.adamKin = Kinematics(urdf_path, mesh_path)
        # self.mjInt = MujocoInterface(xml_path, vis_enabled=False)
        self.mjInt = MujocoInterface(xml_path, vis_enabled=True)

        self.controller = HLIPController(
            T_SSP, z_ref, urdf_path, mesh_path, self.mjInt.mass, angMomState=useAngMomState,
            v_ref=v_ref, pitch_ref=pitch_ref, use_static_com=use_static_com, grav_comp=gravity_comp,
            use_task_space_ctrl=use_task_space_ctrl
        )

        self.q_ik = self.adamKin.getZeroPos()
        self.q_ik[Kinematics.GEN_POS_ID["P_LHP"]] = -0.4
        self.q_ik[Kinematics.GEN_POS_ID["P_RHP"]] = -0.4
        self.q_ik[Kinematics.GEN_POS_ID["P_LKP"]] = 0.8
        self.q_ik[Kinematics.GEN_POS_ID["P_RKP"]] = 0.8

        self.x_pre_ref = self.controller.calcPreImpactStateRef_HLIP(self.v_ref)
        self.u_ref = self.v_ref * self.T_SSP

        self.reachableList = np.hstack(self.getNominalState()).reshape((1, -1))
        self.reachableTable = {0: np.copy(self.reachableList)}
        self.setDiscriptions = {0: {"Method": 'Extreme Points', "Data": np.copy(self.reachableList)}}
        self.iteration = 0
        self.propInSet = 0

        self.logger = Logger(log_path,
            "iter,x0,z0,p0,q10,q20,q30,q40,xd0,zd0,pd0,qd10,qd20,qd30,qd40,p0,sx0,sz0,cx0,cz0,pd0,sxd0,szd0,cxd0,czd0,xF,zF,pF,q1F,q2F,q3F,q4F,xdF,zdF,pdF,qd1F,qd2F,qd3F,qd4F,pF,sxF,szF,cxF,czF,pdF,sxdF,szdF,cxdF,czdF\n"
        )

    def S2S_sim(self, q0, qd0) -> tuple:
        self.mjInt.setState(q0, qd0)
        self.mjInt.forward()
        startTime = self.mjInt.time()
        self.controller.reset()

        while True:
            self.mjInt.updateScene()
            t = self.mjInt.time() - startTime
            qpos = self.mjInt.getGenPosition()
            qvel = self.mjInt.getGenVelocity()   
            q_pos_ref, q_vel_ref, q_ff_ref = self.controller.gaitController(qpos, qpos, qvel, t)

            self.mjInt.jointPosCmd(q_pos_ref)
            self.mjInt.jointVelCmd(q_vel_ref)
            self.mjInt.jointTorCmd(q_ff_ref)

            if not self.controller.cur_stf:
                break

            self.mjInt.step()
        
        qCpos = self.adamKin.calcOutputs(qpos, True)
        qCvel = self.adamKin.calcDOutputs(qpos, qvel, True)

        return qpos, qvel, qCpos, qCvel
    
    def getNominalState(self):
        

        y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
        y_out_ref[Kinematics.OUT_ID["PITCH"]] = self.pitch_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = self.u_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = 0
        y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = self.x_pre_ref[0]
        y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = self.z_ref

        qpos, _ = self.adamKin.solveIK(self.q_ik, y_out_ref, True)
        qpos = self.modifyPoseForContact(qpos, True)
        
        yd_out_ref = np.array([
            0, 0, self.controller.swf_pos_z_poly.evalPoly(self.T_SSP, 1), self.x_pre_ref[1], 0, 0, 0
        ])
        qvel = self.adamKin.solveIKVel(qpos, yd_out_ref, True)

        qCpos = y_out_ref
        qCvel = yd_out_ref[:-2]

        return qpos, qvel, qCpos, qCvel
    
    def modifyPoseForContact(self, qpos, stanceFoot):
        self.mjInt.setState(qpos, np.zeros_like(qpos))
        self.mjInt.forward()
        ft_pos = self.mjInt.getFootPos()
        qpos[Kinematics.GEN_POS_ID['P_Z']] -= ft_pos[1][1] - self.adamKin.getContactOffset(qpos, stanceFoot)
        return qpos
    
    def iterateSetMap(self, verbose=False):

        # Sample points from the current convex outerapproximation
        if verbose:
            print("..... Sampling Set .....")

        points = self.sampleSet()
        propogatedPoints = np.zeros_like(points)

        if verbose:
            print("..... Computing S2S .....")

        # Propogate points through the S2S dynamics
        func = trange if verbose else range
        for ind in func(points.shape[0]):
            q0 = points[ind, :7]
            qd0 = points[ind, 7:14]

            qF, qdF, qCF, qdCF = self.S2S_sim(q0, qd0)
            propogatedPoints[ind, :7] = qF
            propogatedPoints[ind, 7:14] = qdF
            propogatedPoints[ind, 14:19] = qCF
            propogatedPoints[ind, 19:] = qdCF

            self.logger.write(np.hstack((self.iteration, points[ind, :], propogatedPoints[ind, :])))

        self.reachableList = np.vstack((self.reachableList, propogatedPoints))
        self.reachableTable[self.iteration + 1] = propogatedPoints
        
        # Fit a new convex outerapproximation
        if verbose:
            print("..... Fitting Set .....")
        self.fitSet()

        self.iteration += 1
        self.propInSet = self._pointsInSet()
        if verbose:
            self.verboseOut()

    def sampleSet(self):
        # Handle Initial iteration separately
        if self.iteration == 0:
            return self.reachableList
        # Samples a set of points from the most recent convex set
        setMethod = self.setDiscriptions[self.iteration]["Method"]
        setData = self.setDiscriptions[self.iteration]["Data"]
        if setMethod == "Extreme Points":
            setData = setData[:, :14] if self.groupingSpace == "State" else setData[:, 14:]
            numPts = np.random.randint(1, min(11, setData.shape[0]) + 1, size=(self.Nsamples,))
            sampledPoints = np.zeros((self.Nsamples, setData.shape[1]))
            for ii, nPts in enumerate(numPts):
                inds = np.random.choice(range(setData.shape[0]), size=(nPts,), replace=False)
                if nPts == 1:
                    sampledPoints[ii, :] = setData[inds, :]
                else:
                    subset = setData[inds, :]
                    lmbd = np.random.uniform(0, 1, (1, subset.shape[0]))
                    lmbd /= np.sum(lmbd, axis=1)
                    sampledPoints[ii, :] = lmbd @ subset
        elif setMethod == "Polytope":
            raise ValueError("Polytope sampling not implemented yet")
        elif setMethod == "Ellipsoid":
            '''
            uniformly sample a N-dimensional unit UnitBall
            Reference:
            Efficiently sampling vectors and coordinates from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
            Input:
                num - no. of samples
                dim - dimensions
            Output:
                uniformly sampled points within N-dimensional unit ball
            '''
            #Sample on a unit N+1 sphere
            d = 10 if self.groupingSpace == "Output" else 14
            u = np.random.normal(0, 1, (self.Nsamples, d + 2))
            norm = np.linalg.norm(u, axis=-1,keepdims=True)
            u = u/norm
            #The first N coordinates are uniform in a unit N ball
            sampledPoints = u[:,:d]
            sampledPoints = np.linalg.inv(sqrtm(setData["A"])) @ sampledPoints + setData["c"]
        elif setMethod == "InftyNorm":
            sampledPoints = np.random.uniform(setData["LB"], setData["UB"], (self.Nsamples, setData["LB"].shape[0]))
        
        # Complete the points as necessary with either inverse kinematics (constructing state) or forward kinematics (completing outputs)
        if self.groupingSpace == "Output":
            sampledPoints = np.hstack((np.zeros((sampledPoints.shape[0], 14)), sampledPoints))
            for ii in range(sampledPoints.shape[0]):
                qpos, _ = self.adamKin.solveIK(self.q_ik, sampledPoints[ii, 14:19], True)
                sampledPoints[ii, :7] = self.modifyPoseForContact(qpos, True)
                sampledPoints[ii, 7:14] = self.adamKin.solveIKVel(sampledPoints[ii, :7], np.hstack((sampledPoints[ii, 19:], np.zeros((2,)))), True)
        elif self.groupingSpace == "State":
            sampledPoints = np.hstack((sampledPoints, np.zeros((sampledPoints.shape[0], 10))))
            for ii in range(sampledPoints.shape[0]):
                sampledPoints[ii, 14:19] = self.adamKin.calcOutputs(sampledPoints[ii, :7], True)
                sampledPoints[ii, 19:] =  self.adamKin.calcDOutputs(sampledPoints[ii, :7], sampledPoints[ii, 7:14], True)

        return sampledPoints
    
    def sampleSetSurface(self):
        # Handle Initial iteration separately
        if self.iteration == 0:
            return self.reachableList
        # Samples a set of points from the most recent convex set
        setMethod = self.setDiscriptions[self.iteration]["Method"]
        setData = self.setDiscriptions[self.iteration]["Data"]
        if setMethod == "Extreme Points":
            raise ValueError("Extreme Points surface sampling not implemented yet")
        elif setMethod == "Polytope":
            raise ValueError("Polytope surface sampling not implemented yet")
        elif setMethod == "Ellipsoid":
            '''
            uniformly sample a N-dimensional unit UnitBall
            Reference:
            Efficiently sampling vectors and coordinates from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
            Input:
                num - no. of samples
                dim - dimensions
            Output:
                uniformly sampled points within N-dimensional unit ball
            '''
            #Sample on a unit N+1 sphere
            d = 10 if self.groupingSpace == "Output" else 14
            u = np.random.normal(0, 1, (self.Nsamples, d))
            norm = np.linalg.norm(u, axis=-1,keepdims=True)
            sampledPoints = u / norm
            sampledPoints = np.linalg.inv(sqrtm(setData["A"])) @ sampledPoints + setData["c"]
        elif setMethod == "InftyNorm":
            sampledPoints = np.random.uniform(setData["LB"], setData["UB"], self.Nsamples)
            d = 10 if self.groupingSpace == "Output" else 14
            rowind = np.array(range(self.Nsamples))
            probs = np.product(np.vstack([np.diag([0 if ii == jj else 1 for ii in range(d)]) @ (setData["UB"] - setData["LB"]) for jj in range(d)]) + np.eye(d))
            colind = np.random.choice(range(d), size=self.Nsamples, p=probs / np.sum(probs))
            high_low = np.random.randint(0, 2, self.Nsamples)
            sampledPoints[rowind, colind] = high_low * setData["UB"][rowind, colind] + (1 - high_low) * setData["UB"][rowind, colind]

        
        # Complete the points as necessary with either inverse kinematics (constructing state) or forward kinematics (completing outputs)
        if self.groupingSpace == "Output":
            sampledPoints = np.hstack((np.zeros((sampledPoints.shape[0], 14), sampledPoints)))
            for ii in range(sampledPoints.shape[0]):
                sampledPoints[ii, :7] = self.adamKin.solveIK(sampledPoints[ii, 14:19], True)
                sampledPoints[ii, 7:14] = self.adamKin.solveIKVel(sampledPoints[ii, :7], np.hstack((sampledPoints[ii, 19:], np.zeros((2,)))), True)
        elif self.groupingSpace == "State":
            sampledPoints = np.hstack((sampledPoints, np.zeros((sampledPoints.shape[0], 10))))
            for ii in range(sampledPoints.shape[0]):
                sampledPoints[ii, 14:19] = self.adamKin.calcOutputs(sampledPoints[ii, :7], True)
                sampledPoints[ii, 19:] =  self.adamKin.calcDOutputs(sampledPoints[ii, :7], sampledPoints[ii, 7:14], True)

        return sampledPoints
    
    def setSampleSize(self, Nsamples):
        self.Nsamples = Nsamples

    def setApproxMethod(self, method):
        self.approxMethod = method

    def getPropInSet(self):
        return self.propInSet

    def fitSet(self):
        if self.groupingSpace == "State":
            points = self.reachableList[:, :14]
            # May need to remove x variable to account for translational symmetry here
        elif self.groupingSpace == "Output":
            points = self.reachableList[:, 14:]
        # Fits a convex outer approximation to a set of points
        if self.approxMethod == "InftyNorm":
            desc = {}
            desc["UB"] = np.max(points, axis=0)
            desc["LB"] = np.min(points, axis=0)
        elif self.approxMethod == "Ellipse":
            """
            Finds the ellipse equation in "center form"
            (x-c).T * A * (x-c) = 1
            """
            N, d = points.shape
            Q = np.column_stack((points, np.ones(N))).T
            tol = 1e-3
            err = tol+1.0
            u = np.ones(N)/N
            while err > tol:
                # assert u.sum() == 1 # invariant
                X = np.dot(np.dot(Q, np.diag(u)), Q.T)
                M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
                jdx = np.argmax(M)
                step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
                new_u = (1-step_size)*u
                new_u[jdx] += step_size
                err = np.linalg.norm(new_u-u)
                u = new_u
            c = np.dot(u,points)
            A = np.linalg.inv(np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(c,c))/d
            desc = {"c": c, "A": A}
        elif self.approxMethod == "Polytope":
            raise ValueError("Polytope not yet implemented")
        elif self.approxMethod == "Extreme Points":
            desc = self.reachableList
        
        self.setDiscriptions[self.iteration + 1] = {"Method": self.approxMethod, "Data": desc}

    def _pointsInSet(self, iteration=None):
        if iteration is None:
            iteration = self.iteration
        if iteration == 0:
            return 0
        if self.groupingSpace == "Output":
            points = self.reachableTable[iteration][:, 14:]
        elif self.groupingSpace == "State":
            points = self.reachableTable[iteration, :14]

        points_in = 0
        for ii in range(points.shape[0]):
            point = points[ii, :]
            points_in += self._pointInSet(point, iteration - 1)

        return points_in / points.shape[0]
    
    def _pointInSet(self, point, iteration):
        setMethod = self.setDiscriptions[iteration]["Method"]
        setData = self.setDiscriptions[iteration]["Data"]
        if setMethod == "Extreme Points":
            # raise ValueError("Polytope sampling not implemented yet")
            print("Extreme point in set not implemented yet\n")
            return 0
        elif setMethod == "Polytope":
            raise ValueError("Polytope point in set not implemented yet")
        elif setMethod == "Ellipsoid":
            '''
            uniformly sample a N-dimensional unit UnitBall
            Reference:
            Efficiently sampling vectors and coordinates from the n-sphere and n-ball
            http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
            Input:
                num - no. of samples
                dim - dimensions
            Output:
                uniformly sampled points within N-dimensional unit ball
            '''
            #Sample on a unit N+1 sphere
            return np.dot(point - setData["c"], setData["A"] @ (point - setData["c"])) <= 1
        elif setMethod == "InftyNorm":
            return np.all(point <= setData["UB"]) and np.all(point >= setData["LB"])
        
    
    def verboseOut(self):
        print(f"Proportion of points in set: {self.propInSet}")
        if self.iteration < 2:
            return
        if self.approxMethod == "InftyNorm":
            dUB = self.setDiscriptions[self.iteration]["Data"]["UB"] - self.setDiscriptions[self.iteration - 1]["Data"]["UB"]
            dLB = self.setDiscriptions[self.iteration]["Data"]["LB"] - self.setDiscriptions[self.iteration - 1]["Data"]["LB"]

            print(f"UB Change: {dUB}\nLB Change: {dLB}")
        
        print(f"\n\nIteration {self.iteration}")
        
        