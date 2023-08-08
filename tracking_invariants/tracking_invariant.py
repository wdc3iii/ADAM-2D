from tqdm import trange
import numpy as np
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
    
    def __init__(
            self, v_ref:float, z_ref:float, pitch_ref:float, T_SSP:float, approxMethod:str, Nsamples:int=1000, NsampleSchedule:function=None,
            approxSpace:str="Output", useAngMomState:bool=False, use_static_com:bool=False, gravity_comp:bool=True, use_task_space_ctrl:bool=False
        ):
        """Initialized a class to compute tracking invariants of the HLIP reduced order model

        Args:
            v_ref (float): reference velocity
            z_ref (float): reference CoM hieght
            pitch_ref (float): pitch angle reference
            T_SSP (float): stepping period
            approxMethod (str): type of convex set to use as outer approximation of propogated samples
            Nsamples (int, optional): Number of samples to use per iteration. (Alternatively specify a schedule). Defaults to 1000.
            NsampleSchedule (function, optional): function taking an iteration number (int) to number of samples to take. Defaults to None.
            approxSpace (str, optional): Space to perform convex approximation in, "Output" or "State". Defaults to "Output".
            useAngMomState (bool, optional): Whether to use angular momentum as second HLIP state. Defaults to False.
            use_static_com (bool, optional): Whether to use a fixed approximation of the CoM. Defaults to False.
            gravity_comp (bool, optional): whether to use gravity compensation in controller. Defaults to True.
            use_task_space_ctrl (bool, optional): whether to use task space controller. Defaults to False.

        Raises:
            ValueError: invalid approximation method
            ValueError: invalid approximation space
        """
        # Check for illegal input arguments
        if approxMethod not in ["InftyNorm", "Ellipsoid", "Polytope", "Extreme Points"]:
            raise ValueError(f"Set Approximation Method '{self.approxMethod}' not allowed")
        if approxSpace not in ["Output", "State"]:
            raise ValueError(f"Grouping Space '{self.approxSpace}' not allowed")
        
        # Save parameters to instance variables
        self.v_ref = v_ref                      # Reference velocity
        self.z_ref = z_ref                      # Reference CoM hieght
        self.T_SSP = T_SSP                      # Step period
        self.pitch_ref = pitch_ref              # Pitch Reference
        
        self.Nsamples = Nsamples                # Number of samples per iteration (fixed)
        self.NsampleSchedule = NsampleSchedule  # Samples per iteration (schedule)
        self.approxMethod = approxMethod        # Type of convex set to use for outer approximation
        self.approxSpace = approxSpace          # Space (output or state) to use for set approximation
        
    
        self.d = 10 if self.approxSpace == "Output" else 14 # Dimension of the approxSpace
        self.minSampleConvex = 2 * self.d                   # Use Extreme Points for first iterations, until this number of samples


        self.adamKin = Kinematics(urdf_path, mesh_path)             # Initialize kinematics solve
        self.mjInt = MujocoInterface(xml_path, vis_enabled=False)   # Initialize simulator
        # self.mjInt = MujocoInterface(xml_path, vis_enabled=True)

        # Initialized closed loop HLIP controller
        self.controller = HLIPController(
            T_SSP, z_ref, urdf_path, mesh_path, self.mjInt.mass, angMomState=useAngMomState,
            v_ref=v_ref, pitch_ref=pitch_ref, use_static_com=use_static_com, grav_comp=gravity_comp,
            use_task_space_ctrl=use_task_space_ctrl
        )

        # Default configuration for initial guess on IK for initial configurations
        self.q_ik = self.adamKin.getZeroPos()
        self.q_ik[Kinematics.GEN_POS_ID["P_LHP"]] = -0.4
        self.q_ik[Kinematics.GEN_POS_ID["P_RHP"]] = -0.4
        self.q_ik[Kinematics.GEN_POS_ID["P_LKP"]] = 0.8
        self.q_ik[Kinematics.GEN_POS_ID["P_RKP"]] = 0.8

        # Compute HLIP reference preimact state and step length
        self.x_pre_ref = self.controller.calcPreImpactStateRef_HLIP(self.v_ref)
        self.u_ref = self.v_ref * self.T_SSP

        # Initialize data structures for computation of tracking invariant
        self.reachableList = np.hstack(self.getNominalState()).reshape((1, -1))                         # List of all points which have been reached
        self.reachableTable = {0: np.copy(self.reachableList)}                                          # Table of points, reached at each iteration
        self.setDiscriptions = {0: {"Method": 'Extreme Points', "Data": np.copy(self.reachableList)}}   # Dictionary of set descriptions after each iteration
        # Counter variables
        self.iteration = 0
        self.propInSet = 0

        # Initialize a logger to track S2S dynamics 
        self.logger = Logger(log_path,
            "iter,x0,z0,p0,q10,q20,q30,q40,xd0,zd0,pd0,qd10,qd20,qd30,qd40,p0,sx0,sz0,cx0,cz0,pd0,sxd0,szd0,cxd0,czd0,xF,zF,pF,q1F,q2F,q3F,q4F,xdF,zdF,pdF,qd1F,qd2F,qd3F,qd4F,pF,sxF,szF,cxF,czF,pdF,sxdF,szdF,cxdF,czdF\n"
        )

    def S2S_sim(self, q0:np.ndarray, qd0:np.ndarray) -> tuple:
        """Performs a closed loop simulation of the step to step dynamics (pre-impact to pre-impact)

        Args:
            q0 (np.ndarray): Full order initial configuration
            qd0 (np.ndarray): Full order initial velocity

        Returns:
            tuple: position, velocity, outputs, and output derivates resulting from S2S dynamics
        """
        # Set the simulator state
        self.mjInt.setState(q0, qd0)
        self.mjInt.forward()
        # Get the starting time of the simulation
        startTime = self.mjInt.time()
        # Reset the controller
        self.controller.reset()

        while True:
            # self.mjInt.updateScene()

            # Compute the S2S time
            t = self.mjInt.time() - startTime
            # Query state from Mujoco
            qpos = self.mjInt.getGenPosition()
            qvel = self.mjInt.getGenVelocity() 
            # Compute control action 
            q_pos_ref, q_vel_ref, q_ff_ref = self.controller.gaitController(qpos, qpos, qvel, t)

            # Apply control action
            self.mjInt.jointPosCmd(q_pos_ref)
            self.mjInt.jointVelCmd(q_vel_ref)
            self.mjInt.jointTorCmd(q_ff_ref)

            # If stance foot has changed, stop simulation
            if not self.controller.cur_stf:
                break

            # Step simulation forward
            self.mjInt.step()
        
        # Swap legs (take advantage of symmetry)
        qpos_copy = np.copy(qpos)
        qvel_copy = np.copy(qvel)
        qpos[3:5] = qpos_copy[5:]
        qpos[5:] = qpos_copy[3:5]
        qvel[3:5] = qvel_copy[5:]
        qvel[5:] = qvel_copy[3:5]
        # Compute with stanceFoot = False to swap legs
        qCpos = self.adamKin.calcOutputs(qpos, False)
        qCvel = self.adamKin.calcDOutputs(qpos, qvel, False)

        return qpos, qvel, qCpos, qCvel
    
    def iterateSetMap(self, verbose:bool=True) -> None:
        """Approximates the application of the set-valued S2S dynamics via propogating samples and convex outer approximation

        Args:
            verbose (bool, optional): Whether to give verbose output. Defaults to True.
        """
        if verbose:
            print("..... Sampling Set .....")
        # Sample points from the current convex outerapproximation
        points = self.sampleSet()
        propogatedPoints = np.zeros_like(points)

        if verbose:
            print("..... Computing S2S .....")
        # Propogate points through the S2S dynamics
        func = trange if verbose else range # use progress bar if verbose
        for ind in func(points.shape[0]):
            # Extract initial condition
            q0 = points[ind, :7]
            qd0 = points[ind, 7:14]

            # Compute S2S dynamics
            qF, qdF, qCF, qdCF = self.S2S_sim(q0, qd0)

            # Store results
            propogatedPoints[ind, :7] = qF
            propogatedPoints[ind, 7:14] = qdF
            propogatedPoints[ind, 14:19] = qCF
            propogatedPoints[ind, 19:] = qdCF

            # Log S2S dynamics
            self.logger.write(np.hstack((self.iteration, points[ind, :], propogatedPoints[ind, :])))

        # Add propogated points to the reachable list and table
        self.reachableList = np.vstack((self.reachableList, propogatedPoints))
        self.reachableTable[self.iteration + 1] = propogatedPoints

        if verbose:
            print("..... Fitting Set .....")
        # Fit a new convex outerapproximation
        self.fitSet()

        # Update iteration number and number of samples to take
        self.iteration += 1
        if self.NsampleSchedule is not None:
            self.Nsamples = self.NsampleSchedule(self.iteration)

        # Compute proportion of propogated points lying inside the set I.C.s were sampled from
        self.propInSet = self._pointsInSet()

        # Give verbose output
        if verbose:
            self.verboseOut()

    def sampleSet(self) -> np.ndarray:
        """Samples the most recent set

        Raises:
            ValueError: Certain set types have not been implemented

        Returns:
            np.ndarray: points sampled uniformly from the most recent set
        """
        # Handle Initial iteration separately
        if self.iteration == 0:
            return self.reachableList
        # Grab the method and data
        setMethod = self.setDiscriptions[self.iteration]["Method"]
        setData = self.setDiscriptions[self.iteration]["Data"]

        # Sample the set
        if setMethod == "Extreme Points":
            # Initialize return array
            sampledPoints = np.zeros((self.Nsamples, setData.shape[1]))
            # Grab only the set data involved in the set approximation
            setData = setData[:, :14] if self.approxSpace == "State" else setData[:, 14:]
            # Sample random integer to define how many points to take convex combination of
            numPts = np.random.randint(1, min(11, setData.shape[0]) + 1, size=(self.Nsamples,))
            # For each sampled point, construct as convex combination
            for ii, nPts in enumerate(numPts):
                # Randomly sample the given number of points
                inds = np.random.choice(range(setData.shape[0]), size=(nPts,), replace=False)
                # If it is a single point, return the point
                if nPts == 1:
                    sampledPoints[ii, :] = setData[inds, :]
                else:
                    # Return a convex combination of the given poitns
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
            u = np.random.normal(0, 1, (self.Nsamples, self.d + 2))
            norm = np.linalg.norm(u, axis=-1,keepdims=True)
            u = u / norm
            # The first N coordinates are uniform in a unit N ball
            sampledPoints = u[:, :self.d]
            sampledPoints = np.linalg.inv(sqrtm(setData["A"])) @ sampledPoints + setData["c"]
        elif setMethod == "InftyNorm":
            # Sample uniformly from the high dimensional rectangle
            sampledPoints = np.random.uniform(setData["LB"], setData["UB"], (self.Nsamples, setData["LB"].shape[0]))
        
        # Complete the points as necessary with either inverse kinematics (constructing state) or forward kinematics (completing outputs)
        if self.approxSpace == "Output":
            # Construct the initial pose/velo from the outputs
            sampledPoints = np.hstack((np.zeros((sampledPoints.shape[0], 14)), sampledPoints))
            for ii in range(sampledPoints.shape[0]):
                qpos, _ = self.adamKin.solveIK(self.q_ik, sampledPoints[ii, 14:19], False)
                sampledPoints[ii, :7] = self.modifyPoseForContact(qpos, False)
                sampledPoints[ii, 7:14] = self.adamKin.solveIKVel(sampledPoints[ii, :7], np.hstack((sampledPoints[ii, 19:], np.zeros((2,)))), False)
        elif self.approxSpace == "State":
            # Construct the outputs from the pose/velo
            sampledPoints = np.hstack((sampledPoints, np.zeros((sampledPoints.shape[0], 10))))
            for ii in range(sampledPoints.shape[0]):
                sampledPoints[ii, 14:19] = self.adamKin.calcOutputs(sampledPoints[ii, :7], False)
                sampledPoints[ii, 19:] =  self.adamKin.calcDOutputs(sampledPoints[ii, :7], sampledPoints[ii, 7:14], False)

        return sampledPoints
    
    def fitSet(self) -> None:
        """ Fits a convex outer approximation to the given set of points
        """
        # Get either the outputs or states to fit convex set around
        if self.approxSpace == "State":
            points = self.reachableList[:, :14]
            # May need to remove x variable to account for translational symmetry here
        elif self.approxSpace == "Output":
            points = self.reachableList[:, 14:]

        # Fits a convex outer approximation to a set of points
        if self.approxMethod == "Extreme Points" or self.reachableList.shape[0] < self.minSampleConvex:
            # Just use the set of points as the set definition
            # TODO: compute and only retain extreme points
            desc = self.reachableList
        
        elif self.approxMethod == "InftyNorm":
            # Compute infinity norm ball around points
            desc = {}
            desc["UB"] = np.max(points, axis=0)
            desc["LB"] = np.min(points, axis=0)
        
        elif self.approxMethod == "Ellipse":
            # Finds the ellipse equation in "center form" (x-c).T * A * (x-c) = 1
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
        
        # Save the set description for next iteration
        self.setDiscriptions[self.iteration + 1] = {"Method": self.approxMethod, "Data": desc}
    
    def getNominalState(self) -> tuple:
        """Computes the nominal state and outputs matching HLIP reference

        Returns:
            tuple: position, velocity, outputs, and output derivatives matching HLIP Reference
        """

        # Construct output reference
        y_out_ref = np.zeros((Kinematics.N_OUTPUTS))
        y_out_ref[Kinematics.OUT_ID["PITCH"]] = self.pitch_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_X"]] = self.u_ref
        y_out_ref[Kinematics.OUT_ID["SWF_POS_Z"]] = 0
        y_out_ref[Kinematics.OUT_ID["COM_POS_X"]] = self.x_pre_ref[0]
        y_out_ref[Kinematics.OUT_ID["COM_POS_Z"]] = self.z_ref

        # Compute position from inverse kinematics on outputs
        qpos, _ = self.adamKin.solveIK(self.q_ik, y_out_ref, False)
        # Modify the pose for contact
        qpos = self.modifyPoseForContact(qpos, False)
        
        # Construct the output reference derivatives
        yd_out_ref = np.array([
            0, 0, self.controller.swf_pos_z_poly.evalPoly(self.T_SSP, 1), self.x_pre_ref[1], 0, 0, 0
        ])
        # Compute configuration velocity to achieve output reference derivatives
        qvel = self.adamKin.solveIKVel(qpos, yd_out_ref, False)

        # Relabel outputs
        qCpos = y_out_ref
        qCvel = yd_out_ref[:-2] # Remove holonomic constraints (implied)

        return qpos, qvel, qCpos, qCvel
    
    def modifyPoseForContact(self, qpos:np.ndarray, stanceFoot:bool) -> np.ndarray:
        """Modifies floating base coordinates to initialize Mujoco contact

        Args:
            qpos (np.ndarray): robot configuration
            stanceFoot (bool): which foot is contacting the ground

        Returns:
            np.ndarray: modified robot configuation
        """
        # Initialize Mujoco with the state
        self.mjInt.setState(qpos, np.zeros_like(qpos))
        self.mjInt.forward()
        # Compute the foot position
        ft_pos = self.mjInt.getFootPos()
        # Modify the foot position to ensure nominal contact conditions
        qpos[Kinematics.GEN_POS_ID['P_Z']] -= ft_pos[1][1] - self.adamKin.getContactOffset(qpos, stanceFoot)
        return qpos
    
    # def sampleSetSurface(self):
    #     # Handle Initial iteration separately
    #     if self.iteration == 0:
    #         return self.reachableList
    #     # Samples a set of points from the most recent convex set
    #     setMethod = self.setDiscriptions[self.iteration]["Method"]
    #     setData = self.setDiscriptions[self.iteration]["Data"]
    #     if setMethod == "Extreme Points":
    #         raise ValueError("Extreme Points surface sampling not implemented yet")
    #     elif setMethod == "Polytope":
    #         raise ValueError("Polytope surface sampling not implemented yet")
    #     elif setMethod == "Ellipsoid":
    #         '''
    #         uniformly sample a N-dimensional unit UnitBall
    #         Reference:
    #         Efficiently sampling vectors and coordinates from the n-sphere and n-ball
    #         http://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    #         Input:
    #             num - no. of samples
    #             dim - dimensions
    #         Output:
    #             uniformly sampled points within N-dimensional unit ball
    #         '''
    #         #Sample on a unit N+1 sphere
    #         d = 10 if self.approxSpace == "Output" else 14
    #         u = np.random.normal(0, 1, (self.Nsamples, d))
    #         norm = np.linalg.norm(u, axis=-1,keepdims=True)
    #         sampledPoints = u / norm
    #         sampledPoints = np.linalg.inv(sqrtm(setData["A"])) @ sampledPoints + setData["c"]
    #     elif setMethod == "InftyNorm":
    #         sampledPoints = np.random.uniform(setData["LB"], setData["UB"], self.Nsamples)
    #         d = 10 if self.approxSpace == "Output" else 14
    #         rowind = np.array(range(self.Nsamples))
    #         probs = np.product(np.vstack([np.diag([0 if ii == jj else 1 for ii in range(d)]) @ (setData["UB"] - setData["LB"]) for jj in range(d)]) + np.eye(d))
    #         colind = np.random.choice(range(d), size=self.Nsamples, p=probs / np.sum(probs))
    #         high_low = np.random.randint(0, 2, self.Nsamples)
    #         sampledPoints[rowind, colind] = high_low * setData["UB"][rowind, colind] + (1 - high_low) * setData["UB"][rowind, colind]

        
    #     # Complete the points as necessary with either inverse kinematics (constructing state) or forward kinematics (completing outputs)
    #     if self.approxSpace == "Output":
    #         sampledPoints = np.hstack((np.zeros((sampledPoints.shape[0], 14), sampledPoints)))
    #         for ii in range(sampledPoints.shape[0]):
    #             sampledPoints[ii, :7] = self.adamKin.solveIK(sampledPoints[ii, 14:19], False)
    #             sampledPoints[ii, 7:14] = self.adamKin.solveIKVel(sampledPoints[ii, :7], np.hstack((sampledPoints[ii, 19:], np.zeros((2,)))), False)
    #     elif self.approxSpace == "State":
    #         sampledPoints = np.hstack((sampledPoints, np.zeros((sampledPoints.shape[0], 10))))
    #         for ii in range(sampledPoints.shape[0]):
    #             sampledPoints[ii, 14:19] = self.adamKin.calcOutputs(sampledPoints[ii, :7], False)
    #             sampledPoints[ii, 19:] =  self.adamKin.calcDOutputs(sampledPoints[ii, :7], sampledPoints[ii, 7:14], False)

    #     return sampledPoints
    
    def setSampleSize(self, Nsamples:int) -> None:
        """Sets the number of samples to take

        Args:
            Nsamples (int): number of sampels to take
        """
        self.Nsamples = Nsamples

    def setApproxMethod(self, method:str) -> None:
        """Set the type of convex set to use for outer approximation

        Args:
            method (str): type of convex set (Ellipsoid, InftyNorm, Extreme Points, Polytope)
        """
        self.approxMethod = method

    def getPropInSet(self) -> float:
        """Gets the proportion of propogated points which remained inside the set thery were sampled from

        Returns:
            float: Proportion of invariant points
        """
        return self.propInSet

    def _pointsInSet(self, iteration:int=None) -> float:
        """Computes the proportion of propogated points which remained inside the set thery were sampled from

        Args:
            iteration (int, optional): iteration to be examined. Defaults to None.

        Returns:
            float: proportion of invariant points
        """
        if iteration is None:
            iteration = self.iteration
        if iteration == 0 or iteration < 20:
            return 0
        if self.approxSpace == "Output":
            points = self.reachableTable[iteration][:, 14:]
        elif self.approxSpace == "State":
            points = self.reachableTable[iteration, :14]

        points_in = 0
        for ii in range(points.shape[0]):
            point = points[ii, :]
            points_in += self._pointInSet(point, iteration - 1)

        return points_in / points.shape[0]
    
    def _pointInSet(self, point:np.ndarray, iteration:int) -> bool:
        """Determine whether the given point is in the set defined in the given iteration

        Args:
            point (np.ndarray): point being queried
            iteration (int): set to check membership of

        Raises:
            ValueError: Some set types have not been implemented

        Returns:
            bool: whether the given point is in the given set
        """
        setMethod = self.setDiscriptions[iteration]["Method"]
        setData = self.setDiscriptions[iteration]["Data"]
        if setMethod == "Extreme Points" or self.reachableList.shape[0] < self.minSampleConvex:
            # raise ValueError("Polytope sampling not implemented yet")
            # print("Extreme point in set not implemented yet\n")
            return np.NaN
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
            return np.dot(point - setData["c"], setData["A"] @ (point - setData["c"])) <= 1
        elif setMethod == "InftyNorm":
            return np.all(point <= setData["UB"]) and np.all(point >= setData["LB"])
        
    
    def verboseOut(self) -> None:
        """Print some statistics after a run
        """
        # First, proportion of points which were set invariant
        print(f"Proportion of points in set: {self.propInSet}")
        if self.iteration < 2:
            return
        # Second, a measure of how much the outer approximation changed
        if self.approxMethod == "InftyNorm":
            dUB = self.setDiscriptions[self.iteration]["Data"]["UB"] - self.setDiscriptions[self.iteration - 1]["Data"]["UB"]
            dLB = self.setDiscriptions[self.iteration]["Data"]["LB"] - self.setDiscriptions[self.iteration - 1]["Data"]["LB"]

            print(f"UB Change: {dUB}\nLB Change: {dLB}")

        # Print next iteraation number
        print(f"\n\nIteration {self.iteration}")
        
        