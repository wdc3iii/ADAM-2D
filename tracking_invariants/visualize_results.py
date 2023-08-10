import cv2
import csv
import yaml
import numpy as np
from tracking_invariants.tracking_invariant import TrackingInvariant

mesh_path = "rsc/models/"
log_path = "plot/log_tracking_invariant.csv"
urdf_path = "rsc/models/adam2d.urdf"
xml_path = "rsc/models/adam2d.xml"
video_path = "video"
num_s2s = 10

def main():
    with open('rsc/track_inv_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    use_static_com = config["use_static_com"]
    gravity_comp = config["gravity_comp"]
    useAngMomState = config["use_ang_mom_state"]
    use_task_space_ctrl = config["use_task_space_ctrl"]
    pitch_ref = config["pitch_ref"]

    z_ref = config["z_ref"]
    v_ref = config["v_ref"]
    T_SSP = config["T_SSP"]
    approxMethod = config["approxMethod"]
    vis = config["vis"]

    tracking_invariant = TrackingInvariant(v_ref, z_ref, pitch_ref, T_SSP, approxMethod, 10, use_static_com=use_static_com, useAngMomState=useAngMomState, gravity_comp=gravity_comp, use_task_space_ctrl=use_task_space_ctrl, visualize=vis, log=False)


    with open(log_path, 'r') as x:
        sample_data = list(csv.reader(x, delimiter=","))
    
    # header = sample_data[0]
    data = np.array(sample_data[1:], dtype=float)
    data = data[data[:, 0] == max(data[:, 0]), :]
    inds = np.random.choice(data.shape[0], num_s2s, replace=False)

    for ind in inds:
        print(ind)
        q0 = data[ind, 1:8]
        qd0 = data[ind, 8:15]

        frames = tracking_invariant.S2S_sim(q0, qd0, vis=True)

        out = cv2.VideoWriter(f'{video_path}/{ind}_S2S.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 60, (frames[0].shape[1], frames[0].shape[0]))

        for frame in frames:
            out.write(np.flip(frame, axis=2))
        out.release()




if __name__ == "__main__":
    main()