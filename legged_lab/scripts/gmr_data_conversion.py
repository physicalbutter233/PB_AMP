import pickle
import numpy as np
import argparse
from scipy.spatial.transform import Rotation

# Roban S14 / biped_s14 的 23 关节中，移除手腕后的索引映射
# 原始顺序含 left_wrist_joint(17)、right_wrist_joint(22)
ROBAN_JOINT_INDICES_NO_WRIST = [i for i in range(23) if i not in (17, 22)]


def convert_pkl_to_custom(input_pkl, output_txt, fps, remove_roban_wrist=False):
    dt = 1.0 / fps

    with open(input_pkl, "rb") as f:
        motion_data = pickle.load(f)

    root_pos = motion_data["root_pos"]
    root_rot = motion_data["root_rot"][:, [3, 0, 1, 2]]  # xyzw → wxyz
    dof_pos = motion_data["dof_pos"]

    if remove_roban_wrist:
        if dof_pos.shape[1] != 23:
            raise ValueError(
                f"remove_roban_wrist 要求 dof_pos 为 23 列 (Roban S14)，当前为 {dof_pos.shape[1]} 列"
            )
        dof_pos = dof_pos[:, ROBAN_JOINT_INDICES_NO_WRIST]
        print(f"  已移除 Roban 手腕关节 (索引 17, 22)，输出 21 关节")

    root_lin_vel = (root_pos[1:] - root_pos[:-1]) / dt

    # 角速度：用 scipy 计算相邻帧四元数差对应的 axis-angle，不依赖 isaaclab
    root_rot_xyzw = root_rot[:, [1, 2, 3, 0]]  # wxyz → xyzw for scipy
    root_ang_vel = np.zeros((len(root_pos) - 1, 3))
    for i in range(len(root_pos) - 1):
        r1 = Rotation.from_quat(root_rot_xyzw[i])
        r2 = Rotation.from_quat(root_rot_xyzw[i + 1])
        dq_rot = r2 * r1.inv()
        root_ang_vel[i] = dq_rot.as_rotvec() / dt

    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt

    euler_angles = Rotation.from_quat(root_rot[:-1, [1, 2, 3, 0]]).as_euler('XYZ', degrees=False)
    euler_angles = np.unwrap(euler_angles, axis=0)

    data_output = np.concatenate(
        (root_pos[:-1], euler_angles, dof_pos[:-1],  
         root_lin_vel, root_ang_vel, dof_vel),
        axis=1
    )

    np.savetxt(output_txt, data_output, fmt='%f', delimiter=', ')
    with open(output_txt, 'r') as f:
        frames_data = f.readlines()

    frames_data_len = len(frames_data)
    with open(output_txt, 'w') as f:
        f.write('{\n')
        f.write('"LoopMode": "Wrap",\n')
        f.write(f'"FrameDuration": {1.0/fps:.3f},\n')
        f.write('"EnableCycleOffsetPosition": true,\n')
        f.write('"EnableCycleOffsetRotation": true,\n')
        f.write('"MotionWeight": 0.5,\n\n')
        f.write('"Frames":\n[\n')

        for i, line in enumerate(frames_data):
            line_start_str = '  ['
            if i == frames_data_len - 1:
                f.write(line_start_str + line.rstrip() + ']\n')
            else:
                f.write(line_start_str + line.rstrip() + '],\n')

        f.write(']\n}')
    print(f"✅ Successfully converted {input_pkl} to {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True)
    parser.add_argument("--output_txt", type=str, default=None, help="Output .txt path (AMP format)")
    parser.add_argument("--output_file", type=str, default=None, help="Alias for --output_txt")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--remove_roban_wrist",
        action="store_true",
        help="移除 Roban S14 的两个手腕关节 (索引 17, 22)，将 23 关节转为 21 关节",
    )
    args = parser.parse_args()

    output_txt = args.output_txt or args.output_file
    if not output_txt:
        parser.error("必须指定 --output_txt 或 --output_file")
    convert_pkl_to_custom(args.input_pkl, output_txt, args.fps, args.remove_roban_wrist)
