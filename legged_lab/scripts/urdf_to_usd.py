#!/usr/bin/env python3
"""
将 URDF 机器人模型导出为 USD（仅视觉 mesh，保留层级与变换）。

依赖（见 scripts/requirements-urdf2usd.txt）:
  pip install -r legged_lab/scripts/requirements-urdf2usd.txt

用法:
  python urdf_to_usd.py <input.urdf> [output.usd]
  python urdf_to_usd.py assets/roban_s14/urdf/biped_s14.urdf  # 输出 biped_s14.usd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 提前校验依赖，便于报错时看到具体缺失的模块
def _check_deps():
    try:
        import urdfpy  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"缺少依赖 urdfpy，请执行: pip install -r scripts/requirements-urdf2usd.txt\n原始错误: {e}"
        ) from e
    try:
        import trimesh  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"缺少依赖 trimesh，请执行: pip install trimesh\n原始错误: {e}") from e
    try:
        from pxr import Usd, UsdGeom  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"缺少依赖 usd-core，请执行: pip install usd-core\n原始错误: {e}"
        ) from e


def _rpy_to_rotation_matrix(rpy):
    """Roll-Pitch-Yaw (rad) 转 3x3 旋转矩阵。"""
    import numpy as np
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _origin_to_matrix(origin_xyz, origin_rpy):
    """URDF origin (xyz, rpy) 转 4x4 齐次变换矩阵。"""
    import numpy as np
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rotation_matrix(origin_rpy)
    T[:3, 3] = [float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])]
    return T


def _get_link_parent_and_joint_origin(robot):
    """返回每个 link 的 (parent_link_name, joint_origin_4x4)，根为 (None, I)。"""
    import numpy as np
    link_to_parent = {}
    link_to_joint_origin = {}
    for joint in robot.joints:
        child = joint.child
        parent = joint.parent
        xyz = getattr(joint.origin, 'xyz', [0, 0, 0])
        rpy = getattr(joint.origin, 'rpy', [0, 0, 0])
        origin = _origin_to_matrix(xyz, rpy)
        link_to_parent[child] = parent
        link_to_joint_origin[child] = origin
    # 根 link（没有作为 child 出现的）
    all_children = set(link_to_parent)
    for link in robot.links:
        if link.name not in all_children:
            link_to_parent[link.name] = None
            link_to_joint_origin[link.name] = np.eye(4)
            break
    return link_to_parent, link_to_joint_origin


def _link_world_transforms(robot):
    """从根到叶计算每个 link 在世界坐标系下的 4x4 变换。"""
    import numpy as np
    parent, joint_origin = _get_link_parent_and_joint_origin(robot)
    world_tf = {}
    # 找根
    root = next((n for n, p in parent.items() if p is None), None)
    if root is None:
        raise ValueError("URDF 中未找到根 link")
    world_tf[root] = np.eye(4)
    # BFS 计算子 link 的世界变换
    from collections import deque
    q = deque([root])
    while q:
        name = q.popleft()
        T_world = world_tf[name]
        for j in robot.joints:
            if j.parent != name:
                continue
            child = j.child
            xyz = getattr(j.origin, 'xyz', [0, 0, 0])
            rpy = getattr(j.origin, 'rpy', [0, 0, 0])
            T_joint = _origin_to_matrix(xyz, rpy)
            world_tf[child] = T_world @ T_joint
            q.append(child)
    return world_tf


def _add_mesh_to_stage(stage, path, trimesh_obj, world_transform, name):
    """将 trimesh 几何写入 USD 并在给定 path 下创建 Mesh prim，应用 world_transform。"""
    from pxr import UsdGeom, Gf, Sdf
    mesh_prim = UsdGeom.Mesh.Define(stage, path)
    pts = trimesh_obj.vertices
    mesh_prim.CreatePointsAttr(pts.tolist())
    # USD 面: faceVertexCounts + faceVertexIndices
    faces = trimesh_obj.faces
    if len(faces) == 0:
        return
    fvc = [3] * len(faces)  # 三角形
    fvi = faces.flatten().tolist()
    mesh_prim.CreateFaceVertexCountsAttr(fvc)
    mesh_prim.CreateFaceVertexIndicesAttr(fvi)
    # 法线（可选，便于显示）
    if hasattr(trimesh_obj, 'vertex_normals') and trimesh_obj.vertex_normals is not None:
        mesh_prim.CreateNormalsAttr(trimesh_obj.vertex_normals.flatten().tolist())
        mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    # 应用世界变换到 Xformable
    xform = UsdGeom.Xformable(mesh_prim)
    m = world_transform
    xform.AddTransformOp().Set(
        Gf.Matrix4d(
            m[0, 0], m[0, 1], m[0, 2], m[0, 3],
            m[1, 0], m[1, 1], m[1, 2], m[1, 3],
            m[2, 0], m[2, 1], m[2, 2], m[2, 3],
            m[3, 0], m[3, 1], m[3, 2], m[3, 3],
        )
    )


def urdf_to_usd(urdf_path: str | Path, usd_path: str | Path, robot_name: str | None = None) -> None:
    """将 URDF 文件转换为 USD（仅视觉 mesh，保留层级与变换）。"""
    import numpy as np
    from pxr import Usd, UsdGeom, Gf, Sdf
    try:
        import urdfpy
    except ImportError as e:
        raise ImportError("请安装 urdfpy: pip install urdfpy") from e
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("请安装 trimesh: pip install trimesh") from e

    urdf_path = Path(urdf_path).resolve()
    usd_path = Path(usd_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 不存在: {urdf_path}")

    # 若 URDF 含非标准 <mujoco>，先复制为临时文件并去掉该块，避免解析错误
    urdf_dir = urdf_path.parent
    with open(urdf_path, "r", encoding="utf-8") as f:
        raw = f.read()
    if "<mujoco>" in raw and "</mujoco>" in raw:
        import re
        raw = re.sub(r"\s*<mujoco>.*?</mujoco>\s*", "\n", raw, flags=re.DOTALL)
        import tempfile
        # 在 URDF 同目录创建临时文件，保证 mesh 相对路径 ../meshes/ 有效
        fd, tmp_path = tempfile.mkstemp(suffix=".urdf", prefix="urdf_", dir=str(urdf_dir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                tmp.write(raw)
            robot = urdfpy.URDF.load(tmp_path, load_meshes=True)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    else:
        robot = urdfpy.URDF.load(str(urdf_path), load_meshes=True)

    name = robot_name or robot.name or "robot"
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_prim = stage.DefinePrim("/" + name, "Xform")
    stage.SetDefaultPrim(root_prim)

    link_tfs = _link_world_transforms(robot)

    for link in robot.links:
        link_name = link.name
        T_link = link_tfs.get(link_name, np.eye(4))
        for i, vis in enumerate(link.visuals):
            geom = vis.geometry
            if not hasattr(geom, "mesh") or geom.mesh is None:
                continue
            # 视觉的局部变换
            xyz = getattr(vis.origin, "xyz", [0, 0, 0])
            rpy = getattr(vis.origin, "rpy", [0, 0, 0])
            T_vis = _origin_to_matrix(xyz, rpy)
            T_world = T_link @ T_vis
            # 获取 trimesh：优先已加载的 meshes
            meshes = getattr(geom.mesh, "meshes", None)
            if meshes:
                for j, tm in enumerate(meshes):
                    if tm is None:
                        continue
                    mesh_name = f"{link_name}_vis{i}_{j}"
                    path = f"/{name}/{mesh_name}"
                    _add_mesh_to_stage(stage, path, tm, T_world, mesh_name)
            else:
                # 回退：用 filename 再加载一次
                fn = getattr(geom.mesh, "filename", None)
                if not fn:
                    continue
                if not os.path.isabs(fn):
                    fn = (urdf_dir / fn).resolve()
                if not os.path.exists(fn):
                    continue
                try:
                    tm = trimesh.load(str(fn), force="mesh")
                    if isinstance(tm, trimesh.Scene):
                        for j, g in enumerate(tm.geometry.values()):
                            _add_mesh_to_stage(
                                stage,
                                f"/{name}/{link_name}_vis{i}_{j}",
                                g,
                                T_world,
                                f"{link_name}_vis{i}_{j}",
                            )
                    else:
                        _add_mesh_to_stage(
                            stage, f"/{name}/{link_name}_vis{i}", tm, T_world, link_name
                        )
                except Exception as e:
                    print(f"警告: 无法加载 mesh {fn}: {e}", file=sys.stderr)

    stage.GetRootLayer().Export(str(usd_path))
    print(f"已导出: {usd_path}")


def main():
    parser = argparse.ArgumentParser(description="URDF 转 USD（仅视觉 mesh）")
    parser.add_argument("urdf", type=str, help="输入 URDF 文件路径")
    parser.add_argument(
        "usd",
        type=str,
        nargs="?",
        default=None,
        help="输出 USD 文件路径（默认与 URDF 同目录、同名 .usd）",
    )
    parser.add_argument("--name", type=str, default=None, help="USD 根 prim 名称")
    args = parser.parse_args()
    _check_deps()

    urdf_path = Path(args.urdf).resolve()
    if args.usd:
        usd_path = Path(args.usd).resolve()
    else:
        usd_path = urdf_path.with_suffix(".usd")

    urdf_to_usd(urdf_path, usd_path, robot_name=args.name)


if __name__ == "__main__":
    main()
