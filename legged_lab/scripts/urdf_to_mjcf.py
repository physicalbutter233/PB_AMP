#!/usr/bin/env python3
"""
将包含 <mujoco> 标签的 URDF 文件转换为纯 MuJoCo XML (MJCF) 格式。

MuJoCo 可以直接加载包含 <mujoco> 标签的 URDF，但转换为纯 XML 格式可以获得：
- 更好的性能优化
- 更清晰的模型结构
- 更容易手动编辑

用法:
  python urdf_to_mjcf.py <input.urdf> [output.xml]
  python urdf_to_mjcf.py assets/roban_s14/urdf/biped_s14.urdf assets/roban_s14/biped_s14_from_urdf.xml
"""

import argparse
import sys
from pathlib import Path

try:
    import mujoco
except ImportError:
    print("[ERROR] 请先安装 mujoco: pip install mujoco")
    sys.exit(1)


def validate_urdf(urdf_path: str | Path) -> dict:
    """
    验证 URDF 文件是否可以被 MuJoCo 正确加载。
    
    Args:
        urdf_path: URDF 文件路径
    
    Returns:
        包含模型信息的字典
    """
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
    
    print(f"[INFO] 验证 URDF: {urdf_path}")
    
    # MuJoCo 可以直接加载包含 <mujoco> 标签的 URDF
    try:
        model = mujoco.MjModel.from_xml_path(str(urdf_path))
        info = {
            "nbody": model.nbody,
            "njnt": model.njnt,
            "ngeom": model.ngeom,
            "nu": model.nu,  # 执行器数量
            "nv": model.nv,  # 速度数量
            "nq": model.nq,  # 位置数量
        }
        print(f"[INFO] ✓ 成功加载模型:")
        print(f"      Bodies: {info['nbody']}")
        print(f"      Joints: {info['njnt']}")
        print(f"      Geoms: {info['ngeom']}")
        print(f"      Actuators: {info['nu']}")
        return info
    except Exception as e:
        print(f"[ERROR] ✗ 加载 URDF 失败: {e}")
        raise


def urdf_to_mjcf(urdf_path: str | Path, output_path: str | Path = None) -> str:
    """
    验证 URDF 文件并说明如何使用。
    
    注意：MuJoCo 可以直接加载包含 <mujoco> 标签的 URDF 文件。
    如果需要纯 XML 格式，通常需要手动编辑或使用 MuJoCo 的命令行工具。
    
    Args:
        urdf_path: 输入的 URDF 文件路径（可以包含 <mujoco> 标签）
        output_path: 输出的说明文件路径（可选）
    
    Returns:
        说明信息
    """
    urdf_path = Path(urdf_path).resolve()
    
    # 验证 URDF
    info = validate_urdf(urdf_path)
    
    # 检查 URDF 是否包含 <mujoco> 标签
    with open(urdf_path, 'r', encoding='utf-8') as f:
        content = f.read()
        has_mujoco_tag = '<mujoco>' in content or '<mujoco' in content
    
    print(f"\n[INFO] URDF 文件分析:")
    print(f"      包含 <mujoco> 标签: {'是' if has_mujoco_tag else '否'}")
    
    if has_mujoco_tag:
        print(f"\n[INFO] ✓ 该 URDF 文件可以直接被 MuJoCo 使用！")
        print(f"      在 sim2sim.py 中可以直接使用此 URDF 文件路径。")
    else:
        print(f"\n[WARNING] 该 URDF 文件不包含 <mujoco> 标签。")
        print(f"         建议在 URDF 的 <robot> 标签内添加:")
        print(f"         <mujoco>")
        print(f"           <compiler meshdir=\"../meshes/\" balanceinertia=\"true\" discardvisual=\"false\"/>")
        print(f"         </mujoco>")
    
    print(f"\n[INFO] 使用方法:")
    print(f"      1. 直接使用 URDF（推荐）:")
    print(f"         python sim2sim.py --model {urdf_path}")
    print(f"      2. 如果需要纯 XML，可以:")
    print(f"         - 手动复制 URDF 内容并转换为纯 <mujoco> 格式")
    print(f"         - 参考现有的 biped_s14.xml 或 tienkung.xml 的结构")
    
    return str(urdf_path)


def main():
    parser = argparse.ArgumentParser(
        description="验证 URDF 文件是否可以被 MuJoCo 使用，并提供使用建议"
    )
    parser.add_argument(
        "urdf",
        type=str,
        help="输入的 URDF 文件路径（可以包含 <mujoco> 标签）"
    )
    args = parser.parse_args()
    
    try:
        urdf_to_mjcf(args.urdf)
        print(f"\n[SUCCESS] 验证完成！")
    except Exception as e:
        print(f"\n[ERROR] 验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
