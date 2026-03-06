#!/usr/bin/env python3
"""
将 URDF 文件转换为 JSON 格式
使用 Python 标准库，无需额外依赖
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_value(val_str):
    """尝试将字符串转换为数字"""
    if val_str is None:
        return None
    val_str = val_str.strip()
    # 空格分隔的数字（如 "0 0 0"）
    if " " in val_str and not val_str.startswith("../"):
        parts = val_str.split()
        try:
            return [float(p) if "." in p or "e" in p.lower() else int(p) for p in parts]
        except ValueError:
            return val_str
    try:
        return float(val_str) if "." in val_str or "e" in val_str.lower() else int(val_str)
    except ValueError:
        return val_str


def element_to_dict(element):
    """递归将 XML 元素转换为 Python 字典"""
    # 构建结果
    result = {}
    
    # 处理属性
    if element.attrib:
        result["@"] = {}
        for key, val in element.attrib.items():
            # 对常见数值属性进行转换
            if key in ("xyz", "rpy", "xy", "rgba", "size", "scale") or "value" in key.lower():
                parsed = parse_value(val)
                result["@"][key] = parsed if parsed is not None else val
            else:
                result["@"][key] = val
    
    # 处理子元素
    children = list(element)
    if children:
        child_dict = {}
        for child in children:
            tag = child.tag
            child_data = element_to_dict(child)
            
            # 简化单子元素的结构：如果子元素只有 @ 属性且无其他内容，可扁平化
            if tag in child_dict:
                # 同类型元素出现多次，转为数组
                if not isinstance(child_dict[tag], list):
                    child_dict[tag] = [child_dict[tag]]
                child_dict[tag].append(child_data)
            else:
                child_dict[tag] = child_data
        result.update(child_dict)
    
    # 处理文本内容
    text = (element.text or "").strip()
    if text:
        result["#text"] = text
    
    return result


def urdf_to_json(urdf_path: str, json_path: str = None, indent: int = 2):
    """
    将 URDF 文件转换为 JSON
    
    Args:
        urdf_path: URDF 文件路径
        json_path: 输出的 JSON 文件路径，默认为同目录下的 .json 文件
        indent: JSON 缩进空格数
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")
    
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # 移除默认的 XML 命名空间（如果存在）
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
    
    data = element_to_dict(root)
    
    if json_path is None:
        json_path = urdf_path.with_suffix(".json")
    else:
        json_path = Path(json_path)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    return json_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        urdf_file = Path(__file__).parent / "biped_s14.urdf"
        print(f"使用默认文件: {urdf_file}")
    else:
        urdf_file = sys.argv[1]
    
    output = sys.argv[2] if len(sys.argv) > 2 else None
    result = urdf_to_json(urdf_file, output)
    print(f"已转换: {result}")
