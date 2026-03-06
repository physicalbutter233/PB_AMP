#!/usr/bin/env python3
"""
简单检查 AMP 配置的脚本（直接读取配置文件）
"""

import re
import sys

def check_amp_config():
    """检查配置文件中的 AMP 参数"""
    
    config_file = "legged_lab/envs/roban/walk_cfg.py"
    
    print("=" * 80)
    print("检查 AMP 配置")
    print("=" * 80)
    print(f"配置文件: {config_file}\n")
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # 查找 amp_reward_coef
        coef_match = re.search(r'amp_reward_coef\s*=\s*([0-9.]+)', content)
        if coef_match:
            coef_value = float(coef_match.group(1))
            print(f"✅ amp_reward_coef = {coef_value}")
            if coef_value == 0.01:
                print("   ✅ 配置正确（期望值: 0.01）")
            else:
                print(f"   ⚠️  配置值不是期望的 0.01，当前值: {coef_value}")
        else:
            print("❌ 未找到 amp_reward_coef")
        
        # 查找 amp_task_reward_lerp
        lerp_match = re.search(r'amp_task_reward_lerp\s*=\s*([0-9.]+)', content)
        if lerp_match:
            lerp_value = float(lerp_match.group(1))
            print(f"✅ amp_task_reward_lerp = {lerp_value}")
            if lerp_value == 0.96:
                print("   ✅ 配置正确（期望值: 0.96）")
            else:
                print(f"   ⚠️  配置值不是期望的 0.96，当前值: {lerp_value}")
        else:
            print("❌ 未找到 amp_task_reward_lerp")
        
        # 查找移动奖励权重
        lin_vel_match = re.search(r'track_lin_vel_xy_exp.*weight\s*=\s*([0-9.]+)', content)
        if lin_vel_match:
            lin_vel_weight = float(lin_vel_match.group(1))
            print(f"✅ track_lin_vel_xy_exp.weight = {lin_vel_weight}")
            if lin_vel_weight >= 10.0:
                print("   ✅ 配置正确（期望值: >= 10.0）")
            else:
                print(f"   ⚠️  配置值可能不够大，当前值: {lin_vel_weight}")
        
        ang_vel_match = re.search(r'track_ang_vel_z_exp.*weight\s*=\s*([0-9.]+)', content)
        if ang_vel_match:
            ang_vel_weight = float(ang_vel_match.group(1))
            print(f"✅ track_ang_vel_z_exp.weight = {ang_vel_weight}")
            if ang_vel_weight >= 8.0:
                print("   ✅ 配置正确（期望值: >= 8.0）")
            else:
                print(f"   ⚠️  配置值可能不够大，当前值: {ang_vel_weight}")
        
        print("\n" + "=" * 80)
        print("检查完成")
        print("=" * 80)
        
        # 返回是否配置正确
        coef_ok = coef_match and float(coef_match.group(1)) == 0.01
        lerp_ok = lerp_match and float(lerp_match.group(1)) == 0.96
        
        return coef_ok and lerp_ok
        
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    except Exception as e:
        print(f"❌ 读取配置文件时出错: {e}")
        return False

if __name__ == "__main__":
    success = check_amp_config()
    sys.exit(0 if success else 1)
