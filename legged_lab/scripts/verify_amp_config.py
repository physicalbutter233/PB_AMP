#!/usr/bin/env python3
"""
验证 AMP 配置是否正确加载的脚本
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from legged_lab.utils import task_registry
from legged_lab.envs import *  # noqa:F401, F403

def verify_amp_config():
    """验证 AMP 配置是否正确加载"""
    
    print("=" * 80)
    print("AMP 配置验证")
    print("=" * 80)
    
    # 1. 读取配置
    env_class_name = "RobanWalkFlatEnvCfg"
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    
    print("\n1. 从配置文件读取的值:")
    print(f"   amp_reward_coef = {agent_cfg.amp_reward_coef}")
    print(f"   amp_task_reward_lerp = {agent_cfg.amp_task_reward_lerp}")
    print(f"   track_lin_vel_xy_exp.weight = {env_cfg.reward.track_lin_vel_xy_exp.weight}")
    print(f"   track_ang_vel_z_exp.weight = {env_cfg.reward.track_ang_vel_z_exp.weight}")
    
    # 2. 转换为字典
    agent_dict = agent_cfg.to_dict()
    
    print("\n2. 转换为字典后的值:")
    print(f"   agent_dict['amp_reward_coef'] = {agent_dict.get('amp_reward_coef', 'NOT FOUND')}")
    print(f"   agent_dict['amp_task_reward_lerp'] = {agent_dict.get('amp_task_reward_lerp', 'NOT FOUND')}")
    
    # 3. 验证值是否正确
    print("\n3. 配置验证:")
    expected_coef = 0.01
    expected_lerp = 0.96
    
    if agent_dict.get('amp_reward_coef') == expected_coef:
        print(f"   ✅ amp_reward_coef = {expected_coef} (正确)")
    else:
        print(f"   ❌ amp_reward_coef = {agent_dict.get('amp_reward_coef')} (期望: {expected_coef})")
    
    if agent_dict.get('amp_task_reward_lerp') == expected_lerp:
        print(f"   ✅ amp_task_reward_lerp = {expected_lerp} (正确)")
    else:
        print(f"   ❌ amp_task_reward_lerp = {agent_dict.get('amp_task_reward_lerp')} (期望: {expected_lerp})")
    
    # 4. 检查所有 AMP 相关配置
    print("\n4. 所有 AMP 相关配置:")
    amp_keys = [k for k in agent_dict.keys() if 'amp' in k.lower()]
    for key in sorted(amp_keys):
        print(f"   {key} = {agent_dict[key]}")
    
    print("\n" + "=" * 80)
    print("验证完成")
    print("=" * 80)
    
    return agent_dict.get('amp_reward_coef') == expected_coef and agent_dict.get('amp_task_reward_lerp') == expected_lerp

if __name__ == "__main__":
    success = verify_amp_config()
    sys.exit(0 if success else 1)
