# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


# from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg

from isaaclab.utils import configclass

# from .actuator_pd import DelayedImplicitActuator
from .actuator_pd import DelayedPDActuator_RobanS2
from dataclasses import MISSING
import torch


@configclass
class DelayedPDActuatorCfg_RobanS2(DelayedPDActuatorCfg):
    """Configuration for a delayed PD actuator."""

    class_type: type = DelayedPDActuator_RobanS2
    friction_static: float | dict[str, float] = 0
    activation_vel: float = torch.inf
    friction_dynamic: float | dict[str, float] = 0


