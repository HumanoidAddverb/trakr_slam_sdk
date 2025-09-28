# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import io
from typing import Optional

import numpy as np
import omni
import omni.kit.commands
import yaml
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from isaacsim.storage.native import get_assets_root_path


class TrakrFlatTerrainPolicy(PolicyController):
    """The trakr quadruped"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "spot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize robot and load RL policy.

        Args:
            prim_path {str} -- prim path of the robot on the stage
            name {str} -- name of the quadruped
            usd_path {str} -- robot usd filepath in the directory
            position {np.ndarray} -- position of the robot
            orientation {np.ndarray} -- orientation of the robot

        """
        self.load_params()
        super().__init__(name, prim_path, root_path, usd_path, position,
                         orientation)

        self.load_policy(
            "/home/quad/isaacsim/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/trakr_simulator/models/policy.pt",
            "/home/quad/isaacsim/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/trakr_simulator/config/env.yaml",
        )
        self._previous_action = np.zeros(12)
        self._policy_counter = 0

    def load_params(self):

        file_path = "../config/run_params.yaml"
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        self._action_scale = data["action_scale"]
        self._default_joint_pos = np.array(data["default_joint_pos"])
        self._decimation = data["decimation"]
        self.linear_vel_scale = data["linear_vel_scale"]
        self.angular_vel_scale = data["angular_vel_scale"]
        self.p_gain = data["p_gain"]
        self.d_gain = data["d_gain"]

    def _compute_observation(self, command):
        """
        Computes the the observation vector for the policy

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        lin_vel_I = self.robot.get_linear_velocity() * self.linear_vel_scale
        ang_vel_I = self.robot.get_angular_velocity() * self.angular_vel_scale
        pos_IB, q_IB = self.robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = np.matmul(R_BI, lin_vel_I)
        ang_vel_b = np.matmul(R_BI, ang_vel_I)
        gravity_b = np.matmul(R_BI, np.array([0.0, 0.0, -1.0]))

        obs = np.zeros(48)
        # Base lin vel
        obs[:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command
        obs[9:12] = command * [
            self.linear_vel_scale,
            self.linear_vel_scale,
            self.angular_vel_scale,
        ]
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()
        obs[12:24] = current_joint_pos - self._default_joint_pos
        obs[24:36] = current_joint_vel

        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    def advance(self, dt, command):
        """
        Compute the desired torques and apply them to the articulation

        Argument:
        dt {float} -- Timestep update in the world.
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        """
        # print(command)
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)
            self._previous_action = self.action.copy()

        action_scaled = ArticulationAction(
            joint_positions=self._default_joint_pos +
            (self.action * self._action_scale))
        dof_pos = self.robot.get_joint_positions()
        dof_vel = self.robot.get_joint_velocities()
        p_gain = self.p_gain
        d_gain = self.d_gain
        joint_torques = p_gain * (action_scaled.joint_positions -
                                  dof_pos) - d_gain * (dof_vel)
        joint_torques_articulation = ArticulationAction(
            joint_efforts=joint_torques)
        self.robot.apply_action(joint_torques_articulation)
        self._policy_counter += 1

    def initialize(self, physics_sim_view=None) -> None:
        """
        Initialize robot the articulation interface, set up drive mode
        """
        super().initialize(physics_sim_view=physics_sim_view,
                           control_mode="effort")

    def post_reset(self) -> None:
        """
        Post reset articulation
        """
        self.robot.post_reset()
