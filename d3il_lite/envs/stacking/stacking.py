import copy

import cv2
import numpy as np
from gymnasium.spaces import Box, Dict

from d3il_lite.d3il_sim.core import Scene
from d3il_lite.d3il_sim.core.Logger import CamLogger, ObjectLogger
from d3il_lite.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from d3il_lite.d3il_sim.gyms.gym_utils.helpers import obj_distance
from d3il_lite.d3il_sim.sims import MjCamera
from d3il_lite.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from d3il_lite.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from d3il_lite.d3il_sim.utils.geometric_transformation import (
    euler2quat,
    quat2euler,
)
from d3il_lite.d3il_sim.utils.sim_path import d3il_path

from .stacking_objects import get_obj_list, init_end_eff_pos

obj_list = get_obj_list()


class BPCageCam(MjCamera):
    """
    Cage camera. Extends the camera base class.
    """

    def __init__(self, width: int = 96, height: int = 96, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width,
            height,
            init_pos=[1.05, 0, 1.2],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],  # Looking with 30 deg to the robot
            *args,
            **kwargs,
        )


class BlockContextManager:
    def __init__(self, scene, index=0, seed=42) -> None:
        self.scene = scene

        np.random.seed(seed)

        self.red_space = Box(
            low=np.array([0.35, -0.25, -90]),
            high=np.array([0.45, -0.15, 90]),  # seed=seed
        )

        self.green_space = Box(
            low=np.array([0.35, -0.1, -90]),
            high=np.array([0.45, 0, 90]),  # seed=seed
        )

        self.blue_space = Box(
            low=np.array([0.55, -0.2, -90]),
            high=np.array([0.6, 0, 90]),  # seed=seed
        )

        self.target_space = Box(
            low=np.array([0.4, 0.15, -90]),
            high=np.array([0.6, 0.25, 90]),  # seed=seed
        )

        self.index = index

    def start(self, random=True, context=None):
        if random:
            self.context = self.sample()
        else:
            self.context = context

        self.set_context(self.context)

    def sample(self):
        pos_1 = self.red_space.sample()
        angle_1 = [0, 0, pos_1[-1] * np.pi / 180]
        quat_1 = euler2quat(angle_1)

        pos_2 = self.green_space.sample()
        angle_2 = [0, 0, pos_2[-1] * np.pi / 180]
        quat_2 = euler2quat(angle_2)

        pos_3 = self.blue_space.sample()
        angle_3 = [0, 0, pos_3[-1] * np.pi / 180]
        quat_3 = euler2quat(angle_3)

        pos_4 = self.target_space.sample()
        angle_4 = [0, 0, pos_4[-1] * np.pi / 180]
        quat_4 = euler2quat(angle_4)

        return [pos_1, quat_1], [pos_2, quat_2], [pos_3, quat_3], [pos_4, quat_4]

    def set_context(self, context):
        red_pos = context[0][0]
        red_quat = context[0][1]

        green_pos = context[1][0]
        green_quat = context[1][1]

        blue_pos = context[2][0]
        blue_quat = context[2][1]

        target_pos = context[3][0]
        target_quat = context[3][1]

        self.scene.set_obj_pos_and_quat(
            [red_pos[0], red_pos[1], 0],
            red_quat,
            obj_name="red_box",
        )

        self.scene.set_obj_pos_and_quat(
            [green_pos[0], green_pos[1], 0],
            green_quat,
            obj_name="green_box",
        )

        self.scene.set_obj_pos_and_quat(
            [blue_pos[0], blue_pos[1], 0],
            blue_quat,
            obj_name="blue_box",
        )

    def set_index(self, index):
        self.index = index


class StackingEnv(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 30,
        max_steps_per_episode: int = 1000,
        debug: bool = False,
        random_env: bool = False,
        interactive: bool = False,
        render: bool = False,
        if_vision: bool = False,
    ):

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        robot = MjRobot(
            scene,
            xml_path=d3il_path("./models/mj/robot/panda_invisible.xml"),
        )
        controller = robot.jointTrackingController

        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.if_vision = if_vision

        self.action_space = Box(low=-0.01, high=0.01, shape=(8,))  # 7 joint + gripper
        if self.if_vision:
            self.observation_space = Dict(
                {
                    "bp-cam": Box(
                        low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
                    ),
                    "inhand-cam": Box(
                        low=0, high=255, shape=(96, 96, 3), dtype=np.uint8
                    ),
                    "proprio": Box(low=-np.inf, high=np.inf, shape=(8,)),
                }
            )
        else:
            self.observation_space = Dict(
                {
                    "state": Box(low=-np.inf, high=np.inf, shape=(20,)),
                }
            )

        self.interactive = interactive

        self.random_env = random_env
        self.manager = BlockContextManager(scene, index=0)

        self.bp_cam = BPCageCam()
        self.inhand_cam = robot.inhand_cam

        self.scene.add_object(self.bp_cam)

        self.red_box = obj_list[0]
        self.green_box = obj_list[1]
        self.blue_box = obj_list[2]
        self.target_box = obj_list[3]

        self.log_dict = {
            "red-box": ObjectLogger(scene, self.red_box),
            "green-box": ObjectLogger(scene, self.green_box),
            "blue-box": ObjectLogger(scene, self.blue_box),
            "target-box": ObjectLogger(scene, self.target_box),
        }

        self.cam_dict = {
            "bp-cam": CamLogger(scene, self.bp_cam),
            "inhand-cam": CamLogger(scene, self.inhand_cam),
        }

        for _, v in self.log_dict.items():
            scene.add_logger(v)

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.pos_min_dist = 0.06

        self.min_inds = []
        self.mode_encoding = []

        # Start simulation
        self.start()

    def robot_state(self):
        # Update Robot State
        self.robot.receiveState()

        # joint state
        joint_pos = self.robot.current_j_pos
        joint_vel = self.robot.current_j_vel
        gripper_width = np.array([self.robot.gripper_width])

        tcp_pos = self.robot.current_c_pos
        tcp_quad = self.robot.current_c_quat

        return np.concatenate((joint_pos, gripper_width)), joint_pos, tcp_quad

    def get_observation(self) -> dict[str, np.ndarray]:
        j_state, robot_c_pos, robot_c_quat = self.robot_state()

        if self.if_vision:
            bp_image = self.bp_cam.get_image(depth=False).copy()
            inhand_image = self.inhand_cam.get_image(depth=False).copy()
            return {
                "bp-cam": bp_image,
                "inhand-cam": inhand_image,
                "proprio": j_state.astype(np.float32),
            }

        red_box_pos = self.scene.get_obj_pos(self.red_box)
        red_box_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.red_box))[-1:])

        green_box_pos = self.scene.get_obj_pos(self.green_box)
        green_box_quat = np.tan(
            quat2euler(self.scene.get_obj_quat(self.green_box))[-1:]
        )

        blue_box_pos = self.scene.get_obj_pos(self.blue_box)
        blue_box_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.blue_box))[-1:])

        env_state = np.concatenate(
            [   
                j_state,
                red_box_pos,
                red_box_quat,
                green_box_pos,
                green_box_quat,
                blue_box_pos,
                blue_box_quat,
            ]
        )

        return {"state": env_state.astype(np.float32)}

    def start(self):
        self.scene.start()

        # reset view of the camera
        if self.scene.viewer is not None:
            self.scene.viewer.cam.elevation = -55
            self.scene.viewer.cam.distance = 2.0
            self.scene.viewer.cam.lookat[0] += 0
            self.scene.viewer.cam.lookat[2] -= 0.2

        # reset the initial state of the robot
        initial_cart_position = copy.deepcopy(init_end_eff_pos)
        self.robot.gotoCartPosQuatController.setDesiredPos(
            [
                initial_cart_position[0],
                initial_cart_position[1],
                initial_cart_position[2],
                0,
                1,
                0,
                0,
            ]
        )
        self.robot.gotoCartPosQuatController.initController(self.robot, 1)

        self.robot.init_qpos = self.robot.gotoCartPosQuatController.trajectory[
            -1
        ].copy()
        self.robot.init_tcp_pos = initial_cart_position
        self.robot.init_tcp_quat = [0, 1, 0, 0]

        self.robot.beam_to_joint_pos(
            self.robot.gotoCartPosQuatController.trajectory[-1]
        )

        self.robot.gotoCartPositionAndQuat(
            desiredPos=initial_cart_position,
            desiredQuat=[0, 1, 0, 0],
            duration=0.5,
            log=False,
        )

    def step(self, action, gripper_width=None, desired_vel=None, desired_acc=None):
        gripper_width = action[-1]
        if gripper_width > 0.075:
            self.robot.open_fingers()
        else:
            self.robot.close_fingers(duration=0.0)

        self.controller.setSetPoint(action[:-1])
        self.controller.executeControllerTimeSteps(
            self.robot, self.n_substeps, block=False
        )

        observation = self.get_observation()
        reward = self.get_reward()
        terminated = self.is_finished()
        truncated = False

        for i in range(self.n_substeps):
            self.scene.next_step()

        self.env_step_counter += 1

        self.success = self._check_early_termination()
        mode_encoding, mean_distance = self.check_mode()
        mode = "".join(mode_encoding)

        return (
            observation,
            reward,
            terminated,
            truncated,
            {
                "mode": mode,
                "success": self.success,
                "success_1": len(mode) > 0,
                "success_2": len(mode) > 1,
                "mean_distance": mean_distance,
            },
        )

    def check_mode(self):
        modes = ["r", "g", "b"]

        red_pos = self.scene.get_obj_pos(self.red_box)[:2]
        green_pos = self.scene.get_obj_pos(self.green_box)[:2]
        blue_pos = self.scene.get_obj_pos(self.blue_box)[:2]

        target_pos = self.scene.get_obj_pos(self.target_box)[:2]

        box_pos = np.vstack((red_pos, green_pos, blue_pos))

        dists = np.linalg.norm(box_pos - np.reshape(target_pos, (1, -1)), axis=-1)
        mean_dists = np.mean(dists)

        dists[self.min_inds] = 100000

        min_ind = np.argmin(dists)

        if dists[min_ind] <= self.pos_min_dist:

            self.mode_encoding.append(modes[min_ind])
            self.min_inds.append(min_ind)

        return self.mode_encoding, mean_dists

    def get_reward(self, if_sparse=False):
        return 0

    def _check_early_termination(self) -> bool:
        red_pos = self.scene.get_obj_pos(self.red_box)
        green_pos = self.scene.get_obj_pos(self.green_box)
        blue_pos = self.scene.get_obj_pos(self.blue_box)

        diff_z = min(
            [
                np.linalg.norm(red_pos[-1] - green_pos[-1]),
                np.linalg.norm(red_pos[-1] - blue_pos[-1]),
                np.linalg.norm(green_pos[-1] - blue_pos[-1]),
            ]
        )

        target_pos = self.scene.get_obj_pos(self.target_box)[:2]

        dis_rt, _ = obj_distance(red_pos[:2], target_pos)
        dis_gt, _ = obj_distance(green_pos[:2], target_pos)
        dis_bt, _ = obj_distance(blue_pos[:2], target_pos)

        if (
            (dis_rt <= self.pos_min_dist)
            and (dis_gt <= self.pos_min_dist)
            and (dis_bt <= self.pos_min_dist)
            and (diff_z > 0.03)
        ):
            # terminate if end effector is close enough
            self.terminated = True
            return True

        return False

    def reset(self, seed=None, options=None, random=True, context=None):
        self.seed(seed)
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1

        self.min_inds = []
        self.mode_encoding = []

        obs = self._reset_env(
            random=options.get("random", True), 
            context=options.get("context", None),
        )
        return obs, {}

    def _reset_env(self, random=True, context=None):
        self.scene.reset()
        self.robot.beam_to_joint_pos(self.robot.init_qpos)
        self.robot.open_fingers()
        self.manager.start(random=random, context=context)
        self.scene.next_step(log=False)
        return self.get_observation()
