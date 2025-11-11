import mujoco
import pickle
import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.spatial.transform import Rotation, Slerp
import argparse
from playground.common.onnx_infer import OnnxInfer
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion
from playground.common.utils import LowPassActionFilter

from playground.open_duck_mini_v2.mujoco_infer_base import MJInferBase

USE_MOTOR_SPEED_LIMITS = True


class MjInfer(MJInferBase):
    def __init__(
        self, model_path: str, reference_data: str, onnx_model_path: str, standing: bool
    ):
        super().__init__(model_path)

        self.standing = standing
        self.head_control_mode = self.standing

        # Params
        self.linearVelocityScale = 1.0
        self.angularVelocityScale = 1.0
        self.dof_pos_scale = 1.0
        self.dof_vel_scale = 0.05
        self.action_scale = 0.25

        self.action_filter = LowPassActionFilter(50, cutoff_frequency=37.5)

        if not self.standing:
            self.PRM = PolyReferenceMotion(reference_data)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.COMMANDS_RANGE_X = [-0.15, 0.15]
        self.COMMANDS_RANGE_Y = [-0.2, 0.2]
        self.COMMANDS_RANGE_THETA = [-1.0, 1.0]  # [-1.0, 1.0]

        self.NECK_PITCH_RANGE = [-0.34, 1.1]
        self.HEAD_PITCH_RANGE = [-0.78, 0.78]
        self.HEAD_YAW_RANGE = [-1.5, 1.5]
        self.HEAD_ROLL_RANGE = [-0.5, 0.5]

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)
        self.commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.saved_obs = []

        self.max_motor_velocity = 5.24  # rad/s

        self.phase_frequency_factor = 1.0

        self.K_DS = 1.0

        self.curr_t = [0, 0, 0]
        self.curr_q = [0, 0, 0]

        print(f"joint names: {self.joint_names}")
        print(f"actuator names: {self.actuator_names}")
        print(f"backlash joint names: {self.backlash_joint_names}")
        # print(f"actual joints idx: {self.get_actual_joints_idx()}")

    def get_obs(
        self,
        data,
        command,  # , qvel_history, qpos_error_history, gravity_history
    ):
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        accelerometer[0] += 1.3

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        # if not self.standing:
        # ref = self.PRM.get_reference_motion(*command[:3], self.imitation_i)

        obs = np.concatenate(
            [
                gyro,
                accelerometer,
                # gravity,
                command,
                joint_angles - self.default_actuator,
                joint_vel * self.dof_vel_scale,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                # ref if not self.standing else np.array([]),
                # [self.imitation_i]
                self.imitation_phase,
            ]
        )

        return obs

    def quat_DS(self, q, q_des):
        # Expect q and q_des to be 4-element quaternions in the form [w, x, y, z]
        q = np.array(q, dtype=float)
        q_des = np.array(q_des, dtype=float)

        # Normalize inputs to avoid drift/scale issues
        if np.linalg.norm(q) == 0 or np.linalg.norm(q_des) == 0:
            return np.zeros(3)
        q = q / np.linalg.norm(q)
        q_des = q_des / np.linalg.norm(q_des)

        # Decompose into scalar (s) and vector (u) parts assuming q = [s, x, y, z]
        s1, u1 = q[0], np.array(q[1:4])
        s2, u2 = q_des[0], np.array(q_des[1:4])

        # skew-symmetric matrix of u1
        Su1 = np.array([[0, -u1[2], u1[1]],
                        [u1[2], 0, -u1[0]],
                        [-u1[1], u1[0], 0]])

        # quaternion composition error (dori = q * conj(q_des) or similar depending on convention)
        dori = np.array([s1 * s2 + u1 @ u2.T,
                         *(-s1 * u2 + s2 * u1 - Su1 @ u2)])

        v = np.array(dori[1:])
        v_norm = np.linalg.norm(v)

        # numeric safety for arccos domain
        dori[0] = np.clip(dori[0], -1.0, 1.0)

        # If the vector part is (almost) zero, the rotation difference is negligible
        if v_norm <= 1e-6:
            return np.zeros(3)

        # log map of quaternion to axis-angle-like vector (rotation vector)
        angle = np.arccos(dori[0])
        logdq = angle * (v / v_norm)

        orientation_error_rad = np.linalg.norm(logdq)
        orientation_error_deg = np.degrees(orientation_error_rad)
        # use print here (no rospy in this module)
        print(f"Orientation error: {orientation_error_rad:.4f} rad | {orientation_error_deg:.2f} deg")

        # Return angular velocity command (drive toward desired orientation)
        return -self.K_DS * logdq

    def linear_DS(self, x_des, y_des):
        curr_x = self.curr_t[0]
        curr_y = self.curr_t[1]
        # DS is a simple v = -k(x_curr-x_des)
        x_vel = -self.K_DS * (curr_x - x_des)
        y_vel = -self.K_DS * (curr_y - y_des)
        # Clip it to the command value ranges
        x_vel = np.clip(x_vel, self.COMMANDS_RANGE_X[0], self.COMMANDS_RANGE_X[1])
        y_vel = np.clip(y_vel, self.COMMANDS_RANGE_Y[0], self.COMMANDS_RANGE_Y[1])
        return x_vel, y_vel

    def pos_callback(self):
        # get_floating_base_qpos expects a qpos array, not the full MjData object
        full_position = self.get_floating_base_qpos(self.data.qpos)
        self.curr_t = full_position[:3]
        self.curr_q = full_position[3:]

    def set_target_position(self, x_des, y_des):
        """Set the target position for the DS control"""
        x_vel, y_vel = self.linear_DS(x_des, y_des)
        self.commands[0] = x_vel
        self.commands[1] = y_vel
        self.commands[2] = 0.0  # No angular velocity for now


    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False
            ) as viewer:
                counter = 0
                while True:

                    step_start = time.time()

                    mujoco.mj_step(self.model, self.data)
                    
                    counter += 1

                    self.pos_callback()
                    # Update target position using DS
                    self.set_target_position(1.0, 0.5)  # Example fixed target
                    print("translation")
                    print(self.curr_t)
                    print("rotation")
                    print(self.curr_q)

                    if counter % self.decimation == 0:
                        if not self.standing:
                            self.imitation_i += 1.0 * self.phase_frequency_factor
                            self.imitation_i = (
                                self.imitation_i % self.PRM.nb_steps_in_period
                            )
                            # print(self.PRM.nb_steps_in_period)
                            # exit()
                            self.imitation_phase = np.array(
                                [
                                    np.cos(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                    np.sin(
                                        self.imitation_i
                                        / self.PRM.nb_steps_in_period
                                        * 2
                                        * np.pi
                                    ),
                                ]
                            )
                        obs = self.get_obs(
                            self.data,
                            self.commands,
                        )
                        self.saved_obs.append(obs)
                        action = self.policy.infer(obs)

                        # self.action_filter.push(action)
                        # action = self.action_filter.get_filtered_action()

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * self.action_scale
                        )

                        if USE_MOTOR_SPEED_LIMITS:
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets
                                - self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                                self.prev_motor_targets
                                + self.max_motor_velocity
                                * (self.sim_dt * self.decimation),
                            )

                            self.prev_motor_targets = self.motor_targets.copy()

                        # head_targets = self.commands[3:]
                        # self.motor_targets[5:9] = head_targets
                        self.data.ctrl = self.motor_targets.copy()

                    viewer.sync()

                    time_until_next_step = self.model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pickle.dump(self.saved_obs, open("mujoco_saved_obs.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    # parser.add_argument("-k", action="store_true", default=False)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)

    args = parser.parse_args()

    mjinfer = MjInfer(
        args.model_path, args.reference_data, args.onnx_model_path, args.standing
    )
    mjinfer.run()
