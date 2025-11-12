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

        self.COMMANDS_RANGE_X = [-0.2, 0.2]
        self.COMMANDS_RANGE_Y = [-0.15, 0.15]
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

        self.K_DS = 3.0

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

    def quat_DS(self, q, theta_des_deg):
        # Expect q as quaternion [w, x, y, z]
        # Expect q_des_deg as desired yaw in degrees
        q = np.array(q, dtype=float)

        # Normalize quaternion
        if np.linalg.norm(q) == 0:
            return 0.0
        q = q / np.linalg.norm(q)

        # Decompose quaternion (w, x, y, z)
        w, x, y, z = q

        # Compute current yaw in degrees
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_curr_deg = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

        # Desired yaw in degrees
        yaw_des_deg = float(theta_des_deg)

        # Compute shortest angle difference in degrees [-180, 180]
        diff_deg = yaw_des_deg - yaw_curr_deg
        diff_deg = (diff_deg + 180) % 360 - 180

        # Proportional DS on yaw error
        yaw_cmd = self.K_DS * diff_deg

        # Clip to allowed command range (assumed in degrees)
        yaw_cmd = np.clip(yaw_cmd, self.COMMANDS_RANGE_THETA[0], self.COMMANDS_RANGE_THETA[1])

        print(f"Yaw error: {diff_deg:.4f} deg -> yaw_cmd {yaw_cmd:.4f}")

        return float(yaw_cmd)

    def linear_DS(self, x_des, y_des):
        curr_x = self.curr_t[0]
        curr_y = self.curr_t[1]
        # DS is a simple v = -k(x_curr-x_des)
        x_vel = -self.K_DS * (curr_x - x_des)
        y_vel = -self.K_DS * (curr_y - y_des)
        return x_vel, y_vel

    def pos_callback(self):
        # get_floating_base_qpos expects a qpos array, not the full MjData object
        full_position = self.get_floating_base_qpos(self.data.qpos)
        self.curr_t = full_position[:3]
        self.curr_q = full_position[3:]

    def set_target_position(self, x_des, y_des, theta_des=None):

        # --- Linear DS control (in world frame) ---
        x_vel_world, y_vel_world = self.linear_DS(x_des, y_des)

        # --- Get current orientation (to transform into body frame) ---
        q_curr = np.array(self.curr_q, dtype=float)
        if np.linalg.norm(q_curr) == 0:
            yaw_curr_deg = 0.0
        else:
            q_curr /= np.linalg.norm(q_curr)
            w, x, y, z = q_curr
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw_curr_deg = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

        yaw_curr_rad = np.radians(yaw_curr_deg)

        # --- Rotate world velocities into the robot (body) frame ---
        v_body_x =  np.cos(yaw_curr_rad) * x_vel_world + np.sin(yaw_curr_rad) * y_vel_world
        v_body_y = -np.sin(yaw_curr_rad) * x_vel_world + np.cos(yaw_curr_rad) * y_vel_world

        v_body_x = np.clip(
            v_body_x, self.COMMANDS_RANGE_X[0], self.COMMANDS_RANGE_X[1]
        )
        v_body_y = np.clip(
            v_body_y, self.COMMANDS_RANGE_Y[0], self.COMMANDS_RANGE_Y[1]
        )
        # --- Assign to command vector (body frame: forward, sideways) ---
        self.commands[0] = float(v_body_x)  # forward/backward
        self.commands[1] = float(v_body_y)  # sideways
        print("body frame commands:")
        print(f"  forward/backward: {self.commands[0]:.4f}")
        print(f"  sideways: {self.commands[1]:.4f}")
        # --- Orientation DS (yaw control) ---
        if theta_des is not None:
            yaw_cmd = self.quat_DS(q_curr, theta_des)
            self.commands[2] = float(yaw_cmd)
        else:
            self.commands[2] = 0.0

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
                    self.set_target_position(1.0, 2.5, 90.0)  # Example fixed target
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
