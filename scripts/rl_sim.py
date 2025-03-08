#!/usr/bin/env python3
import os
import time
import torch
import mujoco
import glfw
import numpy as np
from observation_buffer import ObservationBuffer

class MuJoCoControl:
    """MuJoCo仿真控制主类"""
    def __init__(self):
        # 初始化MuJoCo环境
        self.model_path = os.path.join(
            os.path.dirname(__file__), 
            '../urdf/wl_mjcf.xml'
        )
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.mj_model)
        self.viewer = None

        # 加载策略网络
        policy_path = os.path.join(os.path.dirname(__file__), '../policy/policy_1.pt')
        self.policy_model = torch.jit.load(policy_path)
        self.policy_model.eval()

        # 修改观测缓冲区参数
        self.num_obs = 10  # 四元数4+陀螺仪3+加速度计3
        self.history_steps = 3
        self.obs_buffer = ObservationBuffer(
            num_envs=1,
            num_obs=self.num_obs,
            include_history_steps=self.history_steps
        )
        self.control_rate = 30  # Hz

        # 新增速度指令变量
        self.target_velocity = np.zeros(3)      # [vx, vy, vz]
        self.target_angular_vel = np.zeros(3)   # [wx, wy, wz]
        
        # 初始化GLFW窗口
        glfw.init()
        self.window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
        glfw.set_key_callback(self.window, self._key_callback)

    def _key_callback(self, window, key, scancode, action, mods):
        """键盘回调"""
        # 清除原有键位状态逻辑
        # 新增速度指令设置
        speed_value = 0.5 if action == glfw.PRESS else 0.0
        angular_value = 0.2 if action == glfw.PRESS else 0.0
        
        if key == glfw.KEY_W:
            self.target_velocity[0] = speed_value   # X+方向速度
        elif key == glfw.KEY_S:
            self.target_velocity[0] = -speed_value  # X-方向速度
        elif key == glfw.KEY_A:
            self.target_angular_vel[2] = angular_value  # Z+方向角速度
        elif key == glfw.KEY_D:
            self.target_angular_vel[2] = -angular_value # Z-方向角速度

    def _get_imu_data(self):
        """IMU数据获取"""
        return {
            'quaternion': self.data.sensor('orientation').data.copy(),
            'gyroscope': self.data.sensor('angular-velocity').data.copy(),
            'accelerometer': self.data.sensor('linear-acceleration').data.copy()
        }

    def _get_current_observation(self):
        """修改观测生成"""
        imu = self._get_imu_data()
        # 仅返回IMU数据（10维）
        return torch.FloatTensor(np.concatenate([
            imu['quaternion'],
            imu['gyroscope'],
            imu['accelerometer']
        ]))

    def _apply_control(self, output):
        """保持原有的执行器控制"""
        # 髋关节位置控制（前4个执行器）
        hip_pos = torch.clamp(output[0, :4], -3.14, 3.14).tolist()
        for i in range(4):
            self.data.ctrl[i] = hip_pos[i]
        
        # 轮子速度控制（后2个执行器）
        wheel_vel = torch.clamp(output[0, 4:6], -30, 30).tolist()
        self.data.ctrl[4] = wheel_vel[0]
        self.data.ctrl[5] = wheel_vel[1]

    def run(self):
        """修改后的主循环"""
        last_time = time.time()
        init_obs = self._get_current_observation().unsqueeze(0)
        self.obs_buffer.reset(torch.tensor([0]), init_obs)

        while not glfw.window_should_close(self.window):
            # 保持原有的频率控制
            dt = 1.0 / self.control_rate
            while time.time() - last_time < dt:
                time.sleep(0.001)
            last_time = time.time()

            mujoco.mj_step(self.mj_model, self.data)

            # 获取并处理观测数据
            current_obs = self._get_current_observation()
            self.obs_buffer.insert(current_obs.unsqueeze(0))
            
            # 构建36维输入
            history_obs = self.obs_buffer.get_obs_vec([0, 1, 2])  # 3x10=30
            speed_command = torch.FloatTensor(
                np.concatenate([self.target_velocity, self.target_angular_vel])
            ).unsqueeze(0)
            combined_input = torch.cat([speed_command, history_obs], dim=1)  # 30+6=36
            
            with torch.no_grad():
                output = self.policy_model(combined_input)  # 输入维度修正
            
            self._apply_control(output)

            # 保持原有的渲染逻辑
            if self.viewer is None:
                self.viewer = mujoco.MjViewer(self.mj_model, self.data)
            self.viewer.render()

            glfw.poll_events()

        glfw.terminate()

if __name__ == '__main__':
    sim = MuJoCoControl()
    sim.run()
