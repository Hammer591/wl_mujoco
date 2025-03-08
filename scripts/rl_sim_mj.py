#!/usr/bin/env python3
import os
import time
import torch
import mujoco
import glfw
import numpy as np
from collections import deque
from mujoco import viewer

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

        # 初始化关节和执行器信息
        self._init_joint_info()
        
        # 加载策略网络
        policy_path_1 = os.path.join(os.path.dirname(__file__), '../policy/policy_1.pt')
        self.policy_model_1 = torch.jit.load(policy_path_1)
        self.policy_model_1.eval()

        policy_path_2 = os.path.join(os.path.dirname(__file__), '../policy/policy_2.pt')
        self.policy_model_2 = torch.jit.load(policy_path_2)
        self.policy_model_2.eval()

        # 观测配置
        self.obs_history = deque(maxlen=5)  # 存储最近5次观测
        self.control_rate = 1000  # Hz
        self.last_actions = np.zeros(6)     # 存储上次执行的动作

        # 状态参数
        self.gravity_vec = np.array([0, 0, -9.81])
        self.offset = 0
        self.l1 = 0.15
        self.l2 = 0.30
        self.pi = np.pi

        # 初始化 theta0, theta0_dot, L0 和 L0_dot
        self.theta0 = np.zeros(2)
        self.theta0_dot = np.zeros(2)
        self.L0 = np.zeros(2)
        self.L0_dot = np.zeros(2)

        # 运动指令
        self.commands = np.array([0, 0, 0])
        self.target_linear_velocity = 0.0  # 初始化目标线速度
        self.target_angular_vel = 0.0      # 初始化目标角速度
        self.target_z = 0.0                # 初始化目标z轴位置
        
        # 初始化GLFW窗口
        glfw.init()
        self.window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
        glfw.set_key_callback(self.window, self._key_callback)

    def compute_L0_theta0(self):
        """计算虚拟腿部参数"""
        self.l_theta1 = torch.tensor([self.data.sensor('lf0_Joint_p').data[0]])
        self.l_theta2 = torch.tensor([self.data.sensor('lf1_Joint_p').data[0] + self.pi/2])

        self.r_theta1 = torch.tensor([-self.data.sensor('rf0_Joint_p').data[0]])
        self.r_theta2 = torch.tensor([-self.data.sensor('rf1_Joint_p').data[0] + self.pi/2])
        
        self.l_theta1_dot = torch.tensor([self.data.sensor('lf0_Joint_v').data[0]])
        self.l_theta2_dot = torch.tensor([self.data.sensor('lf1_Joint_v').data[0]])
        self.r_theta1_dot = torch.tensor([-self.data.sensor('rf0_Joint_v').data[0]])
        self.r_theta2_dot = torch.tensor([-self.data.sensor('rf1_Joint_v').data[0]])

        self.theta1 = torch.stack(
            (self.l_theta1, self.r_theta1), dim=1
        )
        self.theta2 = torch.stack(
            (self.l_theta2, self.r_theta2), dim=1
        )
        theta1_dot = torch.stack(
            (self.l_theta1_dot, self.r_theta1_dot), dim=1
        )
        theta2_dot = torch.stack(
            (self.r_theta1_dot, self.r_theta2_dot), dim=1
        )

        self.L0, self.theta0 = self.forward_kinematics(self.theta1, self.theta2)

        dt = 0.001
        L0_temp, theta0_temp = self.forward_kinematics(
            self.theta1 + theta1_dot * dt, self.theta2 + theta2_dot * dt
        )
        self.L0_dot = (L0_temp - self.L0) / dt
        self.theta0_dot = (theta0_temp - self.theta0) / dt

    def forward_kinematics(self, theta1, theta2):
        """正运动学计算"""
        end_x = (
            self.offset
           + self.l1 * torch.cos(theta1)
           + self.l2 * torch.cos(theta1 + theta2)
        )   
        end_y = self.l1 * torch.sin(theta1) + self.l2 * torch.sin(theta1 + theta2)
        L0 = torch.sqrt(end_x**2 + end_y**2)
        theta0 = torch.arctan2(end_y, end_x) - self.pi / 2
        return L0.squeeze(), theta0.squeeze()  # 转换为一维数组

    def _init_joint_info(self):
        """初始化关节和执行器信息"""
        print("Total actuators:", self.mj_model.nu)
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Actuator {i}: {name}")


        self.motor_joint_names = [
            'lf0_Joint', 'lf1_Joint', 
            'rf0_Joint', 'rf1_Joint'
        ]
        self.wheel_joint_names = [
            'l_wheel_Joint', 'r_wheel_Joint'
        ]
        
        # 获取关节ID
        self.motor_joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.motor_joint_names
        ]
        self.wheel_joint_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.wheel_joint_names
        ]

        # 执行器顺序验证
        self.actuator_names = [
            'lf0_Joint', 'lf1_Joint', 'l_wheel_Joint',
            'rf0_Joint', 'rf1_Joint', 'r_wheel_Joint'
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.actuator_names
        ]
        print(f"Actuator IDs: {self.actuator_ids}")

    def _key_callback(self, window, key, scancode, action, mods):
        """键盘控制回调"""
        speed_value = 0.5 if action == glfw.PRESS else 0.0
        angular_value = 0.02 if action == glfw.PRESS else 0.0
        z_value = 0.02 if action == glfw.PRESS else 0.0
        
        if key == glfw.KEY_W:
            self.target_linear_velocity = speed_value    # 前进
        elif key == glfw.KEY_S:
            self.target_linear_velocity = -speed_value   # 后退
        elif key == glfw.KEY_A:
            self.target_angular_vel = angular_value  # 左转
        elif key == glfw.KEY_D:
            self.target_angular_vel = -angular_value # 右转
        elif key == glfw.KEY_J:
            self.target_z = z_value     # 上升
        elif key == glfw.KEY_K:
            self.target_z = -z_value    # 下降

    def _get_imu_data(self):
        """获取所有传感器数据"""
        return {
            'quaternion': self.data.sensor('orientation').data.copy(),
            'gyroscope': self.data.sensor('angular-velocity').data.copy(),
            'accelerometer': self.data.sensor('linear-acceleration').data.copy(),
            'motor_positions': [
                self.data.sensor('lf0_Joint_p').data[0],
                self.data.sensor('lf1_Joint_p').data[0],
                self.data.sensor('rf0_Joint_p').data[0],
                self.data.sensor('rf1_Joint_p').data[0]
            ],
            'motor_velocities': [
                self.data.sensor('lf0_Joint_v').data[0],
                self.data.sensor('lf1_Joint_v').data[0],
                self.data.sensor('rf0_Joint_v').data[0],
                self.data.sensor('rf1_Joint_v').data[0]
            ],
            'wheel_positions': [
                self.data.sensor('l_wheel_Joint_p').data[0],
                self.data.sensor('r_wheel_Joint_p').data[0]
            ],
            'wheel_velocities': [
                self.data.sensor('l_wheel_Joint_v').data[0],
                self.data.sensor('r_wheel_Joint_v').data[0]
            ]
        }
    
    @staticmethod
    def quat_rotate_inverse(q, v):
        if isinstance(q, np.ndarray):
            q = torch.tensor(q)
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c
    
    def _get_obs(self):
        """生成当前观测向量（27维）"""
        imu = self._get_imu_data()
        
        self.base_quat = torch.tensor(imu['quaternion'])
        self.base_ang_vel = torch.tensor(imu['gyroscope'])
        self.base_acc = torch.tensor(imu['accelerometer'])

        self.angular_velocity = self.quat_rotate_inverse(
            self.base_quat, self.base_ang_vel
        )

        self.projected_gravity = self.quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        
        self.target_linear_vel_scale = 1
        self.target_z_scale = 1
        self.target_angular_vel_scale = 1

        # 控制指令 [vx, z, wz]
        self.commands = np.array(
            [
                self.target_linear_velocity * self.target_linear_vel_scale ,
                self.target_angular_vel * self.target_angular_vel_scale,
                self.target_z * self.target_z_scale,
            ]
        )
        
        # 轮毂状态
        self.wheel_pos = torch.tensor(
            [
                self.data.sensor('l_wheel_Joint_p').data[0], 
                self.data.sensor('r_wheel_Joint_p').data[0]
            ]
        )  

        self.wheel_vel = torch.tensor(
            [
                self.data.sensor('l_wheel_Joint_v').data[0], 
                self.data.sensor('r_wheel_Joint_v').data[0]
            ]
        )
        
        # 拼接观测向量
        obs = np.concatenate([
            self.angular_velocity.numpy().flatten(),  # 3
            self.gravity_vec.flatten(),               # 3
            self.commands.flatten(),                  # 3
            self.theta0.flatten(),                    # 2
            self.theta0_dot.flatten(),                # 2
            self.L0.flatten(),                        # 2
            self.L0_dot.flatten(),                    # 2    
            self.wheel_pos.numpy().flatten(),         # 2
            self.wheel_vel.numpy().flatten(),         # 2
            self.last_actions.flatten()               # 6
        ])
        return obs.astype(np.float32)  # 27维
    
    def _get_obs_history(self):
        """获取历史观测（27*5）"""
        history = []
        missing_steps = 5 - len(self.obs_history)
        
        # 填充缺失的历史观测
        for _ in range(missing_steps):
            history.append(np.zeros(27, dtype=np.float32))  # 修改为27维
        
        # 添加实际历史数据
        history.extend(list(self.obs_history))
        
        return np.concatenate(history)

    def _apply_control(self, output):
        """应用控制信号到执行器"""
        # 电机位置控制
        # motor_pos = torch.clamp(output[:4], -3.14, 3.14).tolist()
        # self.data.ctrl[0] = motor_pos[0]  # lf0
        # self.data.ctrl[1] = motor_pos[1]  # rf0
        # self.data.ctrl[2] = motor_pos[2]  # lf1
        # self.data.ctrl[4] = motor_pos[3]  # rf1
        
        # 轮毂速度控制
        # wheel_vel = torch.clamp(output[4:6], -5, 5).tolist()
        # self.data.ctrl[2] = wheel_vel[0]  # 左轮
        # self.data.ctrl[5] = wheel_vel[1]  # 右轮
        
        print(f"output: {output}")

        self.data.ctrl[0] = 0  # lf0
        self.data.ctrl[1] = 1.5  # lf1
        self.data.ctrl[3] = 0  # rf0
        self.data.ctrl[4] = 0  # rf1
        self.data.ctrl[2] = 0  # 左轮
        self.data.ctrl[5] = 0  # 右轮

        # 保存当前动作
        self.last_actions = output.detach().numpy().copy()

    def run(self):
        """主运行循环"""
        last_time = time.time()
        
        # 初始观测
        self.compute_L0_theta0()  # 计算初始 theta0 和 theta0_dot
        init_obs = self._get_obs()
        self.obs_history.append(init_obs)
        # print(self.obs_history)
        step = 0
        
        with mujoco.viewer.launch_passive(self.mj_model, self.data) as viewer:     
            # 获取摄像头ID并绑定
            camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            viewer.cam.fixedcamid = camera_id  # 指定摄像头ID
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # 固定摄像头模式
            while viewer.is_running():       # 控制频率保持
                dt = 1.0 / self.control_rate

                # 渲染
                step = step + 1
                step = step % 10
                if step == 1:
                    # 获取当前观测
                    self.compute_L0_theta0()  # 更新 theta0 和 theta0_dot
                    current_obs = torch.tensor(self._get_obs()).unsqueeze(0)  # 转换为 Tensor 并添加维度
                    # print(f"current_obs shape: {current_obs.shape}")  # 打印 current_obs 的形状
                    # torch.set_printoptions(threshold=torch.inf)
                    # print("obs:", current_obs)

                    # 获取历史观测（用于策略网络输入）
                    obs_his = torch.tensor(self._get_obs_history()).unsqueeze(0)  # 转换为 Tensor 并添加维度
                    # print(f"obs_his shape: {obs_his.shape}")  # 打印 obs_his 的形状
                    
                    # test_obs = torch.tensor(np.zeros(27, dtype=np.float32)).unsqueeze(0)
                    # current_obs = test_obs

                    # 策略推理
                    with torch.no_grad():
                        output_2 = self.policy_model_2(obs_his)
                        # print(f"output_2 shape: {output_2.shape}")  # 打印 output_2 的形状
                        combined_input = torch.cat([current_obs, output_2], dim=1)  # 拼接输入
                        # print(f"combined_input shape: {combined_input.shape}")  # 打印 combined_input 的形状
                        output_1 = self.policy_model_1(combined_input).squeeze(0)  # 确保 output_1 是一维张量
                        # print(output_1)
                
                    # self._apply_control(output_1)
                    # print(output_1.shape)
                    # print(combined_input.shape)
                self.data.ctrl[0] = 0  # lf0
                self.data.ctrl[1] = 1  # lf1
                self.data.ctrl[3] = 0  # rf0
                self.data.ctrl[4] = 0  # rf1
                self.data.ctrl[2] = 10  # 左轮
                self.data.ctrl[5] = -10  # 右轮
                mujoco.mj_step(self.mj_model, self.data) 
                print(f"lf0: {self.data.ctrl[0]}, lf1: {self.data.ctrl[1]}, rf0: {self.data.ctrl[3]}, rf1: {self.data.ctrl[4]}, left: {self.data.ctrl[2]}, right: {self.data.ctrl[5]}")
                # 渲染
                viewer.sync()
                while time.time() - last_time < dt:
                    time.sleep(0.001)
                last_time = time.time()

            # glfw.poll_events()

        glfw.terminate()

if __name__ == '__main__':
    sim = MuJoCoControl()
    sim.run()
