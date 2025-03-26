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
        self.control_rate = 100  # Hz
        self.last_actions = np.zeros(6)     # 存储上次执行的动作

        # 状态参数
        self.gravity_vec = np.array([0, 0, -1.0])
        self.offset = 0.054
        self.l1 = 0.15
        self.l2 = 0.25
        self.pi = np.pi
        self.vel_err_intergral = 0.0

        # 初始化 theta0, theta0_dot, L0 和 L0_dot
        self.theta0 = np.zeros(2)
        self.theta0_dot = np.zeros(2)
        self.L0 = np.zeros(2)
        self.L0_dot = np.zeros(2)

        # 运动指令
        self.commands = np.array([0, 0, 0])
        self.target_linear_velocity = 0.0  # 初始化目标线速度
        self.target_angular_vel = 0.0      # 初始化目标角速度
        self.target_z = 0.18               # 初始化目标z轴位置

        # self.data.qpos[self.mj_model.jnt_qposadr[1]] = 0.5
        # self.data.qpos[self.mj_model.jnt_qposadr[2]] = 0.35
        # self.data.qpos[self.mj_model.jnt_qposadr[3]] = 0.0
        # self.data.qpos[self.mj_model.jnt_qposadr[4]] = -0.5
        # self.data.qpos[self.mj_model.jnt_qposadr[5]] = -0.35
        # self.data.qpos[self.mj_model.jnt_qposadr[6]] = 0.0
        
        # 初始化 d_gains
        self.d_gains = torch.tensor(np.zeros((1, 6)))
        
        self._init_joint_positions()
        # 初始化GLFW窗口
        glfw.init()
        self.window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
        glfw.set_key_callback(self.window, self._key_callback)

    def _init_joint_positions(self):
        """初始化所有关节的初始角度"""
        # 初始化root位置
        root_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "root")
        if root_id != -1:
            self.data.qpos[0:3] = np.array([0, 0, 0.20])
        else:
            print("未找到root body")

        # 定义需要初始化的关节及其目标角度（单位：弧度）
        joint_init_angles = {
            "lf0_Joint": 0.5,
            "lf1_Joint": 0.35,
            "l_wheel_Joint": 0.0,
            "rf0_Joint": -0.5,
            "rf1_Joint": -0.35,
            "r_wheel_Joint": 0.0,
        }

        # 遍历所有关节进行初始化
        for joint_name, angle in joint_init_angles.items():
            # 获取关节ID
            joint_id = mujoco.mj_name2id(
                self.mj_model, 
                mujoco.mjtObj.mjOBJ_JOINT, 
                joint_name
            )
            
            if joint_id != -1:
                # 获取该关节在qpos数组中的地址
                qpos_addr = self.mj_model.jnt_qposadr[joint_id]
                
                # 根据关节类型设置初始值
                joint_type = self.mj_model.jnt_type[joint_id]
                
                if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    # 旋转关节直接设置角度
                    self.data.qpos[qpos_addr] = angle
                elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    # 滑动关节设置位置（如有需要）
                    self.data.qpos[qpos_addr] = angle
                else:
                    print(f"未知关节类型: {joint_name}")
            else:
                print(f"未找到关节: {joint_name}")

        # 使初始位置生效
        mujoco.mj_forward(self.mj_model, self.data)
    
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

        # print(f"L0: {L0}, theta0: {theta0}")
        # print(f"theta1: {theta1}, theta2: {theta2}")
        return L0.squeeze(), theta0.squeeze()  # 转换为一维数组
        

    def _init_joint_info(self):
        """初始化关节和执行器信息"""
        # print("Total actuators:", self.mj_model.nu)
        # for i in range(self.mj_model.nu):
        #     name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        #     print(f"Actuator {i}: {name}")


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
        # print(f"Actuator IDs: {self.actuator_ids}")

    def _key_callback(self, window, key, scancode, action, mods):
        """键盘控制回调"""
        speed_value = 0.5 if action == glfw.PRESS else 0.0
        angular_value = 0.02 if action == glfw.PRESS else 0.0
        z_value = 0.02 if action == glfw.PRESS else 0.0
        
        if key == glfw.KEY_W:
            self.target_linear_velocity += speed_value    # 前进
        elif key == glfw.KEY_S:
            self.target_linear_velocity -= speed_value   # 后退
        elif key == glfw.KEY_A:
            self.target_angular_vel += angular_value  # 左转
        elif key == glfw.KEY_D:
            self.target_angular_vel -= -angular_value # 右转
        elif key == glfw.KEY_J:
            self.target_z += z_value     # 上升
        elif key == glfw.KEY_K:
            self.target_z -= -z_value    # 下降

    def _get_imu_data(self):
        """获取所有传感器数据"""
        return {
            'quaternion': self.data.sensor('torso_quat').data.copy(),
            'gyroscope': self.data.sensor('torso_angvel').data.copy(),
            'accelerometer': self.data.sensor('torso_linacc').data.copy(),
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
        self.torso_pos = self.data.sensordata[0:3]     # 位置 (x, y, z)
        # print("torso_pos",self.torso_pos)
        self.base_quat = self.data.sensordata[[4,5,6,3]]    # 四元数 (x, y, z, w)
        # print("base_quat",self.base_quat)
        self.base_linvel = self.data.sensordata[7:10] # 线速度 (vx, vy, vz)
        print("base_linvel",self.base_linvel)
        self.base_ang_vel = self.data.sensordata[10:13] # 角速度 (wx, wy, wz)
        self.base_acc = self.data.sensordata[13:16] # 线加速度 (ax, ay, az)

        self.angular_velocity = self.quat_rotate_inverse(
            self.base_quat, self.base_ang_vel
        )
        # self.angular_velocity = torch.tensor(self.base_ang_vel)
        # print("angular_velocity",self.angular_velocity)

        self.projected_gravity = self.quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )
        # self.projected_gravity = torch.tensor(self.gravity_vec)

        self.base_lin_vel = self.quat_rotate_inverse(
            self.base_quat, self.base_linvel
        )
        print("base_lin_vel",self.base_lin_vel)
        
        self.target_linear_vel_scale = 2.0
        self.target_z_scale = 5.0
        self.target_angular_vel_scale = 0.25

        # self.target_linear_velocity = 0.5

        # 将 self.target_linear_velocity 和 self.base_linvel[0] 转换为 torch 张量
        target_linear_velocity_tensor = torch.tensor(self.target_linear_velocity)
        base_linvel_tensor = self.base_lin_vel.flatten()[0]
        # print("base_linvel_tensor",base_linvel_tensor)

        self.vel_err_intergral += (
            (target_linear_velocity_tensor - base_linvel_tensor)
            * 0.001
            * ((target_linear_velocity_tensor - base_linvel_tensor).abs() < 0.5)
        )
        # print("vel_err_intergral",vel_err_intergral)
        self.vel_err_intergral = torch.clip(self.vel_err_intergral, -0.5, 0.5)
        vel_cmd = target_linear_velocity_tensor + self.vel_err_intergral

        # 控制指令 [vx, wz, z]
        self.commands = np.array(
            [
                vel_cmd.item() * self.target_linear_vel_scale,
                # self.target_linear_velocity * self.target_linear_vel_scale,
                self.target_angular_vel * self.target_angular_vel_scale,
                self.target_z * self.target_z_scale,
            ]
        )
        # print("commands",self.commands)
        
        # 轮毂状态
        self.wheel_pos = torch.tensor(
            [
                (self.data.sensor('l_wheel_Joint_p').data[0] + 2 * self.pi) % (4 * self.pi) - 2 * self.pi, 
                (self.data.sensor('r_wheel_Joint_p').data[0] + 2 * self.pi) % (4 * self.pi) - 2 * self.pi
            ]
        )  

        self.wheel_vel = torch.tensor(
            [
                self.data.sensor('l_wheel_Joint_v').data[0], 
                self.data.sensor('r_wheel_Joint_v').data[0]
            ]
        )
        
        angular_vel_scale = 0.25
        L0_scale = 5.0
        L0_dot_scale = 0.25
        dof_pos_scale = 1.0
        dof_vel_scale = 0.05
        clip_obs = 100.0
        # 拼接观测向量
        # print("theta0",self.theta0)
        obs = np.concatenate([
            self.angular_velocity.numpy().flatten() * angular_vel_scale,    # 3
            self.projected_gravity.flatten(),                               # 3
            self.commands.flatten(),                                        # 3
            self.theta0.flatten() * dof_pos_scale,                          # 2
            self.theta0_dot.flatten() * dof_vel_scale,                      # 2
            self.L0.flatten() * L0_scale,                                   # 2
            self.L0_dot.flatten() * L0_dot_scale,                           # 2    
            self.wheel_pos.numpy().flatten() * dof_pos_scale,               # 2
            self.wheel_vel.numpy().flatten() * dof_vel_scale,               # 2
            self.last_actions.flatten()                                     # 6
        ])
        obs = torch.tensor(obs)
        obs = torch.clip(obs, -clip_obs, clip_obs)
        return obs.numpy().astype(np.float32)                                      # 27维
    
    def _init_obs_history(self):
        """获取历史观测（27*5）"""
        history = []
        missing_steps = 5
        # print("missing_steps",missing_steps)
        obs_zero = torch.tensor([
            0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
            -1.0000e+00, 0.0000e+00,  0.0000e+00,  9.0000e-01,  0.0000e+00,
            0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
            0.0000e+00,  0.0000e+00])

        # 填充缺失的历史观测
        for _ in range(missing_steps):
            history.append(obs_zero)  
        
        # 添加实际历史数据
        history.extend(list(self.obs_history))

        return np.concatenate(history)
    
    def _get_obs_history(self):
        """获取历史观测（27*5）"""
        history = []
        history.append(self.obs_his.flatten()[27:135])
        history.append(self.obs.flatten())
        self.obs_his = history
        return np.concatenate(self.obs_his)

    def VMC(self, F, T):
        theta0 = self.theta0 + self.pi / 2
        t11 = self.l1 * torch.sin(
            theta0 - self.theta1
        ) - self.l2 * torch.sin(self.theta1 + self.theta2 - theta0)

        t12 = self.l1 * torch.cos(
            theta0 - self.theta1
        ) - self.l2 * torch.cos(self.theta1 + self.theta2 - theta0)
        t12 = t12 / self.L0

        t21 = -self.l2 * torch.sin(self.theta1 + self.theta2 - theta0)

        t22 = -self.l2 * torch.cos(self.theta1 + self.theta2 - theta0)
        t22 = t22 / self.L0

        T1 = t11 * F - t12 * T
        T2 = t21 * F - t22 * T

        return T1, T2
    
    def _apply_control(self, actions):
        """应用控制信号到执行器"""
        # print(actions)
        
        self.action_scale_theta = 0.5
        self.action_scale_l0 = 0.1
        self.action_scale_vel = 10.0

        self.l0_offset = 0.175
        self.feedforward_force = 40.0  # [N]

        self.kp_theta = 50.0  # [N*m/rad]
        self.kd_theta = 3.0  # [N*m*s/rad]
        self.kp_l0 = 900.0  # [N/m]
        self.kd_l0 = 20.0  # [N*s/m]

        theta0_ref = (
            torch.cat(
                (
                    (actions[:, 0]).unsqueeze(1),
                    (actions[:, 3]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.action_scale_theta
        )
        l0_ref = (
            torch.cat(
                (
                    (actions[:, 1]).unsqueeze(1),
                    (actions[:, 4]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.action_scale_l0
        ) + self.l0_offset
        wheel_vel_ref = (
            torch.cat(
                (
                    (actions[:, 2]).unsqueeze(1),
                    (actions[:, 5]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.action_scale_vel
        )

        self.torque_leg = (
            self.kp_theta * (theta0_ref - self.theta0) - self.kd_theta * self.theta0_dot
        )
        self.force_leg = self.kp_l0 * (l0_ref - self.L0) - self.kd_l0 * self.L0_dot
        # print(f"force_leg: {self.force_leg}")
        # print(f"force_leg: {self.force_leg + self.feedforward_force}")
        # self.d_gains[:, 2] = 0.5
        # self.d_gains[:, 5] = 0.5
        self.torque_wheel = 0.5 * (
            wheel_vel_ref - self.wheel_vel
        )
        T1, T2 = self.VMC(
            self.force_leg + self.feedforward_force, self.torque_leg
        )
        T1 = torch.clip(T1, -30, 30)
        T2 = torch.clip(T2, -30, 30)
        self.torque_wheel = torch.clip(self.torque_wheel, -5, 5)
        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                T2[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                -T1[:, 1].unsqueeze(1),
                -T2[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )

        # print(f"torques: {torques}")

        k = 1
        # 电机控制
        self.data.ctrl[0] = torques[0, 0].item() * k  # lf0
        self.data.ctrl[1] = torques[0, 1].item() * k  # lf1
        self.data.ctrl[3] = torques[0, 3].item() * k  # rf0
        self.data.ctrl[4] = torques[0, 4].item() * k  # rf1

        # test motor
        # self.data.ctrl[0] = 10 # lf0
        # self.data.ctrl[1] = -10 # lf1
        # self.data.ctrl[2] = -1 # rf0
        # # self.data.ctrl[3] = -10 # rf0
        # # self.data.ctrl[4] = 10 # rf1
        # self.data.ctrl[5] = 1 # rf1

        # 轮毂控制
        self.data.ctrl[2] = torques[0, 2].item() * k  # 左轮
        self.data.ctrl[5] = torques[0, 5].item() * k  # 右轮
        
        # print(f"output: {output}")

        # self.data.ctrl[0] = 0  # lf0
        # self.data.ctrl[1] = 1.5  # lf1
        # self.data.ctrl[3] = 0  # rf0
        # self.data.ctrl[4] = 0  # rf1
        # self.data.ctrl[2] = 0  # 左轮
        # self.data.ctrl[5] = 0  # 右轮

        # 保存当前动作
        self.last_actions = actions.detach().numpy().copy()

    def run(self):
        """主运行循环"""
        last_time = time.time()
        self.__init__()
        # self._init_joint_positions()
        # 初始观测
        self.obs_his = self._init_obs_history()
        self.compute_L0_theta0()  # 计算初始 theta0 和 theta0_dot
        # print(self.obs_history)
        step = 0
        # obs_test = torch.zeros_like(torch.tensor(self.obs)).unsqueeze(0)
        # obs_his_test = torch.zeros_like(torch.tensor(self._get_obs_history())).unsqueeze(0)
        # joint_name = "rf1_Joint"
        # joint_id = self.mj_model.joint(joint_name).id
        # print(f"joint_id: {joint_id}")
        
        with mujoco.viewer.launch_passive(self.mj_model, self.data) as viewer:     
            # 获取摄像头ID并绑定
            camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            viewer.cam.fixedcamid = camera_id  # 指定摄像头ID
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # 固定摄像头模式
            while viewer.is_running():       # 控制频率保持
                dt = 1.0 / self.control_rate
                while (time.time() - last_time) < dt:
                    time.sleep(0.001)
                # 获取当前观测
                self.compute_L0_theta0()  # 更新 theta0 和 theta0_dot
                self.obs = torch.tensor(self._get_obs()).unsqueeze(0)  # 转换为 Tensor 并添加维度
                # print(f"current_obs shape: {current_obs.shape}")  # 打印 current_obs 的形状
                # torch.set_printoptions(threshold=torch.inf)
                # print("obs:", self.obs)

                # 获取历史观测（用于策略网络输入）
                self.obs_his = torch.tensor(self._get_obs_history()).unsqueeze(0)  # 转换为 Tensor 并添加维度
                # print(f"obs_his: {self.obs_his}")  # 打印 obs_his 
                    
                # test_obs = torch.tensor(np.zeros(27, dtype=np.float32)).unsqueeze(0)
                # current_obs = test_obs

                # 策略推理
                with torch.no_grad():

                    output_2 = self.policy_model_2(self.obs_his)
                    # print(f"output_2 shape: {output_2.shape}")  # 打印 output_2 的形状
                    combined_input = torch.cat([self.obs, output_2], dim=1)  # 拼接输入
                    # print(f"combined_input shape: {combined_input.shape}")  # 打印 combined_input 的形状
                    output_1 = self.policy_model_1(combined_input)
                    output_1 = torch.clamp(output_1, -100, 100)  # 限制输出范围
                    # print(f"output_1: {output_1}")
                    # print(f"output_2: {output_2/2}")
                    # print("base_quat",self.base_quat) 
                
                # time.sleep(30)
                if(step > 10000 and step < 15000):
                    self.target_linear_velocity = 0.5
                    self.target_angular_vel = 0.0
                else:
                    self.target_linear_velocity = 0.0
                
                self._apply_control(output_1)
                # print(output_1.shape)
                # print(combined_input.shape)
                mujoco.mj_step(self.mj_model, self.data) 
                # 渲染
                if step % 10 == 0:
                    viewer.sync()
                step += 1

        glfw.terminate()

if __name__ == '__main__':
    sim = MuJoCoControl()
    sim.run()
