#!/usr/bin/env python3
import os
import time
import torch
import mujoco
import glfw
import numpy as np
from collections import deque
from mujoco import viewer
from math import sqrt, cos, sin, acos, asin

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

        self.control_rate = 100  # 控制频率

        # 初始化关节和执行器信息
        self._init_joint_info()

        # 状态参数
        self.gravity_vec = np.array([0, 0, 1.0])
        self.offset = 0
        self.l1 = 0.15
        self.l2 = 0.25
        self.pi = np.pi

        # 运动指令
        self.commands = np.array([0, 0, 0])
        self.target_linear_velocity = 0.0  # 初始化目标线速度
        self.target_angular_vel = 0.0      # 初始化目标角速度
        self.target_z = 0.18                # 初始化目标z轴位置
        
        # 初始化 d_gains
        self.d_gains = torch.tensor(np.zeros((1, 6)))
        
        # 初始化GLFW窗口
        glfw.init()
        self.window = glfw.create_window(640, 480, "MuJoCo Viewer", None, None)
        glfw.set_key_callback(self.window, self._key_callback)

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
    
    def get_rpy(self):
        self.base_quat = self.data.sensordata[3:7]    # 四元数 (w, x, y, z)
        # 四元数转pitch、roll、yaw
        w, x, y, z = self.base_quat
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z
    
    def get_taget_theta(self, h):
        a = 1
        b = h / self.l1
        c = (h**2 - self.l2**2) / (2 * self.l1**2)
        sin_theta1 = ( - b + sqrt( b**2 - 4 * a * c ) ) / 2
        theta1 = asin(sin_theta1)
        theta2 = asin( cos(theta1) * self.l1 / self.l2 ) -theta1
        return theta1, theta2
    
    def get_dof_tor(self, theta1, theta2, theta1_obs, theta2_obs):
        kp_theta = 40
        tor1 = kp_theta * (theta1 - theta1_obs)
        tor2 = kp_theta * (theta2 - theta2_obs)
        return tor1, tor2
    
    def get_wheel_tor(self, L0, pitch, vel):
        kp_pitch = 5 * L0
        slowdown_angle = 5 * (0 - vel)
        tor = kp_pitch * (slowdown_angle - pitch)
        print(pitch, slowdown_angle)
        return tor
    
    def apply_commands(self, commands):
        self.data.ctrl[0] = commands[0]# lf0
        self.data.ctrl[1] = commands[1]# lf1
        self.data.ctrl[3] = commands[3]# rf0
        self.data.ctrl[4] = commands[4]# rf1
        self.data.ctrl[2] = commands[2]# l_wheel
        self.data.ctrl[5] = commands[5]# r_wheel

    def _init_joint_positions(self):
        """初始化所有关节的初始角度"""
        # 初始化root位置
        root_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "root")
        if root_id != -1:
            self.data.qpos[0:3] = np.array([0, 0, 0.25])
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
    

    def run(self):
        """主运行循环"""
        last_time = time.time()
        # print(self.obs_history)
        step = 0
        self._init_joint_positions()
        with mujoco.viewer.launch_passive(self.mj_model, self.data) as viewer:     
            # 获取摄像头ID并绑定
            camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
            viewer.cam.fixedcamid = camera_id  # 指定摄像头ID
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED  # 固定摄像头模式
            while viewer.is_running():       # 控制频率保持

                dt = 1.0 / self.control_rate
                while (time.time() - last_time) < dt:
                    time.sleep(0.001)
                h = 0.23
                vel_x = self.data.sensordata[7]
                theta1, theta2 = self.get_taget_theta(h)
                # print(theta1, theta2)
                roll, pitch, yaw = self.get_rpy()
                theta1_obs, theta2_obs = self.data.sensordata[16], self.data.sensordata[18]
                l_tor1, l_tor2 = self.get_dof_tor(theta1, theta2, theta1_obs, theta2_obs)
                l_tor_wheel = -self.get_wheel_tor(h, pitch,vel_x)
                theta1_obs, theta2_obs = self.data.sensordata[22], self.data.sensordata[24]
                r_tor1, r_tor2 = self.get_dof_tor(-theta1, -theta2, theta1_obs, theta2_obs)
                r_tor_wheel = self.get_wheel_tor(h, pitch,vel_x)
                self.apply_commands([l_tor1, l_tor2, l_tor_wheel, r_tor1, r_tor2, r_tor_wheel])

                mujoco.mj_step(self.mj_model, self.data) 
                # 渲染
                # time.sleep(0.05)
                if step % 10 == 0:
                    viewer.sync()
                step += 1
        glfw.terminate()

if __name__ == '__main__':
    sim = MuJoCoControl()
    sim.run()
