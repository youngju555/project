#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from pymycobot import MyCobot320
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time
import tf.transformations as tf
import math
import os 

# --- [필수] 카메라 내부 파라미터 ---
K = np.array([[539.13729067, 0., 329.02126026],
              [0., 542.34217387, 242.10995541],
              [0., 0., 1.]])
D = np.array([[0.205286030, -0.766640681, -0.000966140, 0.001118916, 0.976300036]])

# --- 체커보드 설정 ---
CHECKERBOARD_SIZE = (8, 7)  # 내부 코너 수
SQUARE_SIZE = 0.015  # m 단위

# 3D 체커보드 월드 포인트
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 데이터 저장 리스트
T_base_to_hand_list = []
T_camera_to_target_list = []

# --- 오차 계산 함수 ---
def compute_hand_eye_error(T_A_list, T_B_list, R_hand_to_camera, t_hand_to_camera):
    num_poses = len(T_A_list)
    total_rot_err = 0.0
    total_trans_err = 0.0

    T_hand_to_camera = np.eye(4)
    T_hand_to_camera[0:3, 0:3] = R_hand_to_camera
    T_hand_to_camera[0:3, 3] = t_hand_to_camera.reshape(3)

    for i in range(num_poses - 1):
        T_A_rel = np.dot(np.linalg.inv(T_A_list[i + 1]), T_A_list[i])
        T_B_rel = np.dot(T_B_list[i + 1], np.linalg.inv(T_B_list[i]))
        
        T_Left = np.dot(T_A_rel, T_hand_to_camera)
        T_Right = np.dot(T_hand_to_camera, T_B_rel)
        T_Error = np.dot(np.linalg.inv(T_Left), T_Right)

        R_err = T_Error[0:3, 0:3]
        trace_val = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
        rot_err_deg = math.degrees(math.acos(trace_val))
        trans_err = np.linalg.norm(T_Error[0:3, 3])

        total_rot_err += rot_err_deg
        total_trans_err += trans_err

    avg_rot_err = total_rot_err / (num_poses - 1)
    avg_trans_err = total_trans_err / (num_poses - 1)
    return avg_rot_err, avg_trans_err


class HandEyeCalibrator:
    def __init__(self):
        rospy.init_node('hand_eye_calibrator', anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback)
        self.current_frame = None

        # --- MyCobot 연결 ---
        try:
            port = rospy.get_param("~port", "/dev/ttyACM0")
            baud = rospy.get_param("~baud", 115200)
            self.mc = MyCobot320(port, baud)
            rospy.loginfo(f"✅ myCobot 연결 성공: {port}")

            # 초기 상태: 전원 끄기 (토크 OFF)
            self.mc.power_off()
            rospy.loginfo("🔌 로봇 토크를 OFF했습니다. 손으로 자유롭게 움직이세요.")

            # 종료 시 안전하게 전원 OFF
            rospy.on_shutdown(self.safe_shutdown)

        except Exception as e:
            rospy.logerr(f"❌ myCobot 연결 실패: {e}")
            self.mc = None

    def safe_shutdown(self):
        if self.mc:
            try:
                self.mc.power_off()
                rospy.loginfo("⚡ 종료: 로봇 토크 OFF 완료.")
            except Exception:
                rospy.logwarn("⚠️ 종료 중 토크 OFF 실패 (이미 전원 해제됨).")

    def image_callback(self, msg):
        try:
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

    def convert_coords_to_matrix(self, coords_mm_deg):
        pos_m = np.array(coords_mm_deg[:3], dtype=np.float64) / 1000.0
        angles_rad = np.radians(coords_mm_deg[3:])
        quat = tf.quaternion_from_euler(*angles_rad, axes='rxyz')  # 올바른 회전 순서
        T = tf.quaternion_matrix(quat)
        T[0:3, 3] = pos_m
        return T

    def get_stable_coords(self):
        """좌표를 두 번 읽어서 평균값을 반환 (더 안정적인 측정용)."""
        coords1 = self.mc.get_coords()
        time.sleep(0.5)
        coords2 = self.mc.get_coords()
        if not (isinstance(coords1, list) and isinstance(coords2, list) and len(coords1) == 6 and len(coords2) == 6):
            return None
        avg_coords = [(a + b) / 2 for a, b in zip(coords1, coords2)]
        return avg_coords

    def run_calibration_loop(self):
        if self.mc is None:
            rospy.logerr("MyCobot 연결 안 됨. 종료합니다.")
            return

        pose_count = 0
        rospy.loginfo("=== 핸드-아이 보정 시작 ===")
        rospy.loginfo("로봇 팔을 손으로 움직여 다양한 자세를 만드세요.")
        rospy.loginfo("체커보드는 고정되어 있어야 합니다.")
        rospy.loginfo("[s] → 데이터 저장, [q] → 종료")

        while not rospy.is_shutdown():
            if self.current_frame is None:
                rospy.loginfo_throttle(1, "카메라 이미지를 기다리는 중...")
                time.sleep(0.1)
                continue

            frame = self.current_frame.copy()
            undistorted = cv2.undistort(frame, K, D)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

            if ret:
                cv2.drawChessboardCorners(undistorted, CHECKERBOARD_SIZE, corners, ret)

            cv2.imshow("Hand-Eye Calibration (Undistorted)", undistorted)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('s'):
                if not ret:
                    rospy.logwarn("체커보드를 찾지 못했습니다. 다시 시도하세요.")
                    continue

                rospy.loginfo(f"--- {pose_count + 1}번째 자세 데이터 수집 ---")

                try:
                    self.mc.power_on()
                    rospy.loginfo("   ...좌표 읽기 위해 로봇 고정 (Power ON)")
                    time.sleep(2.0)
                except Exception as e:
                    rospy.logwarn(f"   로봇 전원 켜기 실패: {e}")
                    continue

                coords = self.get_stable_coords()
                if coords is None:
                    rospy.logwarn("❌ 로봇 좌표(A) 획득 실패. 다시 시도하세요.")
                    try:
                        self.mc.power_off()
                    except Exception:
                        pass
                    continue

                T_base_to_hand = self.convert_coords_to_matrix(coords)
                T_base_to_hand_list.append(T_base_to_hand)
                rospy.loginfo(f"✅ 로봇 좌표(A): {np.round(coords, 2)}")

                success, rvec, tvec = cv2.solvePnP(objp, corners, K, D)
                if success:
                    R_cam_to_target, _ = cv2.Rodrigues(rvec)
                    T_camera_to_target = np.eye(4)
                    T_camera_to_target[0:3, 0:3] = R_cam_to_target
                    T_camera_to_target[0:3, 3] = tvec.flatten()
                    T_camera_to_target_list.append(T_camera_to_target)
                    rospy.loginfo("✅ 체커보드 좌표(B) 획득 성공.")
                    pose_count += 1
                else:
                    rospy.logwarn("❌ PnP 실패. 이 자세는 무시합니다.")

                try:
                    self.mc.power_off()
                    rospy.loginfo("   ...다음 자세를 위해 로봇 고정 해제 (Power OFF)")
                except Exception as e:
                    rospy.logwarn(f"   로봇 전원 끄기 실패: {e}")
                
        cv2.destroyAllWindows()
        self.safe_shutdown()

        if pose_count < 5:
            rospy.logerr(f"데이터가 부족합니다 ({pose_count}개). 최소 5개 이상 필요합니다.")
            return

        rospy.loginfo(f"총 {pose_count}개의 자세로 계산 중...")

        # --- 상대 운동 계산 ---
        R_hand_to_base_rel_list, t_hand_to_base_rel_list = [], []
        R_target_to_camera_rel_list, t_target_to_camera_rel_list = [], []

        for i in range(pose_count - 1):
            T_A_rel = np.dot(np.linalg.inv(T_base_to_hand_list[i + 1]), T_base_to_hand_list[i])
            T_B_rel = np.dot(T_camera_to_target_list[i + 1], np.linalg.inv(T_camera_to_target_list[i]))

            R_hand_to_base_rel_list.append(T_A_rel[0:3, 0:3])
            t_hand_to_base_rel_list.append(T_A_rel[0:3, 3].reshape(3, 1))
            R_target_to_camera_rel_list.append(T_B_rel[0:3, 0:3])
            t_target_to_camera_rel_list.append(T_B_rel[0:3, 3].reshape(3, 1))

        rospy.loginfo("상대 운동 계산 완료. 핸드-아이 보정 시작...")

        R_hand_to_camera, t_hand_to_camera = cv2.calibrateHandEye(
            R_hand_to_base_rel_list,
            t_hand_to_base_rel_list,
            R_target_to_camera_rel_list,
            t_target_to_camera_rel_list,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        avg_rot_err, avg_trans_err = compute_hand_eye_error(
            T_base_to_hand_list,
            T_camera_to_target_list,
            R_hand_to_camera,
            t_hand_to_camera
        )

        rospy.loginfo("=== 🔍 핸드-아이 보정 결과 ===")
        rospy.loginfo(f"R_hand_to_camera:\n{R_hand_to_camera}")
        rospy.loginfo(f"t_hand_to_camera (m): {t_hand_to_camera.flatten()}")
        rospy.loginfo(f"평균 회전 오차: {avg_rot_err:.3f} deg")
        rospy.loginfo(f"평균 이동 오차: {avg_trans_err*1000:.3f} mm")

        try:
            T_hand_to_camera = np.eye(4)
            T_hand_to_camera[0:3, 0:3] = R_hand_to_camera
            T_hand_to_camera[0:3, 3] = t_hand_to_camera.flatten()

            home_dir = os.path.expanduser("~")
            save_path = os.path.join(home_dir, "hand_eye_calibration.npz")
            
            np.savez(save_path, T_hand_to_camera=T_hand_to_camera, K=K, D=D)
            rospy.loginfo(f"✅ 보정 결과를 {save_path} 파일로 저장했습니다.")

        except Exception as e:
            rospy.logerr(f"❌ 보정 결과 파일 저장 실패: {e}")

        rospy.loginfo("=== 완료 ===")


if __name__ == '__main__':
    try:
        calibrator = HandEyeCalibrator()
        calibrator.run_calibration_loop()
    except rospy.ROSInterruptException:
        pass
