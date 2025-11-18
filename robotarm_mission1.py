# ===================== 카메라 스레드 =====================
def camera_thread(stop_event, container):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("❌ 카메라 실행 실패")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    w, h = 640, 480
    newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(K, D, None, newK, (w,h), 5)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if ret:
            container["frame"] = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        time.sleep(0.03)

    cap.release()


# ===================== 메인 =====================
def main():

    # 모델 & 포트
    model_path = "C:/Users/peo00/Downloads/best (2).pt"
    port = "COM8"

    model = YOLO(model_path)
    print("YOLO 로드 완료")

    mc = CobotClass(port, 115200)
    mc.power_on()
    time.sleep(1)

    # 그리퍼 초기화
    try:
        if hasattr(mc, "set_gripper_mode"):
            mc.set_gripper_mode(0)
        if hasattr(mc, "init_electric_gripper"):
            mc.init_electric_gripper()
        if hasattr(mc, "set_electric_gripper"):
            mc.set_electric_gripper(0)
        time.sleep(1)
    except:
        pass

    # 초기 위치
    mc.send_angles([0,0,0,0,0,0], 20)
    wait_for_robot_stop(mc)

    mc.send_coords(POSE_CLEAR, DEFAULT_SPEED, 0)
    wait_for_robot_stop(mc)

    mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
    mc.set_gripper_value(50,20,1)
    wait_for_robot_stop(mc)

    # ===================== 반복 수행 =====================
    for cycle in range(9):
        print(f"\n===== 🔁 Cycle {cycle+1}/9 =====")

        # 카메라 실행
        frame_box = {"frame":None}
        stop_event = threading.Event()
        th = threading.Thread(target=camera_thread, args=(stop_event, frame_box))
        th.start()

        ema_cx = ema_cy = ema_ang = None
        alpha = 0.25
        detect_start = None

        detected = None
        final_is_good = None

        while True:
            frame = frame_box["frame"]
            if frame is None:
                continue

            vis, res, ang, conf, fw, fh, name, is_good = detect_object(model, frame)

            if res:
                cx, cy = res

                if ema_cx is None:
                    ema_cx, ema_cy, ema_ang = cx, cy, ang
                    detect_start = time.time()
                else:
                    ema_cx = ema_cx*(1-alpha) + cx*alpha
                    ema_cy = ema_cy*(1-alpha) + cy*alpha
                    ema_ang = ema_ang*(1-alpha) + ang*alpha

                if time.time() - detect_start > 3.0:
                    final_cx = int(ema_cx)
                    final_cy = int(ema_cy)

                    detected = pixel_to_robot(final_cx, final_cy, fw, fh)
                    detected_angle = ema_ang
                    final_is_good = is_good
                    print("📌 픽업 좌표 확정")
                    break

            else:
                ema_cx = ema_cy = ema_ang = None
                detect_start = None

            cv2.imshow("Camera", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        stop_event.set()
        th.join()
        cv2.destroyAllWindows()

        if detected is None:
            print("감지 실패 → 다음 사이클")
            continue

        # ===================== 픽업 =====================
        x, y, z, r, p, yaw = detected

        mc.send_coords([x, y, APPROACH_Z, r, p, yaw], 25, 1)
        wait_for_robot_stop(mc)

        # 회전 보정
        angles = mc.get_angles()
        if angles:
            angles[5] += detected_angle
            mc.send_angles(angles, 25)
            wait_for_robot_stop(mc)

        # Z 하강
        cur = mc.get_coords()
        orig_z = cur[2]
        cur[2] = orig_z - GRAB_DELTA_Z
        mc.send_coords(cur, 20, 0)
        wait_for_robot_stop(mc)

        # 집기
        mc.set_gripper_value(30,40,1)
        time.sleep(1)
        mc.set_gripper_value(15,40,1)
        time.sleep(1)

        # 올리기
        up = mc.get_coords()
        up[2] = orig_z
        mc.send_coords(up,25,0)
        wait_for_robot_stop(mc)

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED,0)
        wait_for_robot_stop(mc)

        # ===================== 분류 이동 =====================
        if final_is_good:
            print("🟢 양품 → PLACE1 위치로 이동")
            place_up = POSE_PLACE1_UP
            place_dn = POSE_PLACE1
        else:
            print("🔴 불량품 → PLACE2 위치로 이동")
            place_up = POSE_PLACE2_UP
            place_dn = POSE_PLACE2

        # 상단으로 이동
        mc.send_coords(place_up, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

       

        # 열기
        mc.set_gripper_value(50,20,1)
        time.sleep(1.0)

        mc.send_coords(POSE_CLEAR, DEFAULT_SPEED,0)
        wait_for_robot_stop(mc)

        mc.send_coords(POSE_HOME, DEFAULT_SPEED, 0)
        wait_for_robot_stop(mc)

        print(f"Cycle {cycle+1} 완료!")

    print("\n🎉 전체 작업 완료!")
   

if __name__ == "__main__":
    main()
