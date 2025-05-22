import cv2
import depthai as dai
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Postavke
model_path = "best.pt"
classes_path = "classes.txt"
output_dir = "trajectories"
os.makedirs(output_dir, exist_ok=True)

# Učitavanje modela
model = YOLO(model_path)
with open(classes_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Inicijalizacija kamere
pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

stereo = pipeline.createStereoDepth()
left = pipeline.create(dai.node.MonoCamera)
right = pipeline.create(dai.node.MonoCamera)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

left.out.link(stereo.left)
right.out.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Start
device = dai.Device(pipeline)
rgb_queue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
depth_queue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

frame_count = 0
made = 0
total_shots = 0
last_shot_frame = -50
last_goal_frame = -50
cooldown_frames_shot = 25
cooldown_frames_goal = 25

current_trajectory = []
trajectory_id = 0

while True:
    in_rgb = rgb_queue.get()
    in_depth = depth_queue.get()

    frame = in_rgb.getCvFrame()
    depth_frame = in_depth.getFrame()

    frame_count += 1
    results = model(frame)[0]

    ball_box = None
    net_box = None
    ball_center = None
    net_center = None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = class_names[cls]

        if label.lower() == "basketball":
            ball_box = (x1, y1, x2, y2)
            ball_center = (int((x1+x2)/2), int((y1+y2)/2))
        elif label.lower() == "net":
            net_box = (x1, y1, x2, y2)
            net_center = (int((x1+x2)/2), int((y1+y2)/2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if ball_center and net_center:
        bx, by = ball_center
        nx, ny = net_center

        net_width = net_box[2] - net_box[0]
        net_height = net_box[3] - net_box[1]
        net_cy = (net_box[1] + net_box[3]) // 2

        # Blizina za pokušaj
        distance = ((bx - nx)**2 + (by - ny)**2) ** 0.5
        radius = 2 * net_height

        if distance < radius and by < net_cy and frame_count - last_shot_frame > cooldown_frames_shot:
            total_shots += 1
            last_shot_frame = frame_count
            current_trajectory = []

        # Dodavanje trenutne 3D pozicije lopte u putanju
        if 0 <= bx < depth_frame.shape[1] and 0 <= by < depth_frame.shape[0]:
            z = depth_frame[by, bx] / 1000.0  # Pretvori mm u metre
            current_trajectory.append((bx, by, z))

        # Provjera pogotka prema dubini
        if 0 <= nx < depth_frame.shape[1] and 0 <= ny < depth_frame.shape[0]:
            ball_depth = depth_frame[by, bx] / 1000.0
            net_depth = depth_frame[ny, nx] / 1000.0

            if abs(ball_depth - net_depth) < 0.2 and frame_count - last_goal_frame > cooldown_frames_goal:
                made += 1
                last_goal_frame = frame_count

                # Spremi putanju
                if len(current_trajectory) > 5:
                    xs, ys, zs = zip(*current_trajectory)
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot(xs, ys, zs, marker='o')
                    ax.set_xlabel('X pixel')
                    ax.set_ylabel('Y pixel')
                    ax.set_zlabel('Depth (m)')
                    plt.title(f"Trajectory {trajectory_id}")
                    plt.savefig(f"{output_dir}/trajectory_{trajectory_id}.png")
                    plt.close()
                    trajectory_id += 1

    # Prikaz statistike
    cv2.putText(frame, f"TOTAL SHOTS: {total_shots}  MADE SHOTS: {made}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    preview_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("Processing...", preview_resized)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
