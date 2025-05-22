import cv2
import os
from ultralytics import YOLO
from tkinter import Tk, filedialog, simpledialog, messagebox

Tk().withdraw()

# Odabir izvora videa
choice = messagebox.askquestion("Izvor videa", "Želite li koristiti kameru?")

if choice == 'yes':
    cam_index = simpledialog.askinteger("Odabir kamere", "Unesite indeks kamere (0 za web kameru):", minvalue=0)
    cap = cv2.VideoCapture(cam_index)
    output_name = f"output_camera_{cam_index}.mp4"
else:
    video_path = filedialog.askopenfilename(title="Odaberite video", filetypes=[("Video files", "*.mp4 *.avi")])
    cap = cv2.VideoCapture(video_path)
    output_name = os.path.splitext(os.path.basename(video_path))[0] + "_output.mp4"

# Putanje i modeli
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "bestV11.pt")
classes_path = os.path.join(base_path, "classesv11.txt")
output_path = os.path.join(base_path, output_name)

model = YOLO(model_path)
with open(classes_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

#postavke zapisa videa
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Inicijalizacija
frame_count = 0
made = 0
total_shots = 0
ball_last_seen = None
ball_y_last_seen = None

# Cooldown (broj frameova između dva pokušaja ili dva pogotka)
cooldown_frames_shot = 25
cooldown_frames_goal = 30
last_shot_frame = -cooldown_frames_shot
last_goal_frame = -cooldown_frames_goal

def overlap_percentage(boxA, boxB):
    """Računa postotak preklapanja dvaju boxova (IoU)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]
    ball_box = None
    net_box = None
    net_box_lower_third = None

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{class_names[cls]} {conf:.2f}"

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if class_names[cls].lower() == "basketball":
            ball_box = (x1, y1, x2, y2)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            ball_last_seen = (center_x, center_y)
            ball_y_last_seen = (y1 + y2) // 2  # srednji y

        elif class_names[cls].lower() == "net":
            net_box = (x1, y1, x2, y2)
            net_box_lower_third = (x1, y2 - (y2 - y1)//3, x2, y2)
        
        elif class_names[cls].lower() == "ballnet":
            ballnet_box = (x1, y1, x2, y2)
    
    if ball_box and net_box:
        bx, by = ball_last_seen if ball_last_seen else (0, 0)
        nx1, ny1, nx2, ny2 = net_box
        net_cx = (nx1 + nx2) // 2
        net_cy = (ny1 + ny2) // 2
        net_width = nx2 - nx1
        net_height = ny2 - ny1

        radius = 2 * net_height
        distance = ((bx - net_cx) ** 2 + (by - net_cy) ** 2) ** 0.5

        #DODANO:Lopta mora biti iznad mreže da bismo brojali pokušaj
        if distance < radius and frame_count - last_shot_frame > cooldown_frames_shot:
            if ball_y_last_seen is not None and ball_y_last_seen < net_cy:
                total_shots += 1
                last_shot_frame = frame_count

        #Provjera postotka preklapanja
        overlap = overlap_percentage(ball_box, net_box_lower_third)
        cv2.putText(frame, f"Overlap: {overlap:.2f}", (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if overlap > 0.1 and frame_count - last_goal_frame > cooldown_frames_goal:
            made += 1
            last_goal_frame = frame_count

    #Prikaz rezultata
    cv2.putText(frame, f"TOTAL SHOTS: {total_shots}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, f"MADE SHOTS: {made}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    # Prikaz i spremanje
    preview_resized = cv2.resize(frame, (960, 540))
    cv2.imshow("Processing...", preview_resized)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video spremljen:", output_path)
