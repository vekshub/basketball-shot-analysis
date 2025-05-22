import cv2
import os
from ultralytics import YOLO
from kalmanfilter import KalmanFilter
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import Tk, filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import time


# Tk().withdraw()

# Odabir izvora videa
#choice = messagebox.askquestion("Izvor videa", "Želite li koristiti kameru?")

# if choice == 'yes':
#     cam_index = simpledialog.askinteger("Odabir kamere", "Unesite indeks kamere (0 za web kameru):", minvalue=0)
#     cap = cv2.VideoCapture(cam_index)
#     output_name = f"output_camera_{cam_index}.mp4"
# else:
#     video_path = filedialog.askopenfilename(title="Odaberite video", filetypes=[("Video files", "*.mp4 *.avi")])
#     cap = cv2.VideoCapture(video_path)
#     output_name = os.path.splitext(os.path.basename(video_path))[0] + "_output.mp4"
 
video_path = filedialog.askopenfilename(title="Odaberite video", filetypes=[("Video files", "*.mp4 *.avi")])
cap = cv2.VideoCapture(video_path)
output_name = os.path.splitext(os.path.basename(video_path))[0] + "_output.mp4"

# Putanje i modeli
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "bestV11.pt")
classes_path = os.path.join(base_path, "classesv11.txt")
output_path = os.path.join(base_path, output_name)

#spremanje modela i klasa
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
kf= KalmanFilter()
ballnet_conf = 0.75
trajectory = []
trajectory_prediction = []
all_ball_positions = []
shot_number = 0
made_basket = False
ballnet_count = 0
last_ballhand_x = None
last_ballhand_y = None
shots = []
start_frame = 0
# Cooldown (broj frameova između dva pokušaja ili dva pogotka)
cooldown_frames_shot = 25
cooldown_frames_goal = 15
last_shot_frame = -cooldown_frames_shot
last_goal_frame = -cooldown_frames_goal
ballhand_conf = 0.8

#konstruktor za spremanje podataka o pokušaju
class ShotData:
    def __init__(self, shot_number, start_frame, end_frame, made, trajectory_points):
        self.shot_number = shot_number
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.made = made
        self.trajectory = trajectory_points  # list of (x, y)
        self.angle = 45.0  # dummy
        self.velocity = 8.0  # dummy

#funkcija za crtanje putanje
def create_trajectory_image(shot_data, output_path):
    x, y = zip(*shot_data.trajectory)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker='o', color='blue', label='Ball Trajectory')
    plt.title(f'Shot #{shot_data.shot_number} is {"made" if shot_data.made else "missed"}\n'
              f'Angle: {shot_data.angle}°, Velocity: {shot_data.velocity} m/s')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()

    plt.savefig(output_path)
    plt.close()

def filter_trajectory_jumps(trajectory, max_dist=100):
    """
    Removes points from the trajectory if the distance from the previous point exceeds max_dist.
    """
    if not trajectory:
        return []

    filtered = [trajectory[0]]
    for pt in trajectory[1:]:
        prev = filtered[-1]
        dx = pt[0] - prev[0]
        dy = pt[1] - prev[1]
        dist = (dx**2 + dy**2)**0.5

        if dist <= max_dist:
            filtered.append(pt)
        else:
            # Skip this point as it's a jump
            continue

    return filtered

# def overlap_percentage(boxA, boxB):
#     """Računa postotak preklapanja dvaju boxova (IoU)"""
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(boxA[2], boxB[2])
#     yB = min(boxA[3], boxB[3])

#     interArea = max(0, xB - xA) * max(0, yB - yA)
#     if interArea == 0:
#         return 0.0

#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

#     iou = interArea / float(boxAArea + boxBArea - interArea)
#     return iou 

#priprema mape za slike
IMAGE_FOLDER = "./trajectories"
for filename in os.listdir(IMAGE_FOLDER):
    file_path = os.path.join(IMAGE_FOLDER, filename)
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

avg_conf = []
# Glavna petlja
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]
    ball_box = None
    net_box = None
    ballnet_box = None
    ballhand_box = None
    net_box_lower_third = None
    tracking = True

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{class_names[cls]} {conf:.2f}"

        if class_names[cls].lower() == "ballnet" and conf < ballnet_conf:
            continue
        if class_names[cls].lower() == "ballhand" and conf < ballhand_conf:
            # avg_conf.append(conf)
            continue

        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if class_names[cls].lower() == "basketball":
            ball_box = (x1, y1, x2, y2)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            ball_last_seen = (center_x, center_y)
            ball_y_last_seen = (y1 + y2) // 2  

            all_ball_positions.append((center_x, center_y))
            # predicted=kf.predict(center_x,center_y)
            # trajectory_prediction.append(predicted)
            

        elif class_names[cls].lower() == "net":
            net_box = (x1, y1, x2, y2)
            net_box_lower_third = (x1, y2 - (y2 - y1)//3, x2, y2)

        elif class_names[cls].lower() == "ballnet":
            ballnet_box = (x1, y1, x2, y2)
            ballnet_count += 1
            all_ball_positions.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))

        elif class_names[cls].lower() == "ballhand":
            # avg_conf.append(conf)
            ballhand_box = (x1, y1, x2, y2)
            last_ballhand_x = int((x1 + x2) / 2)
            last_ballhand_y = int((y1 + y2) / 2)
            all_ball_positions.append((last_ballhand_x, last_ballhand_y))
        
    
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
                ballnet_count = 0



        # for point in trajectory:
        #     cv2.circle(frame, point, radius=5, color=(255, 0, 0), thickness=-1)
        # # for point in trajectory_prediction:
        # #     cv2.circle(frame, point, radius=5, color=(0, 255, 0), thickness=-1)

        # for i in range(1, len(trajectory)):
        #     cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), thickness=2)
        # # for i in range(5, len(trajectory_prediction)):
        # #     cv2.line(frame, trajectory_prediction[i - 1], trajectory_prediction[i], (0, 255, 0), thickness=2)

    #ako je lopta u mreži i prošlo je dovoljno frameova od zadnjeg pokušaja i broj frameova u mreži je veći od 2    
    if ballnet_box and frame_count - last_goal_frame > cooldown_frames_goal and ballnet_count > 1:
        made += 1
        made_basket = True
        last_goal_frame = frame_count

        #log_shot_to_json(shot_number,0, 0,made_basket, trajectory_prediction, filename='shots_log.json')
    # cv2.putText(frame, f"Ballnet count: {ballnet_count}", (10, 100),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    #Prikaz rezultata
    cv2.putText(frame, f"TOTAL SHOTS: {total_shots}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, f"MADE SHOTS: {made}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    for point in all_ball_positions:
        cv2.circle(frame, point, radius=5, color=(0, 0, 255), thickness= 5)
        # cv2.putText(frame, f"({point[0]}, {point[1]})", (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    #prikaz i zapis putanje
    if frame_count == last_shot_frame + 10:
        start = 0
       
        print("last_ballhand_x", last_ballhand_x)
        print("last_ballhand_y", last_ballhand_y)
        for i in range(0, len(all_ball_positions)-1,1):
            x, y = all_ball_positions[i]
            if x==last_ballhand_x and y == last_ballhand_y:
                start = i
                break
        trajectory = all_ball_positions[start:]  # putanja
        # trajectory = filter_trajectory_jumps(trajectory, max_dist=1000)
        shot_number += 1
        print("all_ball_positions", all_ball_positions)
        print("trajectory", trajectory)
        print("start", start)

        for point in trajectory:
            cv2.circle(frame, point, radius=5, color=(255, 0, 0), thickness=-1)
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 0, 0), thickness=2)

        sd = ShotData(shot_number = total_shots, start_frame= frame_count + 5 - len(trajectory), end_frame=frame_count+5, made=made_basket, trajectory_points=trajectory[:-5])
        shots.append(sd)
        trajectory = []
        trajectory_prediction = []
        all_ball_positions = []
        made_basket = False

    # print("average confidence", sum(avg_conf)/len(avg_conf))
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
trajectory_path = os.path.join(base_path, "trajectories")
for shot in shots:
    trajectory_image_path = os.path.join(trajectory_path, f"trajectory_shot_{shot.shot_number}.png")
    create_trajectory_image(shot, trajectory_image_path)

time.sleep(5)


IMAGE_SIZE = (640, 640)
IMAGES_PER_ROW = 2  


def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

top = Tk()
top.title("SHOT ANALYSIS")
top.geometry("1280x1280")

main_frame = Frame(top)
main_frame.pack(fill=BOTH, expand=1)

canvas = Canvas(main_frame)
canvas.pack(side=LEFT, fill=BOTH, expand=1)
scrollbar = Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))  # Linux scroll up
canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))   # Linux scroll down


canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

img_frame = Frame(canvas)
canvas.create_window((0, 0), window=img_frame, anchor="nw")

photo_images = []
row,col = 0,0
for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(img_path)
        img.thumbnail(IMAGE_SIZE)
        photo = ImageTk.PhotoImage(img)
        photo_images.append(photo)

        label = Label(img_frame, image=photo)
        
        label.grid(row=row,column=col, padx=5, pady=5)
        col += 1
        if col >= IMAGES_PER_ROW:
            col = 0
            row += 1

top.mainloop()

