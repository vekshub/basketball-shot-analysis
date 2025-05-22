from tkinter import *
import os
from PIL import Image, ImageTk

IMAGE_FOLDER = "./trajectories"
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