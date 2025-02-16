import cv2
# Contributions: Varshini, Jefrey

def priorityChange(camno, lvl,v_ind):
    factorinc = lvl+1
    co = 0
    fins = v_ind.index(camno)
    v_ind = [e for e in v_ind if e != camno]
    if v_ind.count(str(camno)) < factorinc:
        for i in range(factorinc):
            v_ind.insert(fins+(len(v_ind)//factorinc)*co+(co),camno)
            co+=1
    print(v_ind)
    return v_ind

v_ind=[1,2,3,4,5]

codi = {1:0, 2:0,3:0, 4:0,5:0, 6:0}

v_map= {
    1: r"C:\Zlearning2024\GDG\video1.mp4",
    2: r"C:\Zlearning2024\GDG\video2.mp4",
    3: r"C:\Zlearning2024\GDG\video3.mp4",
    4: r"C:\Zlearning2024\GDG\video4.mp4",
    5: r"C:\Zlearning2024\GDG\video5.mp4"
}

cams = { c_idx: cv2.VideoCapture(v_map[c_idx]) for c_idx in v_ind }



fps=2
frame_display=int(1000/fps)
framespervideo=fps

index=0
deb = True
while True:

    v_ind = priorityChange(3,1,v_ind)
    v_ind = priorityChange(5,4,v_ind)

    v_num=v_ind[index]
    cam=cams[v_num]
    
    if not cam.isOpened():
        print(f"Error in opening {v_map[v_num]}")
        index = (index + 1) % len(v_ind)
        continue

    print(f"Now playing: Video {v_num} - {v_map[v_num]}")

    for _ in range(framespervideo):
         success, frame = cam.read()

         if not success:
             print(f"Video {v_num} has ended.")
             break 
         
        
         number = codi[v_num]  # Example number
         if deb:
            #deb = False
            codi[v_num] = codi[v_num]+1
        # Set font, position, color, thickness, and font size for the text
            height, width, _ = frame.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (width - 100, 50)  # Top-right corner
            color = (0, 255, 0)  
            thickness = 2
            font_scale = 1

    # Put the number on the frame
            cv2.putText(frame, str(number), position, font, font_scale, color, thickness)


         cv2.imshow(f"Playing Video {v_num}", frame)
         
         if cv2.waitKey(100) == ord('q'):
            for c in cams.values():
                c.release()
            cv2.destroyAllWindows()
            exit()
            
    index = (index + 1) % len(v_ind)
    deb = True
for cam in cams.values():
    cam.release()

cv2.destroyAllWindows()
