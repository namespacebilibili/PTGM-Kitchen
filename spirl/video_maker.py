import cv2

image_ori = cv2.imread("img/video_0.jpg")

video_size = (image_ori.shape[1],image_ori.shape[0])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter("img/v_1_50.mp4",  fourcc, 10,video_size,True)

for i in range(50):
    frame = cv2.imread(f"img/video_{i}.jpg")
    video.write(frame)
video.release()
