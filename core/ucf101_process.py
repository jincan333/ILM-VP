import cv2
import os

def extract_frames(video_path, frame_dir, skip_frames=10):
    # Open the video file
    vidcap = cv2.VideoCapture(video_path)

    # Go through each frame
    count = 0
    frame_id = 0 
    while True:
        success, image = vidcap.read()
        if not success:
            break

        # We only save every 'skip_frames' frames
        if count % skip_frames == 0:
            # Save frame as JPEG file
            frame_filename = f"{frame_dir}/frame_{frame_id}.jpg"
            cv2.imwrite(frame_filename, image)
            frame_id += 1

        count += 1

# Define your video path and frame directory
video_path = '/path/to/your/video.avi'
frame_dir = '/path/to/save/frames'
os.makedirs(frame_dir, exist_ok=True)

# Extract frames
extract_frames(video_path, frame_dir)
