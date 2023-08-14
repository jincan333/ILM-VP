import cv2
import os

def extract_frames(video_path, frame_dir):
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



if __name__=='__main__':
    # Define your video path and frame directory
    video_path = '/data4/hop20001/can/ILM-VP/dataset/ucf101/UCF-101'
    frame_dir = 'ucf101/'
    os.makedirs(frame_dir, exist_ok=True)

    classes = os.listdir(video_path)
    total_frames = 0
    avis_cnt = 0
    for c in classes:
        class_path = os.path.join(video_path, c)
        avis = os.listdir(class_path)
        for avi in avis:
            avi_path = os.path.join(class_path, avi)
            # Extract frames
            # extract_frames(avi_path, frame_dir)
            video = cv2.VideoCapture(avi_path)
            frames_cnt = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames += frames_cnt
            avis_cnt += 1
    print(total_frames, avis_cnt)