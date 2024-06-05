# Ultralytics YOLO 🚀, AGPL-3.0 license
# Same as yolo_opencv.py but with the Ultralytics functions (way better imo).

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

def detect_img(image_path):

    model = YOLO(args.model)

    img: np.ndarray = cv2.imread(image_path)

    results = model(img) 

    if results:
        cv2.imshow(f'{image_path}', results[0].plot())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # return results

def detect_video(video_path=None):
    
    model = YOLO(args.model)

    # Open the video capture (use the webcam if no path is provided)
    cap = cv2.VideoCapture(video_path if video_path else 0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        img = frame

        results = model(img)

        if results:
            img = results[0].plot()
     
        cv2.imshow(f'video stream: {video_path}', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

def main():
    if args.video is True:
        detect_video(args.input)
    elif input is not None:
        detect_img(args.input)
    else:
        print('Please provide an input image or video stream.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = 'yolov8n.pt', help='Input your .pt model.')
    parser.add_argument('--video', action = 'store_true', help='Use video stream.')
    parser.add_argument('--input', default = None, help='Path to input.')
    args = parser.parse_args()
    main()