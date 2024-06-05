# Ultralytics YOLO 🚀, AGPL-3.0 license
# https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-OpenCV-ONNX-Python/main.py

import argparse

import cv2
import numpy as np

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

# Load the COCO class names
CLASSES = yaml_load(check_yaml('coco8.yaml'))['names']

# Generate random colors for each class
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ----------------------------- utils -------------------------------------------

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def preprocess(img):
    # Get image shape
    height, width, _ = img.shape

    # Prepare a square image for inference
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = img

    # Calculate scale factor
    scale = length / 640

    # Preprocess the image and prepare blob for model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)

    return blob, scale

def process_detections(img, scale, outputs):

    # Prepare output array
    outputs = np.array([cv2.transpose(outputs[0])])

    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):

        classes_scores = outputs[0][i][4:]

        (min_score, max_score, min_class_loc, (x, max_class_index)) = cv2.minMaxLoc(classes_scores)

        #max_score = np.max(classes_scores)
        #max_class_index = np.argmax(classes_scores)

        if max_score >= 0.25:
            box = [
                outputs[0][i][0] - (outputs[0][i][2] / 2),
                outputs[0][i][1] - (outputs[0][i][3] / 2),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(max_score)
            class_ids.append(max_class_index)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=0.45, eta=0.5)

    detections = []

    # Iterate through NMS results to draw bounding boxes and labels
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box,
            'scale': scale,
        }
        detections.append(detection)
        draw_bounding_box(
            img,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )
    
    return detections

# --------------------- main functions --------------------------

def detect_img(image_path):

    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(args.model)

    img: np.ndarray = cv2.imread(image_path)

    blob, scale = preprocess(img)

    model.setInput(blob)

    outputs = model.forward()

    #print(outputs.shape)

    detections = process_detections(img, scale, outputs)

    cv2.imshow(f'{image_path}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # return detections

def detect_video(video_path=None):
    
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(args.model)

    # Open the video capture (use the webcam if no path is provided)
    cap = cv2.VideoCapture(video_path if video_path else 0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        img: np.ndarray = frame

        blob, scale = preprocess(img)

        model.setInput(blob)

        outputs = model.forward()

        detections = process_detections(img, scale, outputs)
     
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
    parser.add_argument('--model', default = 'yolov8n.onnx', help='Input your ONNX model.')
    parser.add_argument('--video', action = 'store_true', help='Use video stream.')
    parser.add_argument('--input', default = None, help='Path to input.')
    args = parser.parse_args()
    main()