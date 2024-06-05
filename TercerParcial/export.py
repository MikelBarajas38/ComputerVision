import argparse
from ultralytics import YOLO

def main():
    model = YOLO(args.model) 
    model.info() 
    path = model.export(format = 'onnx')      

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default = 'yolov8n.onnx', help='Input your ONNX model.')
    args = parser.parse_args()
    main()