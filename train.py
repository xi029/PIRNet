from ultralytics import YOLO
if __name__ == "__main__":
    # Load a model
    model = YOLO("ultralytics/cfg/models/11/yolo11-cls-HDMNet .yaml")  # build a new model from YAML  -HDMNet
    # Train the model
    results = model.train(data="D:/yolo_paper/ultralytics-main/datasets", epochs=100, imgsz=64,device='cpu')