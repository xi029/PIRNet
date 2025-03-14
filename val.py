from ultralytics import YOLO
if __name__ == "__main__":

    # Load a model
    model = YOLO("runs/classify/train15/weights/best.pt")  # load the trained model

    # Validate the model
    metrics = model.val(data=r"D:\yolo_paper\ultralytics-main\datasets",device=0,split='test')  # no arguments needed, uses the dataset and settings from training
    metrics.top1  # top1 accuracy
    metrics.top5  # top5 accuracy