from ultralytics import YOLO
import os
yaml_content = """
path: dataset
train: images/train
val: images/val
names:
  0: tbank-logo
"""

def train_model():


    with open('dataset.yaml', 'w') as f:
        f.write(yaml_content)


    model = YOLO('models/yolov8n.pt')


    results = model.train(
        data='dataset.yaml',
        epochs=20,
        imgsz=640,
        batch=4,
        patience=5,
        device='cpu'
    )


    best_model_path = 'runs/detect/train/weights/best.pt'
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, 'models/trained.pt')
        print("Модель сохранена как models/trained.pt")
    print("finally trained omg")


if __name__ == '__main__':
    train_model()