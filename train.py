from ultralytics import YOLO


def train_model():
    yaml_content = """
path: /app/dataset
train: images/train
val: images/val
names:
  0: tbank-logo
"""

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


    model.save('models/trained.pt')
    print("finally trained omg")


if __name__ == '__main__':
    train_model()