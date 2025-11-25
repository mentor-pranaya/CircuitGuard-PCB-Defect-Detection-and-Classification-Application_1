from ultralytics import YOLO

def train():
    # Load a model
    model = YOLO("yolov8n.pt")  # load a pretrained model (nano version for speed)

    # Train the model
    # data: path to data.yaml
    # epochs: number of training epochs
    # imgsz: image size
    # device: 0 for GPU, cpu for CPU
    results = model.train(
        data=r"c:\Users\jayza\OneDrive\Desktop\Infosys\yolo_dataset\data.yaml", 
        epochs=50, 
        imgsz=640,
        device=0, # Use GPU if available
        batch=4, # Reduce batch size for 4GB VRAM
        workers=2, # Reduce workers
        project=r"c:\Users\jayza\OneDrive\Desktop\Infosys\yolo_training",
        name="circuitguard_yolo"
    )

    # Validate
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

    # Export
    path = model.export(format="pt")
    print(f"Model exported to {path}")

if __name__ == "__main__":
    train()
