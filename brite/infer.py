import time
from ultralytics import YOLO

# Load a model
model = YOLO(
    "./best_YOLO_ambulance_detect.pt", verbose=False
)  # pretrained YOLO26n model

# Run batched inference on a list of images
results = model(
    source=0, stream=True, verbose=False
)  # return a generator of Results objects

# Process results generator
for result in results:
    time.sleep(0.8)
    boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
