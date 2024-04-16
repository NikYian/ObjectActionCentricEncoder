from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x-seg.pt')  # pretrained YOLOv8n model


# Run batched inference on a list of images
results = model.predict(['data/test.png'], save=True)  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     breakpoint()