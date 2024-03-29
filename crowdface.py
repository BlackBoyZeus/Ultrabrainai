"""
CrowdFace: Advanced Face Detection and Replacement
Leverages YOLOv9 for efficient and accurate face detection, integrating seamless face replacement and privacy enhancements.

Usage:
    import torch  
    model = torch.hub.load('Ultrabrain/CrowdFace', 'crowdface', autoshape=True)
"""
dependencies = [
    "torch",
    "torchvision",
    "opencv-python-headless",  # OpenCV is required for image processing
]

import torch
import cv2
import numpy as np
from pathlib import Path
from yolov9.models.common import AutoShape as _AutoShape
from yolov9.models.common import DetectMultiBackend as _DetectMultiBackend
from yolov9.utils.general import check_img_size, non_max_suppression

# Define the enhanced detector with face replacement capabilities
def crowdface(auto_shape=True):
    """
    Load the CrowdFace model with enhanced face detection and processing capabilities.
    
    Arguments:
      auto_shape (bool): Automatically adjust input shapes.
    Returns:
      CrowdFace model with advanced functionalities.
    """
    url = "https://github.com/Ultrabrain/CrowdFace/releases/download/yolov9FaceRecognition/best.pt"
    model = _DetectMultiBackend(weights=url)
    if auto_shape: 
        model = _AutoShape(model)
    return CrowdFaceModel(model)

class CrowdFaceModel:
    def __init__(self, model):
        self.model = model
        # Pre-load or define any resources needed for face processing (e.g., overlay images)

    def process_frame(self, frame, overlay_img_path, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
        """
        Process a single frame for face detection and apply face replacement/enhancements.
        
        Arguments:
          frame: Frame to process.
          overlay_img_path: Path to the overlay image for face replacement.
          conf_thres: Confidence threshold for detections.
          iou_thres: IOU threshold for non-max suppression.
          classes: Target classes for detection.
          agnostic_nms: Apply class-agnostic NMS.
        """
        # Load the overlay image
        overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
        overlay_img = self.prepare_overlay_image(overlay_img)

        img = torch.from_numpy(frame).to(self.model.device)
        img = img.float() / 255.0  # Normalize image
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension

        # Perform detection
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)

        # Apply face replacement or enhancements on detections
        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    self.apply_face_replacement(frame, list(map(int, xyxy)), overlay_img)

        return frame

    def prepare_overlay_image(self, overlay_img):
        # Implement any preprocessing needed for the overlay image
        return overlay_img

   def apply_face_replacement(self, frame, xyxy, overlay_img):
    x1, y1, x2, y2 = xyxy
    face_region = frame[y1:y2, x1:x2]

    # Apply Gaussian blur for privacy enhancement
    blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)

    # Blend the blurred face with the original face region
    alpha = 0.5  # Adjust alpha between 0 to 1 for blending ratio
    blended_face = cv2.addWeighted(face_region, 1 - alpha, blurred_face, alpha, 0)

    # Replace the original face region with the blended one
    frame[y1:y2, x1:x2] = blended_face

