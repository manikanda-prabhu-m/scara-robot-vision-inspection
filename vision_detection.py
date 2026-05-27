import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = 'exported-models/my_model1/saved_model'

def load_model(path):
    print("Loading TensorFlow model...")
    model = tf.saved_model.load(path)
    return model

def detect(model, frame):
    """
    Run SSD MobileNet V2 inference on a single frame.
    Returns detections: boxes, scores, classes.
    """
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)
    return detections

def get_object_center(detections, frame_shape, threshold=0.5):
    """
    Extract XY center coordinates of detected object
    above confidence threshold.
    """
    h, w = frame_shape[:2]
    scores = detections['detection_scores'][0].numpy()
    boxes = detections['detection_boxes'][0].numpy()

    for i, score in enumerate(scores):
        if score >= threshold:
            ymin, xmin, ymax, xmax = boxes[i]
            cx = int((xmin + xmax) / 2 * w)
            cy = int((ymin + ymax) / 2 * h)
            return cx, cy
    return None

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect(model, frame)
        center = get_object_center(detections, frame.shape)
        if center:
            print(f"Object detected at: {center}")
            cv2.circle(frame, center, 8, (0, 255, 0), -1)
        cv2.imshow('Inspection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
