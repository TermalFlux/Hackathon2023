# This is a sample Python script.
import cv2 as cv
import mediapipe as mp

class DetectedObjects:
    object_id: int
    object_box: mp.tasks.components.containers.BoundingBox

    def __init__(self, object_id:int, object_box:mp.tasks.components.containers.BoundingBox) -> None:
        self.object_id = object_id
        self.object_box = object_box
        pass

def detect(frame):
    person = 1
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    object_detector_results = object_detector.detect(mp_image)
    detectedObjects = list()

    for detection in object_detector_results.detections:
        detectedObjects.append(DetectedObjects(person, detection.bounding_box))
        person += 1
    for p in detectedObjects:
        x = p.object_box.origin_x
        y = p.object_box.origin_y
        w = p.object_box.width
        h = p.object_box.height
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, f"Person {person}", (x, y), cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv.putText(frame, f"Total persons: {person-1}", (40, 440), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)

def detectByCamera(writer):
    video = cv.VideoCapture(1)
    while True:
        check, frame = video.read()
        if writer is not None:
            writer.write(frame)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        detect(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow("Window", frame)
        if cv.waitKey(1) == ord('q'):
            break
    video.release()
    cv.destroyAllWindows()
    return

model_path = 'efficientdet_lite2_uint8.tflite'

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
ImageClassifier = mp.tasks.vision.ImageClassifier
ImageClassifierOptions = mp.tasks.vision.ImageClassifierOptions
VisionRunningMode = mp.tasks.vision.RunningMode

object_options = ObjectDetectorOptions(
    base_options=BaseOptions(model_path),
    running_mode=VisionRunningMode.IMAGE,
    score_threshold=0.60,
    category_allowlist="person"
)

image_options = ImageClassifierOptions(
    base_options=BaseOptions(model_path),
    running_mode=VisionRunningMode.IMAGE,
    max_results=10
)

object_detector = ObjectDetector.create_from_options(object_options)
image_classifier = ImageClassifier.create_from_options(image_options)

detectByCamera(None)