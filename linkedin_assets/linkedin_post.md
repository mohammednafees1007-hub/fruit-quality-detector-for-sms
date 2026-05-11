# LinkedIn Post Draft

I built **FruitVision AI**, a fruit quality detection and grading web system for a Smart Manufacturing Systems project.

The goal was to move beyond only detecting the fruit name and make the system quality-first: given an uploaded or camera-captured fruit image, the system predicts whether the fruit is **Fresh**, **Adulterant / Damaged**, or **Rotten**, then maps it to a practical grade:

- Fresh -> Grade A
- Adulterant / Damaged -> Grade B
- Rotten -> Grade C

What has been implemented:

- FastAPI web app with upload and camera capture
- Robust Camera Mode for better camera and phone-screen inputs
- MobileNetV2-based Keras quality classifier
- YOLO-based fruit region support and annotated output
- Optional fruit naming model toggle
- Non-fruit filtering to avoid forcing a grade when the image is not reliable
- Performance testing script for 50-image evaluation
- Dashboard-style UI showing grade, quality, confidence, probability distribution, recommendation, and recent scans

Current evaluation:

- Holdout test accuracy: 76.79%
- Holdout macro-F1: 76.54%
- 50-image master test accuracy: 72.00%
- 50-image macro-F1: 71.43%
- Average quality confidence: 76.28%
- Average quality-only inference time: about 74 ms per image

Capabilities:

- Grades fruit into A / B / C quality categories
- Supports fresh, adulterated/damaged, and rotten classes
- Works from image upload and camera capture
- Shows confidence and class probabilities
- Rejects uncertain or non-fruit inputs instead of forcing a wrong grade
- Designed with a future Raspberry Pi 5 deployment path in mind

Limitations:

- The model is trained on a limited public dataset, so results can vary with lighting, camera angle, blur, and unseen fruit types
- Rotten and adulterated/damaged classes can visually overlap, which causes some confusion
- Toy/plastic fruit rejection is improved with filtering, but true fake-vs-real detection would require more negative training examples
- Current web version still uses heavier supporting models; future edge deployment should use a lightweight YOLO model plus a TFLite quality classifier

Next steps:

- Expand the dataset with more real-world fruit images
- Add more negative examples such as toy fruits, packaged images, and non-fruit objects
- Convert the quality model to TensorFlow Lite
- Benchmark on Raspberry Pi 5
- Improve rotten/adulterated recall with better dataset balance and augmentation

This project helped me understand how computer vision can support automated inspection and quality grading in smart manufacturing workflows.

#ComputerVision #MachineLearning #DeepLearning #SmartManufacturing #FastAPI #TensorFlow #YOLO #RaspberryPi #AI #FruitQualityDetection
