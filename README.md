
# Autopilot: An advanced perception, localization and path planning techniques for autonomous vehicles using YOLOv7 and MiDaS 

A Project of Self-Driving Car 🚘


## Appendix 

Autopilot system is designed with motive of fully self driving system. The autopilot system takes the video as input. It can be directly taken from webcam or a saved video of dashcam from internet. That video is then processed frame by frame. Each frame produces the result of both object detection and predicts drivable area on the lane. First the frame is passed on to yolov7 state-of-the-art rapid object detector released just a few months ago. The machine learning model is trained and validated from coco dataset. It is capable of identifying 80 different types of objects including car, trucks, motorcycles, pedestrians, animals, etc. Each obstacle is identified and detected by making bounding box around them. These boxes also provide with probability of object which determines accuracy of the detector with minimum prediction of 75%. Lane line detection has many steps in pre-processing of image. Pre-processing includes converting to hls, Greyscale, thresh, canny edge detection and Gaussian blur. It is then transformed to bird-eye-view using perspective transform for prediction. These result are then passed on to inverse perspective transform and polygon is created for drivable space. Lane line is highlighted with a green colour polygon giving the sense of environment by producing the drivable area of lane. This plays critical role in decision making process for self driving without human intervention.

Download weights and raw working video 🔗
```bash
  https://drive.google.com/drive/folders/1mbdWxvgio4PUA0iUTrQgK-s_ZbNdX_3h
```


## Tech Stack

**Language:** Python
**Frameworks:** tensorflow, pytorch


## Run Locally 💻

Clone the project

```bash
  git clone https://github.com/Harsh19012003/Autopilot
```

Go to the project directory

```bash
  cd Autopilot
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run

```bash
  python detect.py
```



## Features 📌

- Realtime low latency
- Upto 80 different objects
- Highly effective in bad weather 
- Real time Depth Perception
- Occupancy Tracker
- Optimal Path Planning



## Roadmap 🛣️

- Perception
    - Object Detection (YOLOv7 Algorithm)
    - Depth Perception (MiDaS)

- Localization
    - Top View Transformation (Perceptive Transformation)

- Path Planning
    - A* Routing Algorithm

## Feedback 📝

If you have any feedback, please reach out to us at harshdevmurari007@gmail.com

## Contributing 🤝

Contributions are always welcome!

Contact harshdevmurari007@gmail.com for ways to get started.



## Support

For support, email harshdevmurari007@gmail.com


## Authors 👨🏻‍💻

- [@harshdevmurari](https://github.com/Harsh19012003)
- [@gauthamkuckian](https://github.com/gauthamkuckian)
- [@prajjwalvishwakarma](https://github.com/PrajjwalV27)

