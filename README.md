
# Autopilot: An advanced perception, localization and path planning techniques for autonomous vehicles using YOLOv7 and MiDaS 
## Research Paper 📄
Implementation of paper -  [Devmurari, Harshkumar, Gautham Kuckian, and Prajjwal Vishwakarma. "AUTOPILOT: An Advanced Perception, Localization and Path Planning Techniques for Autonomous Vehicles using YOLOv7 and MiDaS."(ICACTA), IEEE, 2023](https://ieeexplore.ieee.org/document/10393218)


## Appendix 

Autopilot system is designed with motive of fully self driving system. The autopilot system takes the video as input. It can be directly taken from webcam or a saved video of dashcam from internet. That video is then processed frame by frame. Each frame produces the result of both object detection and depth estimation. First the frame is passed on to YOLOv7 state-of-the-art rapid object detector released just a few months ago. The machine learning model is trained and validated from COCO dataset. It is capable of identifying 80 different types of objects including car, trucks, motorcycles, pedestrians, animals, etc. Each obstacle is identified and detected by making bounding box around them. These boxes also provide with probability of object which determines accuracy of the detector with minimum prediction of 75% then frame is passed on to MiDaS which is trained on 12 different datasets and produces inferno colour depth map representing depth of pixels present inside frame. The road has been  transformed to bird-eye-view using perspective transform for drivable space and A* routing algorithm has been used for path planning to determines the optimal and quickest route between two points. This plays critical role in decision making process for self driving without human intervention.


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

## Output 🏅
#### Output figure of YOLO Object Detection
![object](https://github.com/Harsh19012003/Autopilot/assets/94838404/a2195a31-049a-44f4-8598-48b817bd7cf5) 

#### Output figure of MiDas Depth Perception
![depth](https://github.com/Harsh19012003/Autopilot/assets/94838404/ab868eca-c1cd-48f6-bf8b-2b9d16437be7)

#### Output figure of Top View Localization
![top](https://github.com/Harsh19012003/Autopilot/assets/94838404/dc32f98e-d308-4640-8c98-e9f7368116f2)

#### Output figure of Path Planning
![path](https://github.com/Harsh19012003/Autopilot/assets/94838404/672d1881-c9ae-4d6c-92e6-8c1a1bdc1043)

## Citation
```
@inproceedings{devmurari2023autopilot,
  title={AUTOPILOT: An Advanced Perception, Localization and Path Planning Techniques for Autonomous Vehicles using YOLOv7 and MiDaS},
  author={Devmurari, Harshkumar and Kuckian, Gautham and Vishwakarma, Prajjwal},
  booktitle={2023 International Conference on Advanced Computing Technologies and Applications (ICACTA)},
  pages={1--7},
  year={2023},
  organization={IEEE}
}
```
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

