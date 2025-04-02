# Security-and-Surveillance-System

### Google Developer Groups x Heliverse Project: *Security Surveillance Automation Using AI Agents*

To initiate the project, we aim to create a basic implementation that demonstrates the functionality of AI agents in a security surveillance system. The focus will be on developing a prototype where AI agents can process video streams to detect and classify basic events, such as identifying individuals, tracking their movements, and flagging unusual activities. This initial system will serve as a proof of concept, showcasing how AI agents can collaborate to enhance surveillance capabilities. Once the foundational system is in place, we will optimize its performance by comparing different algorithms, models, and architectures. Later, we plan to incorporate more complex scenarios and environments to expand the system's functionality and robustness.

#### Team Optimization Ideas:  
- **Sharvin's Idea - Camera Synchronization:** Implement a feature where illegal activity detected in one camera triggers alerts in nearby cameras. For instance, if a suspect leaves the frame of one camera, the adjacent camera can track the suspect seamlessly using a unique bounding box identifier. This synchronization ensures continuous monitoring, allowing authorities to follow the suspect's route effectively.

- **Jefery's Idea - Dynamic Framing:** Develop a system where all cameras operate at a minimal frame rate (e.g., 1 FPS) under normal conditions. If a potential threat is detected, the system dynamically scales up the frame rate for relevant cameras, capturing the incident in greater detail. For example, cameras detecting simultaneous robberies on different floors can automatically adjust their FPS, ensuring critical events are recorded with higher precision and appropriate alarms are raised.

This iterative development approach, starting from a basic demonstration to gradual optimization and expansion, ensures a clear path for building a robust and scalable surveillance system.

---

## *Workflow Design Overview*

1. *Data Collection and Camera Setup*  
   - *Action*: Deploy a network of cameras in strategic locations, ensuring optimal coverage of critical areas.  
   - *Purpose*: Gather live video streams for processing. Cameras should be selected based on purpose (e.g., entrance, hallway, parking area) to ensure the right type of agent is deployed per camera.

2. *Video Stream Processing*  
   - *Action*: Each camera streams live video, which is analyzed by dedicated AI agents.  
   - *Technology*: Use deep learning models like YOLO (You Only Look Once) for object detection (e.g., people, vehicles). LLMs (Large Language Models) are integrated for scene understanding (e.g., activity recognition or contextual analysis).  
   - *Proactive Monitoring*: Use algorithms that actively predict when certain events may become a security threat (e.g., crowd gathering, unauthorized access). AI agents continuously analyze streams, looking for patterns and behavior indicative of security concerns.

3. *Event Identification and Classification*  
   - *Action*: After processing video data, AI agents detect events such as unusual behavior, crowd density, or restricted area access.  
   - *Technologies*: YOLO for precise counting of people/objects, LLM for scene/context understanding, and additional models for anomaly detection (e.g., changes in movement patterns, sudden crowding).  
   - *Proactive Alerts*: Raise an alert when an agent identifies an anomaly. Use cascading AI systems (from low-level detections to higher-level classification) to ensure no critical issues are missed.

4. *Semi-Supervised Feedback Loop*  
   - *Action*: Initially, human operators review flagged security concerns to verify and adjust detection thresholds, while AI models continuously improve.  
   - *Technology*: Employ semi-supervised learning for continuous model training, where AI agents propose security concerns, and operators validate these alerts. The feedback from operators is used to train the model for better future decision-making.

5. *Notification and Communication*  
   - *Action*: Once an issue is flagged, an AI agent determines the level of urgency (e.g., minor vs. critical) and sends notifications.  
   - *Semi-Supervised Notifications*: Initially, authorities are notified of potential security risks with a suggestion of the concern's priority. Over time, this becomes fully automated as the models learn from the operator's feedback.  
   - *Talkback Feature*: The camera modules should include an option to send real-time instructions (e.g., "Move away from the area") via audio or text, enabling proactive intervention.

6. *Post-Incident Analysis and Model Refinement*  
   - *Action*: After incidents, use the feedback to retrain models, identifying any missed events or false positives.  
   - *Continuous Refinement*: Employ continual improvement processes, adjusting AI agent thresholds, refining scene understanding, and re-training the system to avoid both missed concerns and false positives.  
   - *Technology*: A combination of supervised and unsupervised learning can be applied, with human input used to label edge cases and fine-tune model performance.

7. *Reduction of False Positives*  
   - *Action*: Integrate cascading layers of AI agents where lower-level agents (e.g., basic object detection) feed into more complex agents (e.g., behavior analysis). This minimizes unnecessary false alarms.  
   - *Continuous Feedback Loop*: AI agents refine thresholds to reduce error rates and optimize the system to handle more complex scenarios over time.

---

## *Bottom-up Task List*

### *1. Data Collection and Camera Integration*
#### *Tasks*:
- *Hardware Integration*:  
  - Identify and set up cameras (hardware drivers and software APIs).  
  - Test camera feeds for stability and ensure compatibility with your software framework.
- *Video Stream Handling*:  
  - Write code to handle live video streams using libraries like OpenCV or GStreamer.  
  - Implement efficient video compression and streaming protocols (e.g., RTSP, WebRTC).
- *Camera Assignment*:  
  - Design a system to assign specific cameras to monitor defined zones (e.g., entrances, parking areas).  
  - Create a database to log camera positions, zones, and configurations.

### *2. Video Stream Preprocessing*
#### *Tasks*:
- *Frame Extraction*:  
  - Extract video frames for analysis at a configurable frame rate.  
  - Ensure frame buffering to avoid data loss during high throughput.
- *Data Preprocessing*:  
  - Normalize images (resize, scale pixel values) for input to deep learning models.  
  - Remove noise and optimize images for object detection (e.g., filtering and sharpening).
- *Edge Computing Setup* (if applicable):  
  - Write code to enable preprocessing directly on edge devices to reduce server load.

### *3. AI Model Development and Integration*
#### *Tasks*:
- *Object Detection Models (YOLO)*:  
  - Implement and train YOLO models for object detection using TensorFlow or PyTorch.  
  - Create a pipeline for detecting objects like people, vehicles, or bags.  
  - Optimize models for real-time performance using tools like ONNX or TensorRT.
- *Scene Understanding (LLMs)*:  
  - Integrate pre-trained LLMs (e.g., OpenAI's GPT, Hugging Face models) to analyze scene context.  
  - Create an interface between detected objects and LLMs for contextual reasoning (e.g., "A crowd is forming").  
- *Anomaly Detection Models*:  
  - Develop models to detect behavioral anomalies or changes in patterns (e.g., crowd density increase).  
  - Train and validate models using surveillance datasets.
- *Cascading AI System*:  
  - Design a cascading architecture where lower-level models (object detection) feed results into higher-level models (behavioral analysis).  
  - Implement logic to prioritize alerts based on confidence levels and severity.

### *4. Event Detection and Classification*
#### *Tasks*:
- *Event Logging*:  
  - Create a database to log detected events with timestamps, location, and context.  
  - Implement an efficient storage format for event metadata and video snippets.
- *Alert Thresholds*:  
  - Write algorithms to define and adjust thresholds for alerts (e.g., number of people in a restricted area).  
  - Implement adaptive threshold logic based on operator feedback.
- *Event Prioritization*:  
  - Build logic to classify events by urgency (e.g., minor vs. critical).  
  - Assign scores to events based on anomaly severity.

### *5. Notification and Communication System*
#### *Tasks*:
- *Notification Module*:  
  - Write code to send notifications via email, SMS, or app-based alerts.  
  - Use APIs like Twilio or Firebase for real-time notifications.
- *Talkback Feature*:  
  - Develop a system to send audio/text messages to the scene (e.g., TTS for audio instructions).  
  - Write a UI for operators to issue real-time commands.
- *Operator Dashboard* (Not a Priority):  
  - Build an interactive dashboard to display live streams, alerts, and event details.  
  - Use web frameworks like React, Angular, or Flask.

### *6. Semi-Supervised Feedback Loop*
#### *Tasks*:
- *Human Review Interface*:  
  - Design an interface for operators to review flagged events and provide feedback.  
  - Log operator responses in a database for model retraining.
- *Feedback Integration*:  
  - Write scripts to integrate operator feedback into model retraining pipelines.  
  - Develop an auto-labeling system to use operator feedback for supervised learning.
- *Active Learning Mechanism*:
  - Implement active learning to retrain models using flagged edge cases.
  - Prioritize retraining on high-error categories (e.g., frequent false positives).

### *7. Post-Incident Analysis and Continuous Improvement*
#### *Tasks*:
- *Incident Replay Tool*:
  - Develop a tool to replay flagged incidents and analyze why events were triggered.
  - Highlight AI decisions for operator review.
- *Model Retraining Pipelines*:
  - Set up automated pipelines for retraining models with new data.
  - Include hyperparameter tuning and model evaluation scripts.
- *Performance Monitoring*:
  - Write scripts to monitor model performance (e.g., detection accuracy, false positive rate).
  - Generate periodic performance reports for the team. (Not a Priority)

### *8. Privacy and Security*
#### *Tasks:
- *Data Encryption*:
  - Encrypt all video streams and event logs. Use SSL/TLS for data transmission.
- *Access Control*:
  - Implement role-based access control (RBAC) for operators and administrators.
  - Write code to track and log user actions within the system.
- *Privacy Filters*:
  - Develop algorithms to blur sensitive areas or anonymize faces in video streams.

---

## *AI Agent Details*

### *1. AI Agent Architecture*
#### *Approach*:
- Design agents as modular, task-specific components, each responsible for a well-defined function.  
- Use a *multi-agent system (MAS)* architecture where agents can communicate and collaborate.  
- Build agents using microservices to allow independent deployment and scaling.

### *2. Define Roles and Responsibilities for AI Agents*
#### *Example AI Agents*:
1. *Object Detection Agent*:  
   - Detects and tracks objects (e.g., people, vehicles) in real-time.  
   - Powered by YOLO or similar models.  
   - Outputs bounding boxes, labels, and confidence scores.

2. *Anomaly Detection Agent*:  
   - Monitors behavioral patterns (e.g., crowd formation, loitering) to identify anomalies.  
   - Uses temporal models like LSTMs, CNNs, or unsupervised learning methods.  
   - Flags events as "normal" or "anomalous."

3. *Scene Understanding Agent*:  
   - Analyzes context using detected objects and events.  
   - Uses LLMs to classify events (e.g., "a crowd forming near an exit").  
   - Outputs event summaries for decision-making.

4. *Proactive Alert Agent*:  
   - Monitors events flagged by other agents.  
   - Decides urgency levels and triggers appropriate notifications or actions.  
   - Can issue commands (e.g., "Evacuate the area") using predefined rules.

5. *Feedback Learning Agent*:  
   - Collects operator feedback to improve detection thresholds and retrain models.  
   - Suggests updates to detection models based on flagged edge cases.

### *3. Communication Between AI Agents*
#### *Approach*:
- Use *message-passing protocols* to enable agents to share data (e.g., events, alerts).  
- Implement communication standards like *gRPC, **MQTT, or **REST APIs* for real-time communication.

#### *Examples*:
- *Object Detection Agent* → *Scene Understanding Agent*:  
  - Sends detected objects and their coordinates for contextual analysis.  
- *Anomaly Detection Agent* → *Proactive Alert Agent*:  
  - Flags unusual behaviors for escalation.  
- *Feedback Learning Agent* ↔ All Agents:  
  - Updates model parameters and thresholds based on human input.

### *4. Integration into the Workflow*
#### *Step-by-Step Integration*:
1. *Data Input*:  
   - The *Object Detection Agent* processes raw video frames and identifies objects.  
2. *Event Analysis*:  
   - The *Anomaly Detection Agent* analyzes object trajectories or behaviors over time.  
   - The *Scene Understanding Agent* classifies the context using data from the Object Detection Agent.
3. *Decision-Making*:  
   - The *Proactive Alert Agent* prioritizes flagged events and triggers actions.  
   - The *Talkback Feature* or real-time instructions are handled by this agent.
4. *Feedback Loop*:  
   - Human operators review flagged events through a dashboard.  
   - Feedback is processed by the *Feedback Learning Agent* to refine models.

### *5. Key Technologies for Agent Development*
#### *AI Models*:
- Object Detection: YOLOv8, Faster R-CNN, SSD.  
- Anomaly Detection: Autoencoders, One-Class SVM, LSTMs, or Gated Recurrent Units (GRUs).  
- Scene Understanding: OpenAI GPT, Hugging Face Transformers.

---

## File Structure
```plaintext
Security-and-Surveillance-System/
├── __init__.py
├── requirements.txt
├── README.md
├── data/
│   ├── ucf_crime_dataset_download.py
│   └── logs/
├── scripts/
│   ├── videostreamhandling/
│   │   ├── streamhandler.py
│   │   └── final_streamhandler.py
│   ├── ai_agents/
│   │   ├── Anomaly.py
│   │   ├── app.py
│   │   ├── scene_understanding_blip_salesforce.py
│   │   ├── scene_understanding_llava.py
│   │   └── scene_understanding_clip_sopenai.py
│   ├── obj_det/
│   │   └── YOLODetector.py
│   ├── pipelining/
│   │   └── TaskDispatcher.py
│   ├── preprocessing/
│   │   ├── FrameProcessor.py
│   │   └── PreprocessingManager.py            
│   ├── notification/
│   │   ├── app.py
│   │   ├── notif.py
│   │   └── index.html               
│   └── utils/
│       ├── logger.py                 
│       └── util.py
├── models/
│   ├── models.txt               
│   ├── yolov8_face_detection_trainer.pt  
│   └── yolov8_face_detection_trainer.py  
└── tests/                   # Unit and integration tests
    ├── assets/
    │   ├── face_detection_test.png
    │   ├── output_video.mp4
    │   ├── video1.mp4
    │   ├── video2.mp4
    │   ├── video3.mp4
    │   ├── video4.mp4
    │   ├── video5.mp4
    │   └── YOLODetector_test.jpg     
    ├── frameprocessing_test.py
    ├── scene_understanding_test.py
    ├── YOLODetectorTest.py
    ├── Integration_test.py       
    └── logger_test.py
```
