# RBE 595 – Hands-On Aerial Robotics 🚁

This repository contains my completed project deliverables for **RBE 595: Hands-On Aerial Robotics**.  
Each folder corresponds to a project milestone in the course, progressing from setup → filtering/state estimation → planning → racing → final autonomy.

---

## 📌 Projects Overview

### **P0: Alohomora!**
Initial setup + tooling + baseline pipeline to get everything running reliably.

### **P1: Magical Filtering!**
State estimation and filtering for attitude/pose understanding.

- **P1a: Magic Madgwick Filter for Attitude Estimation**  
  Implemented and evaluated Madgwick filtering for attitude estimation.

- **P1b: Non-stinky Unscented Kalman Filter for Attitude Estimation**  
  Implemented and evaluated an UKF-based attitude estimation pipeline.

### **P2: Path Following!**
Planning + execution for flying through structured environments.

- **P2a: Tree Planning Through The Trees!**  
  Tree-based planning (e.g., sampling-based search) to navigate cluttered spaces.

- **P2b: Fly through boxes!**  
  Path following and trajectory execution through constrained “gate/box” layouts.

### **P3: Mini Drone Race!**
Integrated perception + planning + control to complete a timed mini race.

### **P4: RAFT – Navigating Through Unknown**
Used **RAFT optical flow** for navigating or reasoning in unknown/uncertain environments (perception-driven autonomy).

### **P5: The Final Race!**
Full pipeline integration for the final end-to-end racing/autonomy challenge.

---

## 📁 Repository Structure

```bash
.
├── Group3_p0/          # P0: Alohomora!
├── Group3_p1a/         # P1a: Madgwick Filter
├── Group3_p1b/         # P1b: UKF for attitude estimation
├── Group3_p2a/         # P2a: Tree planning
├── Group3_p2b/         # P2b: Fly through boxes
├── Group3_p3/          # P3: Mini drone race
├── Group3_p4/          # P4: RAFT – Navigating through unknown
├── Group3_p5/          # P5: The final race
├── .gitattributes      # Git LFS config (large assets/models)
├── .gitignore          # Build + cache ignores
└── README.md
```
## 🧠 What This Repo Covers
- State Estimation: Madgwick filter, UKF attitude estimation
- Planning: Tree-based planning, constrained navigation
- Control: Closed-loop path following and trajectory execution
- Perception: Optical-flow-based reasoning with RAFT
- System Integration: End-to-end autonomy for race tasks

## 🛠 Tools / Stack
- Python
- Simulation: Vizflyt
- Libraries: NumPy, OpenCV, SciPy, PyTorch (RAFT)

## ▶️ Running a Project
Each project folder contains its own code and assets with README individually.

## Project Results
All the results of this course can be found in this link : https://drive.google.com/drive/folders/1rceuIVurChsXd5K2Ko4JZuTTqv9dQpId

## 👤 Author
Shakthi Bala
M.S. Robotics Engineering — Worcester Polytechnic Institute (WPI)
