# ROS 2 Wind Turbine Bolt Detection

This repository contains Python scripts and ROS 2 tools for wind turbine bolt detection research using computer vision, YOLO, and the Azure Kinect camera.

This repository is intended for internal research and experimentation.

## Running the Project

The Azure Kinect ROS 2 driver and Python scripts should run in **separate terminals**.

### Terminal 1: Start the Azure Kinect

Make sure the Azure Kinect camera is connected.

Launch the ROS 2 camera node:

```bash
ros2 run azure_kinect_ros2_driver azure_kinect_node
```

Keep this terminal running while using the camera.

### Terminal 2: Run Python Scripts

Navigate to the repository directory:

```bash
cd ros2-wind-turbine-bolt-detection
```

Activate the Python virtual environment:

```bash
source .venv/bin/activate
```

You can now run scripts from the repository.

For example, to run YOLO segmentation:

```bash
python yolo_tests/segmentation.py
```

**Note:** Some scripts may require configuration changes, such as updating the YOLO model weights path, before running.

To stop a running script, press `Ctrl+C`.

## Project Structure

```text
ros2-wind-turbine-bolt-detection/
├── classical_cv_tests/   # Traditional computer vision experiments
├── kinect_tests/        # Azure Kinect testing scripts
├── recording_tools/     # Data recording utilities
├── yolo_tests/          # YOLO detection and segmentation scripts
├── requirements.txt     # Python dependencies
└── README.md
```

## Before making any edits

Pull the latest changes:

```bash
git pull origin main
```

## Saving and Pushing Changes

After making changes, save them to GitHub so other team members can access them.

**1. Check your changes**

```bash
git status
```

**2. Stage your changes**

```bash
git add .
```

**3. Commit your changes**

```bash
git commit -m "Brief description of your changes"
```

**4. Push to GitHub**

```bash
git push origin main
```

**Important:** Avoid committing virtual environments, large datasets, model weights, or generated output files unless they are intentionally tracked by the team.

## Prerequisites

Before getting started, make sure you have:

- Python 3 and pip
- Git
- ROS 2 installed and configured
- Azure Kinect camera and the `azure_kinect_ros2_driver` package installed

## First-Time Setup

**1. Clone the repository**

```bash
git clone https://github.com/HiddenDevelop/ros2-wind-turbine-bolt-detection.git
cd ros2-wind-turbine-bolt-detection
```

**2. Create a Python virtual environment**

```bash
python3 -m venv .venv
```

**3. Activate the virtual environment**

```bash
source .venv/bin/activate
```

**4. Install the required Python packages**

```bash
pip install -r requirements.txt
```

You only need to complete these steps once per machine, unless the dependencies change.

## Notes

- Always activate the Python virtual environment before running scripts.
- Make sure the Azure Kinect driver is running before starting scripts that depend on live camera data.
- Run Python commands from the repository root unless a script specifies otherwise.
- Keep your local repository updated to avoid conflicts with other team members.
