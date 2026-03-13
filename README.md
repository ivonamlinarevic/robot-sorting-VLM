# PyBullet Robotic Arm Simulation with Vision-Language Models

A Python simulation that combines PyBullet robotic arm control with zero-shot image-text matching capabilities using the CLIP vision-language model. This project was created as part of exploration into applying vision-language models in robotics.

## Overview

This project demonstrates:
- **Physics-based robotic arm simulation** using PyBullet
- **Vision-guided object selection** using CLIP vision-language model
- **Intelligent pick and place operations** based on text prompts

The robot captures images from its camera, analyzes them using CLIP, selects objects based on text descriptions, and performs pick-and-place operations, including color sorting and interactive placement.

## Project Structure

```
├── .gitignore                             # Files ignored by Git
├── panda_vision_simulation.py             # Vision-guided robot simulation Class
├── sorting_vlm.py                         # Color, shape and size sorting and interactive demo with VLM
├── simple_pick_place_demo.py              # Simple pick and place demo
├── requirements.txt                       # Python dependencies
└── README.md                              # This documentation
```

## Features

### Vision-Guided Robot Simulation
- **Camera image capture** from PyBullet simulation
- **Object detection** using 3D-to-2D projection and segmentation masks
- **Vision-language matching** with **CLIP** for object selection
- **Text-prompted pick and place** ("pick up the red cube")
- **Multiple objects** (red cube, red sphere, blue cube, green sphere, yellow cylinder)
- **Color, shape and size sorting zones** with physical borders to prevent objects from falling
- **Interactive zone selection and throw action** for object placement

## Requirements

### Basic Requirements (for robot simulation)
- Python 3.7+
- PyBullet
- NumPy

### Full Requirements (for vision integration)
- All basic requirements plus:
- PyTorch
- Transformers (Hugging Face)
- Pillow (PIL)
- Matplotlib
- Requests

## Installation
1. Clone or download this project:
   ```bash
   git clone https://github.com/ivonamlinarevic/robot-sorting-VLM.git
   ```

2. (Recommended) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Color Sorting and Interactive Demo

To run the color sorting and interactive demo:
```bash
python3 sorting_vlm.py
```

When you start the demo, you will be prompted to choose a mode:

1. **Automatic Color/Shape/Size Sorting**: The robot will automatically sort all objects into their matching zones.
2. **Interactive Text Prompt**: You can enter text prompts to select which object the robot should pick up, and choose where to place it ("pick up the yellow object").
3. **Natural Language Commands**: This mode allows the robot to follow simple natural language instructions ("move the blue object to the red box").

Follow the on-screen instructions to interact with the robot and sorting zones.

---





## Technical Details

### CLIP Vision-Language Matching

The robot uses CLIP to match images of objects to your text prompt:

```python
# For each object crop, CLIP returns a similarity score with the text prompt
similarity_scores = model.compute_object_similarity(crops, text_prompt)
selected_object, best_score = model.select_best_object(similarity_scores)
# Robot picks and places the selected object
```

CLIP compares each cropped object image to your prompt (like "red cube") and returns a score for each. The robot picks the object with the highest score and moves it as you choose.

### Inverse Kinematics

The robot uses PyBullet's built-in inverse kinematics solver to calculate joint angles needed to reach target positions:

```python
joint_positions = p.calculateInverseKinematics(
    self.robot_id,
    endEffectorLinkIndex=11,  # Panda end-effector link
    targetPosition=target_position,
    targetOrientation=target_orientation
)
```

### Robot Control

Joint control uses position control mode:
- **Position Control**: Joints move to target positions
- **Gripper Control**: Two-finger gripper with synchronized motion

## Future Enhancements

Possible extensions to this project:
- Enable multi-object reasoning for commands involving relationships (e.g., “Stack all blue cubes, then put the red sphere on top”).

## Resources

- [PyBullet Documentation](https://pybullet.org/)
- [Panda Robot Specifications](https://www.franka.de/technology)
- [Robotics and Physics Simulation Tutorial](https://pybullet.org/wordpress/)
