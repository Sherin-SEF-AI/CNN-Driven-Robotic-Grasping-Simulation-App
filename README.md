# AI-Driven Robotic Grasping Simulation Application

A complete PyQt6 + PyBullet application for simulating 6-DOF robotic pick-and-place tasks with mock CNN-based grasp prediction.

## Features

- **Professional PyQt6 GUI** with three-panel layout
- **PyBullet Physics Simulation** with Kuka IIWA 7-DOF robot arm
- **Mock CNN Grasp Predictor** simulating AI-based grasp pose estimation
- **Inverse Kinematics (IK)** trajectory execution
- **Real-time Object Manipulation** with interactive sliders
- **Complete Pick-and-Place Workflow** in 5 trajectory steps

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the application
python robotic_grasping_app.py
```

## Workflow

1. **Click "1. Initialize Scene"** - Sets up the PyBullet environment with robot, table, and object
2. **Click "2. Predict Grasp (CNN Mock)"** - Simulates CNN grasp prediction and displays 6-DOF pose
3. **Click "3. Execute Pick & Lift"** - Executes the IK trajectory to grasp and lift the object

## Application Structure

### GUI Panels

- **Left Panel**: PyBullet simulation view (opens in separate window)
- **Top-Right Panel**:
  - Workflow control buttons
  - Object position sliders (X, Y, Z) for testing
- **Bottom-Right Panel**:
  - 6-DOF Grasp Pose table (Position + Orientation)
  - Status log with timestamped messages

### IK Trajectory Phases

1. **Approach**: Move to pre-grasp position (15cm above object)
2. **Descent**: Lower to grasp point
3. **Grasp**: Apply constraint to simulate gripper closure
4. **Lift**: Raise object to safe height (20cm above table)
5. **Stop**: Maintain final position

## Technical Details

- **Robot Model**: Kuka IIWA (7-DOF, end-effector index: 6)
- **Simulation Rate**: 240 Hz physics step
- **Grasp Simulation**: Fixed constraint between end-effector and object
- **Mock CNN**: Returns plausible top-down grasp with randomized orientation
- **Camera**: Virtual top-down view for grasp prediction input

## Dependencies

- Python 3.8+
- PyQt6 (GUI framework)
- PyBullet (physics simulation)
- NumPy (numerical operations)
- Pillow (image processing)

## Notes

- PyBullet GUI window opens separately from the PyQt control panel
- Object position can be adjusted via sliders before grasp prediction
- The application demonstrates the complete workflow without requiring actual CNN training
- All code is contained in a single Python file for easy distribution

## License

MIT License
