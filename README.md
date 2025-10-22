# AI-Driven Robotic Grasping Simulation Application

A complete PyQt6 + PyBullet application for simulating 6-DOF robotic pick-and-place tasks with mock CNN-based grasp prediction.

## 👨‍💻 Author

**Sherin Joseph Roy**
📧 Email: sherin.joseph2217@gmail.com
🔗 GitHub: [Sherin-SEF-AI](https://github.com/Sherin-SEF-AI)

## 🚀 Two Versions Available

### **Basic Version** (`robotic_grasping_app.py`)
Simple, straightforward implementation perfect for learning and basic demonstrations.

### **Advanced Version** (`robotic_grasping_app_advanced.py`) - ⭐ RECOMMENDED
Feature-rich professional application with extensive capabilities (see Advanced Features below).

## ✨ Core Features (Both Versions)

- **Professional PyQt6 GUI** with multi-panel layout
- **PyBullet Physics Simulation** with Kuka IIWA 7-DOF robot arm
- **Mock CNN Grasp Predictor** simulating AI-based grasp pose estimation
- **Inverse Kinematics (IK)** trajectory execution
- **Real-time Object Manipulation** with interactive sliders
- **Complete Pick-and-Place Workflow** with multiple trajectory steps

## 🎯 Advanced Features (Advanced Version Only)

### 1. **Multiple Grasp Candidates**
- Generate 1-10 grasp candidates per prediction
- Each candidate includes:
  - Confidence score (0-100%)
  - Grasp type (top-down, side, angular)
  - Full 6-DOF pose (position + orientation)
- Visual markers in simulation for all candidates
- Interactive selection from candidate table
- Color-coded by confidence level

### 2. **Trajectory Visualization & Preview**
- Preview trajectory before execution
- Visual path display with waypoint markers
- Enhanced 7-step trajectory:
  1. Approach
  2. Pre-Grasp
  3. Descent
  4. Grasp
  5. Lift
  6. Transport
  7. Place
- Real-time trajectory monitoring

### 3. **Force/Torque Sensing**
- Real-time force feedback from end-effector
- Torque measurement
- Visual force indicator with progress bar
- Maximum force tracking
- Force data recording for analysis

### 4. **Multiple Object Types**
- Cylinder (default)
- Cube
- Sphere
- Duck (complex mesh)
- Random selection
- Each object with unique physics properties

### 5. **Advanced Camera System**
- Multiple viewpoints:
  - Top view (overhead)
  - Side view
  - Front view
- Switch cameras during operation
- Image capture from any angle

### 6. **Collision Detection**
- Real-time collision monitoring
- Collision counter
- Enable/disable toggle
- Visual collision indicators
- Safety system integration

### 7. **Recording & Playback**
- Record complete trajectories
- Save execution data:
  - Object type
  - Initial positions
  - Grasp poses
  - Execution time
  - Success/failure status
  - Force data history
  - Joint position history
- Trajectory history browser
- Export recordings to JSON

### 8. **Performance Metrics**
- Total grasps attempted
- Successful grasps
- Success rate calculation
- Average confidence scores
- Average execution time
- Maximum force recorded
- Collision statistics
- Reset metrics functionality

### 9. **Import/Export System**
- Export grasp configurations to JSON
- Import saved configurations
- Share grasp strategies
- Reproduce experiments
- Batch recording export

### 10. **Enhanced GUI**
- **Tabbed Interface** with 5 sections:
  - 🎮 Basic Controls
  - ⚙️ Advanced Settings
  - 🎯 Grasp Candidates
  - 📊 Metrics & Analytics
  - 🎬 Recording & Playback
- **Real-time Status Indicators**
- **Color-coded feedback**
- **Responsive design**
- **Professional styling**

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Version
```bash
python robotic_grasping_app.py
```

**Workflow:**
1. Click "1. Initialize Scene"
2. Click "2. Predict Grasp (CNN Mock)"
3. Click "3. Execute Pick & Lift"

### Advanced Version (Recommended)
```bash
python robotic_grasping_app_advanced.py
```

**Workflow:**
1. Select object type (Cylinder, Cube, Sphere, Duck, Random)
2. Click "1. Initialize Scene"
3. Set number of grasp candidates (1-10)
4. Click "2. Predict Grasps" - generates multiple candidates
5. Review candidates in "Grasp Candidates" tab
6. (Optional) Click "Preview Trajectory" to visualize path
7. Click "3. Execute Pick & Place"
8. Monitor force feedback and collision detection
9. View performance metrics in "Metrics" tab
10. Export recordings or configurations as needed

## Feature Comparison

| Feature | Basic | Advanced |
|---------|-------|----------|
| GUI Layout | 3-panel | 5-tab interface |
| Grasp Candidates | 1 | 1-10 (configurable) |
| Grasp Types | Top-down only | Top-down, Side, Angular |
| Confidence Scores | ❌ | ✅ |
| Trajectory Steps | 5 | 7 |
| Trajectory Preview | ❌ | ✅ |
| Force/Torque Sensors | ❌ | ✅ |
| Object Types | 1 (Cylinder) | 5 (Cylinder, Cube, Sphere, Duck, Random) |
| Camera Angles | 1 | 3 (Top, Side, Front) |
| Collision Detection | ❌ | ✅ |
| Recording/Playback | ❌ | ✅ |
| Performance Metrics | ❌ | ✅ |
| Import/Export | ❌ | ✅ JSON |
| Visual Markers | ❌ | ✅ |
| Status Indicators | Basic log | Real-time indicators |
| File Size | ~682 lines | ~1400 lines |
| Complexity | Beginner | Advanced |

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

### Basic Version
- **Robot Model**: Kuka IIWA (7-DOF, end-effector index: 6)
- **Simulation Rate**: 240 Hz physics step
- **Grasp Simulation**: Fixed constraint between end-effector and object
- **Mock CNN**: Returns single top-down grasp with randomized orientation
- **Camera**: Single virtual top-down view
- **Lines of Code**: ~680

### Advanced Version
- **Robot Model**: Kuka IIWA with force/torque sensors enabled
- **Simulation Rate**: 240 Hz physics step with real-time monitoring
- **Grasp Simulation**: Advanced constraint system with contact detection
- **Mock CNN**: Multi-strategy predictor with confidence scoring
- **Camera System**: 3 viewpoints (top, side, front)
- **Collision System**: Real-time collision detection and counting
- **Data Recording**: Complete trajectory recording with force data
- **Visualization**: PyBullet debug drawing for markers and paths
- **Lines of Code**: ~1400

## Advanced Features Deep Dive

### Grasp Prediction Algorithm
The advanced mock CNN generates diverse grasp strategies:
- **Top-down grasps**: High confidence (75-95%), vertical approach
- **Side grasps**: Medium confidence (60-85%), horizontal approach
- **Angular grasps**: Lower confidence (50-75%), tilted approach

Each candidate includes:
- Full 6-DOF pose (position + quaternion orientation)
- Confidence score based on grasp type
- Visual marker in simulation
- Color coding in UI (green=high, yellow=medium, red=low)

### Force Sensing System
Real-time force/torque feedback:
- Measures reaction forces at end-effector joint
- Calculates force magnitude: `||F|| = √(Fx² + Fy² + Fz²)`
- Tracks maximum force across all executions
- Stores force history for each recorded trajectory
- Visual progress bar shows current force level

### Collision Detection
Advanced collision monitoring:
- Checks contacts between robot and environment
- Excludes self-collisions and ground contacts
- Maintains collision counter
- Updates metrics automatically
- Can be toggled on/off for testing

### Recording System
Complete trajectory recording includes:
```json
{
  "timestamp": 1234567890.0,
  "object_type": "CYLINDER",
  "initial_position": [0.5, 0.0, 0.65],
  "grasp_pose": {
    "position": [0.5, 0.0, 0.8],
    "orientation": [0, 0, 0, 1],
    "confidence": 0.87,
    "grasp_type": "top-down"
  },
  "execution_time": 3.45,
  "success": true,
  "force_data": [0.0, 0.1, 0.3, ...],
  "joint_positions": [[0, 0, 0, ...], ...]
}
```

## Dependencies

- Python 3.8+
- PyQt6 >= 6.4.0 (GUI framework)
- PyBullet >= 3.2.5 (physics simulation)
- NumPy >= 1.24.0 (numerical operations)
- Pillow >= 9.5.0 (image processing)

## Architecture

### Class Structure (Advanced Version)

```
AdvancedPyBulletSimulation
├── Scene Management
│   ├── initialize_scene()
│   ├── _load_object()
│   └── _reset_robot_pose()
├── Grasp Prediction
│   ├── predict_multiple_grasps()
│   ├── visualize_grasp_candidates()
│   └── select_grasp_candidate()
├── Trajectory Execution
│   ├── execute_pick_and_place()
│   ├── step_trajectory()
│   └── _visualize_trajectory()
├── Sensing & Detection
│   ├── get_force_torque_feedback()
│   └── check_collisions()
├── Recording
│   ├── Recording control
│   └── Trajectory history
└── Data Management
    ├── export_configuration()
    └── import_configuration()

AdvancedRoboticGraspingGUI
├── Tab 1: Basic Controls
├── Tab 2: Advanced Features
├── Tab 3: Grasp Candidates
├── Tab 4: Metrics
└── Tab 5: Recording
```

## Performance

Typical performance on modern hardware:
- **Initialization**: < 1 second
- **Grasp Prediction**: < 0.1 seconds (5 candidates)
- **Trajectory Execution**: 3-5 seconds (7 steps)
- **Collision Check**: < 0.001 seconds per frame
- **Recording Overhead**: Negligible (< 1%)

## Use Cases

### Educational
- Learn robotic grasping concepts
- Understand IK trajectory planning
- Study force feedback systems
- Explore collision detection

### Research
- Test grasp prediction algorithms
- Collect trajectory datasets
- Analyze success rates
- Compare grasp strategies

### Development
- Prototype robot control systems
- Test grasp planning algorithms
- Validate control strategies
- Debug robot behaviors

## Notes

- PyBullet GUI window opens separately from the PyQt control panel
- Object position can be adjusted via sliders before grasp prediction
- The application demonstrates the complete workflow without requiring actual CNN training
- All code is contained in single Python files for easy distribution
- Advanced version includes extensive error handling and validation
- Export/import functionality allows sharing configurations between users

## Troubleshooting

**Issue**: PyBullet window doesn't appear
- **Solution**: Make sure you're not running in a headless environment

**Issue**: Slow simulation
- **Solution**: Reduce number of grasp candidates or disable collision detection

**Issue**: Import fails
- **Solution**: Ensure JSON file format matches expected structure

**Issue**: Force readings always zero
- **Solution**: Force sensor activates only after contact/grasping

## Future Enhancements

Potential additions for future versions:
- [ ] Real CNN integration with TensorFlow/PyTorch
- [ ] Multiple robot models (UR5, Franka Emika, etc.)
- [ ] Advanced gripper models with finger simulation
- [ ] Path planning with obstacle avoidance
- [ ] Reinforcement learning integration
- [ ] ROS (Robot Operating System) integration
- [ ] 3D point cloud processing
- [ ] Multi-object grasping scenarios
- [ ] Grasp quality metrics
- [ ] Video recording of simulations

## Contributing

Contributions are welcome! Areas of interest:
- Additional robot models
- More object types
- Enhanced visualization
- Performance optimizations
- Documentation improvements

For major changes, please open an issue first to discuss what you would like to change.

## Credits

### Project Author
**Sherin Joseph Roy**
- 📧 Email: sherin.joseph2217@gmail.com
- 🔗 GitHub: [Sherin-SEF-AI](https://github.com/Sherin-SEF-AI)

### Technologies Used
- **PyQt6** - Cross-platform GUI framework
- **PyBullet** - Open-source physics simulation
- **NumPy** - Fundamental package for scientific computing
- **Pillow** - Python Imaging Library

### Acknowledgments
Special thanks to:
- The PyBullet community for excellent documentation
- The Qt/PyQt6 team for the powerful GUI framework
- Open-source robotics community

## License

MIT License

Copyright (c) 2025 Sherin Joseph Roy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

**Built with ❤️ by Sherin Joseph Roy**
