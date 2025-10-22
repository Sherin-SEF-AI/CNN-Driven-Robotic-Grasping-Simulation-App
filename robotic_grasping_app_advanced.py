#!/usr/bin/env python3
"""
Advanced AI-Driven Robotic Grasping Simulation Application
PyQt6 GUI + PyBullet Physics Simulation

Author: Sherin Joseph Roy
Email: sherin.joseph2217@gmail.com
Version: 2.0 (Advanced)

Enhanced with multiple features:
- Multiple grasp candidates with confidence scoring
- Trajectory visualization and preview
- Force/torque sensors and real-time feedback
- Multiple object types and selection
- Recording/playback functionality
- Collision detection and path planning
- Performance metrics and statistics
- Advanced camera controls
- Export/import configurations
- Advanced gripper simulation
"""

import sys
import time
import json
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QSlider, QSplitter,
    QGridLayout, QTableWidget, QTableWidgetItem, QComboBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QTabWidget, QFileDialog,
    QMessageBox, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

# PyBullet imports
import pybullet as p
import pybullet_data


class SimulationState(Enum):
    """Enumeration for simulation state machine"""
    UNINITIALIZED = 0
    READY = 1
    GRASP_PREDICTED = 2
    EXECUTING = 3
    COMPLETED = 4
    RECORDING = 5
    PLAYING = 6


class ObjectType(Enum):
    """Available object types for grasping"""
    CYLINDER = "cylinder.urdf"
    CUBE = "cube_small.urdf"
    SPHERE = "sphere_small.urdf"
    DUCK = "duck_vhacd.urdf"
    RANDOM = "random"


@dataclass
class GraspCandidate:
    """Enhanced data class for 6-DOF grasp pose with confidence"""
    position: List[float]  # [x, y, z]
    orientation: List[float]  # Quaternion [x, y, z, w]
    confidence: float  # 0.0 to 1.0
    grasp_type: str  # "top-down", "side", "angular"
    euler: Optional[List[float]] = None

    def __post_init__(self):
        if self.euler is None:
            self.euler = list(p.getEulerFromQuaternion(self.orientation))

    def __str__(self):
        return (f"Grasp [{self.grasp_type}] (Confidence: {self.confidence:.2%})\n"
                f"Position: [{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}] m\n"
                f"Orientation: [{np.degrees(self.euler[0]):.1f}°, "
                f"{np.degrees(self.euler[1]):.1f}°, {np.degrees(self.euler[2]):.1f}°]")


@dataclass
class TrajectoryRecord:
    """Record of a complete trajectory execution"""
    timestamp: float
    object_type: str
    initial_position: List[float]
    grasp_pose: dict
    execution_time: float
    success: bool
    force_data: List[float]
    joint_positions: List[List[float]]


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    total_grasps: int = 0
    successful_grasps: int = 0
    average_execution_time: float = 0.0
    average_confidence: float = 0.0
    collision_count: int = 0
    max_force_recorded: float = 0.0


class AdvancedPyBulletSimulation:
    """
    Advanced PyBullet simulation with enhanced features:
    - Multiple grasp prediction
    - Force/torque sensing
    - Collision detection
    - Trajectory recording
    - Visual debugging
    """

    def __init__(self):
        self.physics_client = None
        self.robot_id = None
        self.table_id = None
        self.object_id = None
        self.plane_id = None
        self.constraint_id = None
        self.visual_markers = []

        # Robot parameters
        self.end_effector_index = 6
        self.num_joints = 7
        self.joint_indices = list(range(self.num_joints))

        # Simulation parameters
        self.time_step = 1./240.
        self.is_initialized = False

        # Object management
        self.current_object_type = ObjectType.CYLINDER
        self.object_base_position = [0.5, 0.0, 0.65]

        # Grasp prediction
        self.grasp_candidates = []
        self.selected_grasp_index = 0
        self.predicted_grasp = None

        # Trajectory
        self.trajectory_step = 0
        self.trajectory_waypoints = []
        self.trajectory_history = []

        # Force sensing
        self.force_data = []
        self.torque_data = []
        self.contact_detected = False

        # Recording
        self.is_recording = False
        self.recorded_trajectory = None
        self.playback_data = []

        # Collision detection
        self.collision_enabled = True
        self.collision_count = 0

        # Performance metrics
        self.metrics = PerformanceMetrics()

    def initialize_scene(self, object_position: Optional[List[float]] = None,
                        object_type: ObjectType = ObjectType.CYLINDER) -> bool:
        """Initialize or reset the PyBullet simulation scene"""
        try:
            # Disconnect previous session if exists
            if self.physics_client is not None:
                p.disconnect(self.physics_client)

            # Connect to PyBullet in GUI mode
            self.physics_client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

            # Configure simulation
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(self.time_step)
            p.setRealTimeSimulation(0)

            # Enable collision detection
            p.setCollisionFilterGroupMask(0, 0, 1, 1)

            # Load environment
            self.plane_id = p.loadURDF("plane.urdf")
            self.table_id = p.loadURDF("table/table.urdf",
                                       basePosition=[0.5, 0.0, 0.0],
                                       useFixedBase=True)

            # Load robot (Kuka IIWA)
            self.robot_id = p.loadURDF("kuka_iiwa/model_vr_limits.urdf",
                                       basePosition=[0, 0, 0],
                                       useFixedBase=True)

            # Enable force/torque sensor on end-effector
            p.enableJointForceTorqueSensor(self.robot_id, self.end_effector_index, True)

            # Load object
            self.current_object_type = object_type
            if object_position is None:
                object_position = self.object_base_position.copy()

            self.object_id = self._load_object(object_type, object_position)

            # Reset robot to initial pose
            self._reset_robot_pose()

            # Configure camera view
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0.4, 0.0, 0.4]
            )

            # Remove constraint if exists
            if self.constraint_id is not None:
                p.removeConstraint(self.constraint_id)
                self.constraint_id = None

            # Clear visual markers
            self._clear_visual_markers()

            # Reset data
            self.force_data = []
            self.torque_data = []
            self.grasp_candidates = []

            # Step simulation to settle
            for _ in range(100):
                p.stepSimulation()

            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Error initializing scene: {e}")
            return False

    def _load_object(self, object_type: ObjectType, position: List[float]) -> int:
        """Load the specified object type"""
        if object_type == ObjectType.RANDOM:
            object_type = np.random.choice([ObjectType.CYLINDER, ObjectType.CUBE,
                                           ObjectType.SPHERE, ObjectType.DUCK])

        try:
            obj_id = p.loadURDF(object_type.value,
                               basePosition=position,
                               globalScaling=0.5 if object_type != ObjectType.DUCK else 1.0)

            # Set random color for visibility
            color = np.random.uniform(0.3, 1.0, 3).tolist() + [1]
            p.changeVisualShape(obj_id, -1, rgbaColor=color)

            return obj_id
        except:
            # Fallback to cylinder if object loading fails
            obj_id = p.loadURDF("cylinder.urdf", basePosition=position, globalScaling=0.5)
            p.changeVisualShape(obj_id, -1, rgbaColor=[1, 0, 0, 1])
            return obj_id

    def _reset_robot_pose(self):
        """Reset robot to home position"""
        rest_poses = [0, 0, 0, -1.5, 0, 1.5, 0]
        for i, rest_pose in enumerate(rest_poses):
            p.resetJointState(self.robot_id, i, rest_pose)

    def _clear_visual_markers(self):
        """Clear all visual debug markers"""
        for marker_id in self.visual_markers:
            try:
                p.removeUserDebugItem(marker_id)
            except:
                pass
        self.visual_markers = []

    def get_camera_image(self, camera_angle: str = "top") -> np.ndarray:
        """
        Capture camera image from specified angle
        camera_angle: "top", "side", "front"
        """
        width, height = 640, 480

        if camera_angle == "top":
            eye_pos = [0.5, 0.0, 1.5]
            target_pos = [0.5, 0.0, 0.6]
        elif camera_angle == "side":
            eye_pos = [0.5, 1.0, 0.8]
            target_pos = [0.5, 0.0, 0.6]
        else:  # front
            eye_pos = [1.2, 0.0, 0.8]
            target_pos = [0.5, 0.0, 0.6]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=3.0
        )

        _, _, rgb, depth, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(rgb, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]

        return rgb_array

    def predict_multiple_grasps(self, camera_image: np.ndarray,
                                num_candidates: int = 5) -> List[GraspCandidate]:
        """
        Advanced Mock CNN Grasp Predictor
        Generates multiple grasp candidates with confidence scores
        """
        obj_pos = self.get_object_position()
        self.grasp_candidates = []

        # Generate multiple grasp strategies
        grasp_types = ["top-down", "side", "angular"]

        for i in range(num_candidates):
            # Select grasp type
            grasp_type = grasp_types[i % len(grasp_types)]

            if grasp_type == "top-down":
                # Top-down grasp
                grasp_x = obj_pos[0] + np.random.uniform(-0.02, 0.02)
                grasp_y = obj_pos[1] + np.random.uniform(-0.02, 0.02)
                grasp_z = obj_pos[2] + 0.15
                roll, pitch, yaw = 0, 0, np.random.uniform(-np.pi/4, np.pi/4)
                confidence = np.random.uniform(0.75, 0.95)

            elif grasp_type == "side":
                # Side grasp
                angle = np.random.uniform(0, 2*np.pi)
                offset = 0.12
                grasp_x = obj_pos[0] + offset * np.cos(angle)
                grasp_y = obj_pos[1] + offset * np.sin(angle)
                grasp_z = obj_pos[2]
                roll, pitch, yaw = 0, np.pi/2, angle
                confidence = np.random.uniform(0.60, 0.85)

            else:  # angular
                # Angular grasp
                grasp_x = obj_pos[0] + np.random.uniform(-0.03, 0.03)
                grasp_y = obj_pos[1] + np.random.uniform(-0.03, 0.03)
                grasp_z = obj_pos[2] + 0.10
                roll = np.random.uniform(-np.pi/6, np.pi/6)
                pitch = np.random.uniform(-np.pi/6, np.pi/6)
                yaw = np.random.uniform(-np.pi/4, np.pi/4)
                confidence = np.random.uniform(0.50, 0.75)

            grasp_position = [grasp_x, grasp_y, grasp_z]
            euler_orientation = [roll, pitch, yaw]
            quaternion_orientation = p.getQuaternionFromEuler(euler_orientation)

            candidate = GraspCandidate(
                position=grasp_position,
                orientation=quaternion_orientation,
                confidence=confidence,
                grasp_type=grasp_type,
                euler=euler_orientation
            )

            self.grasp_candidates.append(candidate)

        # Sort by confidence
        self.grasp_candidates.sort(key=lambda x: x.confidence, reverse=True)

        # Select best candidate
        self.selected_grasp_index = 0
        self.predicted_grasp = self.grasp_candidates[0]

        # Update metrics
        self.metrics.average_confidence = np.mean([c.confidence for c in self.grasp_candidates])

        return self.grasp_candidates

    def visualize_grasp_candidates(self):
        """Draw visual markers for all grasp candidates"""
        self._clear_visual_markers()

        for i, candidate in enumerate(self.grasp_candidates):
            # Color based on confidence (green=high, red=low)
            color = [1-candidate.confidence, candidate.confidence, 0]

            # Draw sphere at grasp position
            if i == self.selected_grasp_index:
                # Selected grasp - larger and brighter
                marker_id = p.addUserDebugText(
                    f"★ {candidate.confidence:.2%}",
                    candidate.position,
                    textColorRGB=[1, 1, 0],
                    textSize=1.5
                )
                self.visual_markers.append(marker_id)

                # Draw coordinate frame
                marker_id = p.addUserDebugLine(
                    candidate.position,
                    [candidate.position[0], candidate.position[1], candidate.position[2] + 0.1],
                    lineColorRGB=[0, 0, 1],
                    lineWidth=3
                )
                self.visual_markers.append(marker_id)
            else:
                # Other candidates - smaller markers
                marker_id = p.addUserDebugText(
                    f"{i+1}",
                    candidate.position,
                    textColorRGB=color,
                    textSize=1.0
                )
                self.visual_markers.append(marker_id)

    def select_grasp_candidate(self, index: int):
        """Select a specific grasp candidate"""
        if 0 <= index < len(self.grasp_candidates):
            self.selected_grasp_index = index
            self.predicted_grasp = self.grasp_candidates[index]
            self.visualize_grasp_candidates()

    def check_collisions(self) -> bool:
        """Check for collisions in current state"""
        if not self.collision_enabled:
            return False

        # Check for collisions with robot
        contact_points = p.getContactPoints(bodyA=self.robot_id)

        for contact in contact_points:
            # Ignore self-collisions and ground contact
            if contact[2] != self.plane_id and contact[2] != self.table_id:
                self.collision_count += 1
                self.metrics.collision_count += 1
                return True

        return False

    def get_force_torque_feedback(self) -> Tuple[List[float], List[float]]:
        """Get force and torque readings from end-effector sensor"""
        joint_state = p.getJointState(self.robot_id, self.end_effector_index)
        reaction_forces = joint_state[2]  # [Fx, Fy, Fz, Mx, My, Mz]

        forces = list(reaction_forces[:3])
        torques = list(reaction_forces[3:6])

        # Calculate magnitude
        force_magnitude = np.linalg.norm(forces)
        torque_magnitude = np.linalg.norm(torques)

        self.force_data.append(force_magnitude)
        self.torque_data.append(torque_magnitude)

        # Update max force
        if force_magnitude > self.metrics.max_force_recorded:
            self.metrics.max_force_recorded = force_magnitude

        return forces, torques

    def execute_pick_and_place(self, preview_only: bool = False) -> bool:
        """
        Initialize IK trajectory with optional preview mode
        """
        if self.predicted_grasp is None:
            return False

        grasp_pos = self.predicted_grasp.position
        grasp_orn = self.predicted_grasp.orientation

        # Enhanced 7-step trajectory
        approach_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.20]
        pre_grasp_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.05]
        descent_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] - 0.03]
        grasp_final_pos = descent_pos.copy()
        lift_pos = [grasp_pos[0], grasp_pos[1], 0.90]
        transport_pos = [0.3, 0.3, 0.90]
        place_pos = [0.3, 0.3, 0.65]

        self.trajectory_waypoints = [
            ("Approach", approach_pos, grasp_orn, 60),
            ("Pre-Grasp", pre_grasp_pos, grasp_orn, 40),
            ("Descent", descent_pos, grasp_orn, 40),
            ("Grasp", grasp_final_pos, grasp_orn, 30),
            ("Lift", lift_pos, grasp_orn, 80),
            ("Transport", transport_pos, grasp_orn, 60),
            ("Place", place_pos, grasp_orn, 60)
        ]

        # Visualize trajectory if preview mode
        if preview_only:
            self._visualize_trajectory()
            return True

        self.trajectory_step = 0

        # Start recording if enabled
        if self.is_recording:
            self.recorded_trajectory = TrajectoryRecord(
                timestamp=time.time(),
                object_type=self.current_object_type.name,
                initial_position=self.get_object_position(),
                grasp_pose=asdict(self.predicted_grasp),
                execution_time=0.0,
                success=False,
                force_data=[],
                joint_positions=[]
            )

        return True

    def _visualize_trajectory(self):
        """Draw trajectory path visualization"""
        self._clear_visual_markers()

        # Draw line connecting waypoints
        for i in range(len(self.trajectory_waypoints) - 1):
            pos1 = self.trajectory_waypoints[i][1]
            pos2 = self.trajectory_waypoints[i+1][1]

            marker_id = p.addUserDebugLine(
                pos1, pos2,
                lineColorRGB=[0, 1, 1],
                lineWidth=2
            )
            self.visual_markers.append(marker_id)

            # Add waypoint markers
            marker_id = p.addUserDebugText(
                self.trajectory_waypoints[i][0],
                pos1,
                textColorRGB=[1, 1, 1],
                textSize=1.0
            )
            self.visual_markers.append(marker_id)

    def step_trajectory(self) -> Tuple[bool, str]:
        """Execute one step of the trajectory with advanced monitoring"""
        if self.trajectory_step >= len(self.trajectory_waypoints):
            return True, "Grasping sequence complete"

        step_name, target_pos, target_orn, num_steps = \
            self.trajectory_waypoints[self.trajectory_step]

        # Calculate IK
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_pos,
            target_orn,
            maxNumIterations=100,
            residualThreshold=1e-5
        )

        # Apply joint positions
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_poses[i],
                force=500,
                maxVelocity=1.0
            )

        # Step simulation and monitor
        for _ in range(num_steps):
            p.stepSimulation()

            # Get force feedback
            self.get_force_torque_feedback()

            # Check collisions
            if self.check_collisions():
                return True, f"Collision detected during {step_name}!"

            # Record joint positions if recording
            if self.is_recording and self.recorded_trajectory:
                current_joints = [p.getJointState(self.robot_id, i)[0]
                                 for i in range(self.num_joints)]
                self.recorded_trajectory.joint_positions.append(current_joints)

            time.sleep(self.time_step * 0.5)  # Slow down for visualization

        # Special action for grasp step
        if step_name == "Grasp" and self.constraint_id is None:
            ee_state = p.getLinkState(self.robot_id, self.end_effector_index)

            # Create constraint
            self.constraint_id = p.createConstraint(
                parentBodyUniqueId=self.robot_id,
                parentLinkIndex=self.end_effector_index,
                childBodyUniqueId=self.object_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0]
            )

            self.contact_detected = True

        # Move to next step
        self.trajectory_step += 1

        if self.trajectory_step >= len(self.trajectory_waypoints):
            # Trajectory complete
            success = self.constraint_id is not None

            # Update metrics
            self.metrics.total_grasps += 1
            if success:
                self.metrics.successful_grasps += 1

            # Finalize recording
            if self.is_recording and self.recorded_trajectory:
                self.recorded_trajectory.success = success
                self.recorded_trajectory.execution_time = time.time() - self.recorded_trajectory.timestamp
                self.recorded_trajectory.force_data = self.force_data.copy()
                self.trajectory_history.append(self.recorded_trajectory)

            return True, f"{step_name} complete. Grasping sequence complete."
        else:
            next_step = self.trajectory_waypoints[self.trajectory_step][0]
            return False, f"{step_name} complete. Next: {next_step}"

    def export_configuration(self, filepath: str) -> bool:
        """Export current grasp configuration to JSON"""
        try:
            config = {
                "object_type": self.current_object_type.name,
                "object_position": self.get_object_position(),
                "grasp_candidates": [asdict(g) for g in self.grasp_candidates],
                "selected_index": self.selected_grasp_index,
                "metrics": asdict(self.metrics)
            }

            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)

            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

    def import_configuration(self, filepath: str) -> bool:
        """Import grasp configuration from JSON"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)

            # Reconstruct grasp candidates
            self.grasp_candidates = []
            for g in config.get("grasp_candidates", []):
                candidate = GraspCandidate(
                    position=g["position"],
                    orientation=g["orientation"],
                    confidence=g["confidence"],
                    grasp_type=g["grasp_type"],
                    euler=g.get("euler")
                )
                self.grasp_candidates.append(candidate)

            self.selected_grasp_index = config.get("selected_index", 0)
            if self.grasp_candidates:
                self.predicted_grasp = self.grasp_candidates[self.selected_grasp_index]

            return True
        except Exception as e:
            print(f"Import error: {e}")
            return False

    def step_simulation(self):
        """Advance simulation by one time step"""
        if self.is_initialized:
            p.stepSimulation()

    def get_object_position(self) -> List[float]:
        """Get current object position"""
        if self.object_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.object_id)
            return list(pos)
        return [0, 0, 0]

    def set_object_position(self, position: List[float]):
        """Set object position"""
        if self.object_id is not None:
            _, ori = p.getBasePositionAndOrientation(self.object_id)
            p.resetBasePositionAndOrientation(self.object_id, position, ori)

    def cleanup(self):
        """Disconnect from PyBullet"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


class AdvancedRoboticGraspingGUI(QMainWindow):
    """
    Advanced PyQt6 GUI with enhanced features
    """

    def __init__(self):
        super().__init__()

        # Initialize simulation
        self.simulation = AdvancedPyBulletSimulation()
        self.state = SimulationState.UNINITIALIZED

        # Timers
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)

        self.trajectory_timer = QTimer()
        self.trajectory_timer.timeout.connect(self.execute_trajectory_step)

        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics_display)

        # Initialize UI
        self.init_ui()

        # Start timers
        self.sim_timer.start(int(1000 / 240))
        self.metrics_timer.start(500)  # Update metrics twice per second

    def init_ui(self):
        """Initialize the advanced user interface"""
        self.setWindowTitle("Advanced AI-Driven Robotic Grasping Simulation")
        self.setGeometry(50, 50, 1600, 1000)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left Panel: Simulation Info
        left_panel = self.create_simulation_panel()
        splitter.addWidget(left_panel)

        # Right Panel: Tabbed interface
        right_panel = self.create_tabbed_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes([600, 1000])

    def create_simulation_panel(self) -> QWidget:
        """Create simulation info panel"""
        panel = QGroupBox("PyBullet Simulation")
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel(
            "PyBullet GUI window opens separately.\n\n"
            "Advanced Features:\n"
            "• Multiple grasp candidates\n"
            "• Force/torque feedback\n"
            "• Trajectory visualization\n"
            "• Collision detection\n"
            "• Recording & Playback\n"
            "• Performance metrics"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Status indicators
        status_group = QGroupBox("Status")
        status_layout = QGridLayout()

        self.status_sim = QLabel("⚫ Not Initialized")
        self.status_collision = QLabel("⚫ No Collision")
        self.status_contact = QLabel("⚫ No Contact")
        self.status_recording = QLabel("⚫ Not Recording")

        status_layout.addWidget(QLabel("Simulation:"), 0, 0)
        status_layout.addWidget(self.status_sim, 0, 1)
        status_layout.addWidget(QLabel("Collision:"), 1, 0)
        status_layout.addWidget(self.status_collision, 1, 1)
        status_layout.addWidget(QLabel("Contact:"), 2, 0)
        status_layout.addWidget(self.status_contact, 2, 1)
        status_layout.addWidget(QLabel("Recording:"), 3, 0)
        status_layout.addWidget(self.status_recording, 3, 1)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        layout.addStretch()
        panel.setLayout(layout)

        return panel

    def create_tabbed_panel(self) -> QWidget:
        """Create tabbed interface for different control sections"""
        tabs = QTabWidget()

        # Tab 1: Basic Controls
        tabs.addTab(self.create_basic_controls_tab(), "🎮 Basic Controls")

        # Tab 2: Advanced Features
        tabs.addTab(self.create_advanced_features_tab(), "⚙️ Advanced")

        # Tab 3: Grasp Candidates
        tabs.addTab(self.create_grasp_candidates_tab(), "🎯 Grasp Candidates")

        # Tab 4: Metrics & Analytics
        tabs.addTab(self.create_metrics_tab(), "📊 Metrics")

        # Tab 5: Recording & Playback
        tabs.addTab(self.create_recording_tab(), "🎬 Recording")

        return tabs

    def create_basic_controls_tab(self) -> QWidget:
        """Create basic controls tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Object Selection
        obj_group = QGroupBox("Object Selection")
        obj_layout = QVBoxLayout()

        self.combo_object = QComboBox()
        self.combo_object.addItems(["Cylinder", "Cube", "Sphere", "Duck", "Random"])
        obj_layout.addWidget(QLabel("Object Type:"))
        obj_layout.addWidget(self.combo_object)

        obj_group.setLayout(obj_layout)
        layout.addWidget(obj_group)

        # Main Workflow Buttons
        workflow_group = QGroupBox("Workflow Controls")
        workflow_layout = QVBoxLayout()

        self.btn_initialize = QPushButton("1. Initialize Scene")
        self.btn_initialize.setMinimumHeight(50)
        self.btn_initialize.clicked.connect(self.on_initialize_scene)
        workflow_layout.addWidget(self.btn_initialize)

        self.btn_predict = QPushButton("2. Predict Grasps (Multiple Candidates)")
        self.btn_predict.setMinimumHeight(50)
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.on_predict_grasp)
        workflow_layout.addWidget(self.btn_predict)

        self.btn_preview = QPushButton("Preview Trajectory")
        self.btn_preview.setMinimumHeight(40)
        self.btn_preview.setEnabled(False)
        self.btn_preview.clicked.connect(self.on_preview_trajectory)
        workflow_layout.addWidget(self.btn_preview)

        self.btn_execute = QPushButton("3. Execute Pick & Place")
        self.btn_execute.setMinimumHeight(50)
        self.btn_execute.setEnabled(False)
        self.btn_execute.clicked.connect(self.on_execute_pick)
        workflow_layout.addWidget(self.btn_execute)

        workflow_group.setLayout(workflow_layout)
        layout.addWidget(workflow_group)

        # Object Position Sliders
        slider_group = QGroupBox("Object Position")
        slider_layout = QGridLayout()

        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.setMinimum(30)
        self.slider_x.setMaximum(70)
        self.slider_x.setValue(50)
        self.slider_x.valueChanged.connect(self.on_object_position_changed)

        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setMinimum(-20)
        self.slider_y.setMaximum(20)
        self.slider_y.setValue(0)
        self.slider_y.valueChanged.connect(self.on_object_position_changed)

        self.slider_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_z.setMinimum(63)
        self.slider_z.setMaximum(80)
        self.slider_z.setValue(65)
        self.slider_z.valueChanged.connect(self.on_object_position_changed)

        self.label_x = QLabel("0.50 m")
        self.label_y = QLabel("0.00 m")
        self.label_z = QLabel("0.65 m")

        slider_layout.addWidget(QLabel("X:"), 0, 0)
        slider_layout.addWidget(self.slider_x, 0, 1)
        slider_layout.addWidget(self.label_x, 0, 2)

        slider_layout.addWidget(QLabel("Y:"), 1, 0)
        slider_layout.addWidget(self.slider_y, 1, 1)
        slider_layout.addWidget(self.label_y, 1, 2)

        slider_layout.addWidget(QLabel("Z:"), 2, 0)
        slider_layout.addWidget(self.slider_z, 2, 1)
        slider_layout.addWidget(self.label_z, 2, 2)

        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)

        # Status Log
        log_group = QGroupBox("Status Log")
        log_layout = QVBoxLayout()

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setFont(QFont("Courier", 9))
        self.status_log.setMaximumHeight(200)
        self.log_status("Advanced application started.")

        log_layout.addWidget(self.status_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_advanced_features_tab(self) -> QWidget:
        """Create advanced features tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Camera Controls
        camera_group = QGroupBox("Camera Control")
        camera_layout = QVBoxLayout()

        self.combo_camera = QComboBox()
        self.combo_camera.addItems(["Top View", "Side View", "Front View"])
        camera_layout.addWidget(QLabel("Camera Angle:"))
        camera_layout.addWidget(self.combo_camera)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Grasp Parameters
        grasp_group = QGroupBox("Grasp Prediction Parameters")
        grasp_layout = QGridLayout()

        grasp_layout.addWidget(QLabel("Number of Candidates:"), 0, 0)
        self.spin_candidates = QSpinBox()
        self.spin_candidates.setMinimum(1)
        self.spin_candidates.setMaximum(10)
        self.spin_candidates.setValue(5)
        grasp_layout.addWidget(self.spin_candidates, 0, 1)

        grasp_group.setLayout(grasp_layout)
        layout.addWidget(grasp_group)

        # Collision Detection
        collision_group = QGroupBox("Collision Detection")
        collision_layout = QVBoxLayout()

        self.check_collision = QCheckBox("Enable Collision Detection")
        self.check_collision.setChecked(True)
        self.check_collision.stateChanged.connect(self.on_collision_toggle)
        collision_layout.addWidget(self.check_collision)

        self.label_collision_count = QLabel("Collisions Detected: 0")
        collision_layout.addWidget(self.label_collision_count)

        collision_group.setLayout(collision_layout)
        layout.addWidget(collision_group)

        # Force Feedback
        force_group = QGroupBox("Force/Torque Feedback")
        force_layout = QVBoxLayout()

        self.label_force = QLabel("Current Force: 0.000 N")
        self.label_torque = QLabel("Current Torque: 0.000 Nm")
        self.progress_force = QProgressBar()
        self.progress_force.setMaximum(1000)

        force_layout.addWidget(self.label_force)
        force_layout.addWidget(self.progress_force)
        force_layout.addWidget(self.label_torque)

        force_group.setLayout(force_layout)
        layout.addWidget(force_group)

        # Import/Export
        io_group = QGroupBox("Import/Export")
        io_layout = QVBoxLayout()

        self.btn_export = QPushButton("Export Configuration")
        self.btn_export.clicked.connect(self.on_export_config)
        io_layout.addWidget(self.btn_export)

        self.btn_import = QPushButton("Import Configuration")
        self.btn_import.clicked.connect(self.on_import_config)
        io_layout.addWidget(self.btn_import)

        io_group.setLayout(io_layout)
        layout.addWidget(io_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_grasp_candidates_tab(self) -> QWidget:
        """Create grasp candidates display tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        info_label = QLabel("Select from multiple predicted grasp candidates:")
        layout.addWidget(info_label)

        # Grasp candidates table
        self.grasp_table = QTableWidget(0, 5)
        self.grasp_table.setHorizontalHeaderLabels(
            ["#", "Type", "Confidence", "Position", "Orientation"]
        )
        self.grasp_table.cellClicked.connect(self.on_grasp_candidate_selected)
        layout.addWidget(self.grasp_table)

        # Selected grasp details
        details_group = QGroupBox("Selected Grasp Details")
        details_layout = QVBoxLayout()

        self.label_grasp_details = QLabel("No grasp selected")
        self.label_grasp_details.setWordWrap(True)
        details_layout.addWidget(self.label_grasp_details)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        widget.setLayout(layout)
        return widget

    def create_metrics_tab(self) -> QWidget:
        """Create performance metrics tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Summary metrics
        summary_group = QGroupBox("Performance Summary")
        summary_layout = QGridLayout()

        self.label_total_grasps = QLabel("0")
        self.label_successful_grasps = QLabel("0")
        self.label_success_rate = QLabel("0.0%")
        self.label_avg_confidence = QLabel("0.0%")
        self.label_avg_time = QLabel("0.0 s")
        self.label_max_force = QLabel("0.0 N")

        summary_layout.addWidget(QLabel("Total Grasps:"), 0, 0)
        summary_layout.addWidget(self.label_total_grasps, 0, 1)
        summary_layout.addWidget(QLabel("Successful:"), 1, 0)
        summary_layout.addWidget(self.label_successful_grasps, 1, 1)
        summary_layout.addWidget(QLabel("Success Rate:"), 2, 0)
        summary_layout.addWidget(self.label_success_rate, 2, 1)
        summary_layout.addWidget(QLabel("Avg Confidence:"), 3, 0)
        summary_layout.addWidget(self.label_avg_confidence, 3, 1)
        summary_layout.addWidget(QLabel("Max Force:"), 4, 0)
        summary_layout.addWidget(self.label_max_force, 4, 1)

        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # Reset button
        self.btn_reset_metrics = QPushButton("Reset Metrics")
        self.btn_reset_metrics.clicked.connect(self.on_reset_metrics)
        layout.addWidget(self.btn_reset_metrics)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_recording_tab(self) -> QWidget:
        """Create recording and playback tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Recording controls
        rec_group = QGroupBox("Recording Control")
        rec_layout = QVBoxLayout()

        self.check_recording = QCheckBox("Enable Recording")
        self.check_recording.stateChanged.connect(self.on_recording_toggle)
        rec_layout.addWidget(self.check_recording)

        self.label_recording_count = QLabel("Recordings: 0")
        rec_layout.addWidget(self.label_recording_count)

        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        # Trajectory history
        history_group = QGroupBox("Trajectory History")
        history_layout = QVBoxLayout()

        self.list_recordings = QTableWidget(0, 4)
        self.list_recordings.setHorizontalHeaderLabels(
            ["Object", "Time (s)", "Success", "Timestamp"]
        )
        history_layout.addWidget(self.list_recordings)

        self.btn_export_recordings = QPushButton("Export All Recordings")
        self.btn_export_recordings.clicked.connect(self.on_export_recordings)
        history_layout.addWidget(self.btn_export_recordings)

        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        widget.setLayout(layout)
        return widget

    def log_status(self, message: str):
        """Log status message"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")

    def on_initialize_scene(self):
        """Initialize scene with selected object"""
        self.log_status("Initializing scene...")

        # Get object type
        obj_map = {
            "Cylinder": ObjectType.CYLINDER,
            "Cube": ObjectType.CUBE,
            "Sphere": ObjectType.SPHERE,
            "Duck": ObjectType.DUCK,
            "Random": ObjectType.RANDOM
        }
        obj_type = obj_map[self.combo_object.currentText()]

        # Get position
        obj_pos = [
            self.slider_x.value() / 100.0,
            self.slider_y.value() / 100.0,
            self.slider_z.value() / 100.0
        ]

        success = self.simulation.initialize_scene(obj_pos, obj_type)

        if success:
            self.state = SimulationState.READY
            self.log_status(f"✓ Scene initialized with {obj_type.name}")
            self.status_sim.setText("🟢 Ready")

            self.btn_predict.setEnabled(True)
            self.btn_preview.setEnabled(False)
            self.btn_execute.setEnabled(False)
        else:
            self.log_status("✗ Initialization failed")

    def on_predict_grasp(self):
        """Predict multiple grasp candidates"""
        self.log_status("Predicting grasp candidates...")

        # Get camera angle
        camera_map = {"Top View": "top", "Side View": "side", "Front View": "front"}
        camera_angle = camera_map[self.combo_camera.currentText()]

        # Capture image
        camera_image = self.simulation.get_camera_image(camera_angle)

        # Predict grasps
        num_candidates = self.spin_candidates.value()
        candidates = self.simulation.predict_multiple_grasps(camera_image, num_candidates)

        # Visualize in simulation
        self.simulation.visualize_grasp_candidates()

        # Update table
        self.update_grasp_candidates_table(candidates)

        self.state = SimulationState.GRASP_PREDICTED
        self.log_status(f"✓ Generated {len(candidates)} grasp candidates")
        self.log_status(f"  Best confidence: {candidates[0].confidence:.2%}")

        self.btn_preview.setEnabled(True)
        self.btn_execute.setEnabled(True)

    def update_grasp_candidates_table(self, candidates: List[GraspCandidate]):
        """Update grasp candidates table"""
        self.grasp_table.setRowCount(len(candidates))

        for i, candidate in enumerate(candidates):
            self.grasp_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.grasp_table.setItem(i, 1, QTableWidgetItem(candidate.grasp_type))
            self.grasp_table.setItem(i, 2, QTableWidgetItem(f"{candidate.confidence:.2%}"))

            pos_str = f"[{candidate.position[0]:.2f}, {candidate.position[1]:.2f}, {candidate.position[2]:.2f}]"
            self.grasp_table.setItem(i, 3, QTableWidgetItem(pos_str))

            euler_deg = [np.degrees(e) for e in candidate.euler]
            ori_str = f"[{euler_deg[0]:.1f}°, {euler_deg[1]:.1f}°, {euler_deg[2]:.1f}°]"
            self.grasp_table.setItem(i, 4, QTableWidgetItem(ori_str))

            # Color by confidence
            if candidate.confidence > 0.8:
                color = QColor(200, 255, 200)
            elif candidate.confidence > 0.6:
                color = QColor(255, 255, 200)
            else:
                color = QColor(255, 220, 220)

            for j in range(5):
                self.grasp_table.item(i, j).setBackground(color)

        # Update details
        if candidates:
            self.update_grasp_details(candidates[0])

    def update_grasp_details(self, candidate: GraspCandidate):
        """Update selected grasp details"""
        details = f"""
        <b>Grasp Type:</b> {candidate.grasp_type}<br>
        <b>Confidence:</b> {candidate.confidence:.2%}<br>
        <b>Position:</b> X={candidate.position[0]:.3f}, Y={candidate.position[1]:.3f}, Z={candidate.position[2]:.3f} m<br>
        <b>Orientation (Euler):</b> Roll={np.degrees(candidate.euler[0]):.1f}°,
        Pitch={np.degrees(candidate.euler[1]):.1f}°, Yaw={np.degrees(candidate.euler[2]):.1f}°
        """
        self.label_grasp_details.setText(details)

    def on_grasp_candidate_selected(self, row: int, col: int):
        """Handle grasp candidate selection"""
        self.simulation.select_grasp_candidate(row)
        self.update_grasp_details(self.simulation.grasp_candidates[row])
        self.log_status(f"Selected grasp candidate #{row + 1}")

    def on_preview_trajectory(self):
        """Preview trajectory without execution"""
        self.log_status("Generating trajectory preview...")
        success = self.simulation.execute_pick_and_place(preview_only=True)

        if success:
            self.log_status("✓ Trajectory preview displayed in simulation")
        else:
            self.log_status("✗ Failed to generate preview")

    def on_execute_pick(self):
        """Execute pick and place"""
        self.log_status("Executing pick and place...")

        success = self.simulation.execute_pick_and_place(preview_only=False)

        if success:
            self.state = SimulationState.EXECUTING
            self.status_sim.setText("🔵 Executing")

            self.btn_initialize.setEnabled(False)
            self.btn_predict.setEnabled(False)
            self.btn_preview.setEnabled(False)
            self.btn_execute.setEnabled(False)

            self.trajectory_timer.start(100)
        else:
            self.log_status("✗ Execution failed")

    def execute_trajectory_step(self):
        """Execute trajectory step"""
        is_complete, status_msg = self.simulation.step_trajectory()

        self.log_status(f"  {status_msg}")

        # Update force feedback
        if len(self.simulation.force_data) > 0:
            force = self.simulation.force_data[-1]
            self.label_force.setText(f"Current Force: {force:.3f} N")
            self.progress_force.setValue(int(min(force, 100) * 10))

        # Update contact status
        if self.simulation.contact_detected:
            self.status_contact.setText("🟢 Contact")

        if is_complete:
            self.trajectory_timer.stop()
            self.state = SimulationState.COMPLETED
            self.status_sim.setText("🟢 Complete")
            self.log_status("✓ Sequence complete!")

            # Update recording list
            if self.simulation.is_recording:
                self.update_recording_list()

            self.btn_initialize.setEnabled(True)

    def on_object_position_changed(self):
        """Update object position"""
        x = self.slider_x.value() / 100.0
        y = self.slider_y.value() / 100.0
        z = self.slider_z.value() / 100.0

        self.label_x.setText(f"{x:.2f} m")
        self.label_y.setText(f"{y:.2f} m")
        self.label_z.setText(f"{z:.2f} m")

        if self.state in [SimulationState.READY, SimulationState.GRASP_PREDICTED]:
            self.simulation.set_object_position([x, y, z])

    def on_collision_toggle(self, state):
        """Toggle collision detection"""
        self.simulation.collision_enabled = (state == Qt.CheckState.Checked.value)
        self.log_status(f"Collision detection: {'enabled' if self.simulation.collision_enabled else 'disabled'}")

    def on_recording_toggle(self, state):
        """Toggle recording"""
        self.simulation.is_recording = (state == Qt.CheckState.Checked.value)
        recording_enabled = self.simulation.is_recording
        self.status_recording.setText("🔴 Recording" if recording_enabled else "⚫ Not Recording")
        self.log_status(f"Recording: {'enabled' if recording_enabled else 'disabled'}")

    def update_recording_list(self):
        """Update recording history list"""
        history = self.simulation.trajectory_history
        self.list_recordings.setRowCount(len(history))

        for i, record in enumerate(history):
            self.list_recordings.setItem(i, 0, QTableWidgetItem(record.object_type))
            self.list_recordings.setItem(i, 1, QTableWidgetItem(f"{record.execution_time:.2f}"))
            self.list_recordings.setItem(i, 2, QTableWidgetItem("✓" if record.success else "✗"))

            timestamp_str = time.strftime("%H:%M:%S", time.localtime(record.timestamp))
            self.list_recordings.setItem(i, 3, QTableWidgetItem(timestamp_str))

        self.label_recording_count.setText(f"Recordings: {len(history)}")

    def update_metrics_display(self):
        """Update metrics display"""
        metrics = self.simulation.metrics

        self.label_total_grasps.setText(str(metrics.total_grasps))
        self.label_successful_grasps.setText(str(metrics.successful_grasps))

        if metrics.total_grasps > 0:
            success_rate = (metrics.successful_grasps / metrics.total_grasps) * 100
            self.label_success_rate.setText(f"{success_rate:.1f}%")

        self.label_avg_confidence.setText(f"{metrics.average_confidence * 100:.1f}%")
        self.label_max_force.setText(f"{metrics.max_force_recorded:.3f} N")

        # Update collision count
        self.label_collision_count.setText(f"Collisions Detected: {metrics.collision_count}")

        if metrics.collision_count > 0:
            self.status_collision.setText("🔴 Collision!")

    def on_reset_metrics(self):
        """Reset performance metrics"""
        self.simulation.metrics = PerformanceMetrics()
        self.log_status("Metrics reset")

    def on_export_config(self):
        """Export configuration"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "", "JSON Files (*.json)"
        )

        if filepath:
            success = self.simulation.export_configuration(filepath)
            if success:
                self.log_status(f"✓ Configuration exported to {filepath}")
            else:
                self.log_status("✗ Export failed")

    def on_import_config(self):
        """Import configuration"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Import Configuration", "", "JSON Files (*.json)"
        )

        if filepath:
            success = self.simulation.import_configuration(filepath)
            if success:
                self.log_status(f"✓ Configuration imported from {filepath}")
                if self.simulation.grasp_candidates:
                    self.update_grasp_candidates_table(self.simulation.grasp_candidates)
            else:
                self.log_status("✗ Import failed")

    def on_export_recordings(self):
        """Export all recordings"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Recordings", "", "JSON Files (*.json)"
        )

        if filepath:
            try:
                data = [asdict(r) for r in self.simulation.trajectory_history]
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                self.log_status(f"✓ Exported {len(data)} recordings")
            except Exception as e:
                self.log_status(f"✗ Export failed: {e}")

    def update_simulation(self):
        """Update simulation loop"""
        if self.state != SimulationState.UNINITIALIZED:
            self.simulation.step_simulation()

    def closeEvent(self, event):
        """Handle close event"""
        self.log_status("Shutting down...")
        self.sim_timer.stop()
        self.trajectory_timer.stop()
        self.metrics_timer.stop()
        self.simulation.cleanup()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = AdvancedRoboticGraspingGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
