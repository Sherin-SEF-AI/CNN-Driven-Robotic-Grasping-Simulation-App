#!/usr/bin/env python3
"""
AI-Driven Robotic Grasping Simulation Application
PyQt6 GUI + PyBullet Physics Simulation
Single-file complete application for 6-DOF pick-and-place task

Author: Sherin Joseph Roy
Email: sherin.joseph2217@gmail.com
Version: 1.0 (Basic)
"""

import sys
import time
import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Optional

# PyQt6 imports
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QSlider, QSplitter,
    QGridLayout, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont

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


class GraspPose:
    """Data class for 6-DOF grasp pose"""
    def __init__(self, position: List[float], orientation: List[float],
                 euler: Optional[List[float]] = None):
        self.position = position  # [x, y, z]
        self.orientation = orientation  # Quaternion [x, y, z, w]
        self.euler = euler if euler else p.getEulerFromQuaternion(orientation)  # [roll, pitch, yaw]

    def __str__(self):
        return (f"Position: [{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}] m\n"
                f"Orientation (Euler): [{np.degrees(self.euler[0]):.1f}°, "
                f"{np.degrees(self.euler[1]):.1f}°, {np.degrees(self.euler[2]):.1f}°]")


class PyBulletSimulation:
    """
    Manages the PyBullet physics simulation for robotic grasping.
    Handles robot control, IK calculations, and object manipulation.
    """

    def __init__(self):
        self.physics_client = None
        self.robot_id = None
        self.table_id = None
        self.object_id = None
        self.plane_id = None
        self.constraint_id = None

        # Robot parameters
        self.end_effector_index = 6  # Kuka IIWA end-effector link
        self.num_joints = 7
        self.joint_indices = list(range(self.num_joints))

        # Simulation parameters
        self.time_step = 1./240.
        self.is_initialized = False

        # Object default position
        self.object_base_position = [0.5, 0.0, 0.65]

        # Grasp parameters
        self.predicted_grasp = None
        self.trajectory_step = 0
        self.trajectory_waypoints = []

    def initialize_scene(self, object_position: Optional[List[float]] = None) -> bool:
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

            # Load environment
            self.plane_id = p.loadURDF("plane.urdf")
            self.table_id = p.loadURDF("table/table.urdf",
                                       basePosition=[0.5, 0.0, 0.0],
                                       useFixedBase=True)

            # Load robot (Kuka IIWA)
            self.robot_id = p.loadURDF("kuka_iiwa/model_vr_limits.urdf",
                                       basePosition=[0, 0, 0],
                                       useFixedBase=True)

            # Load object (cylinder)
            if object_position is None:
                object_position = self.object_base_position.copy()

            self.object_id = p.loadURDF("cylinder.urdf",
                                       basePosition=object_position,
                                       globalScaling=0.5)

            # Set object color (red for visibility)
            p.changeVisualShape(self.object_id, -1, rgbaColor=[1, 0, 0, 1])

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

            # Step simulation to settle
            for _ in range(100):
                p.stepSimulation()

            self.is_initialized = True
            return True

        except Exception as e:
            print(f"Error initializing scene: {e}")
            return False

    def _reset_robot_pose(self):
        """Reset robot to home position"""
        rest_poses = [0, 0, 0, -1.5, 0, 1.5, 0]
        for i, rest_pose in enumerate(rest_poses):
            p.resetJointState(self.robot_id, i, rest_pose)

    def get_camera_image(self) -> np.ndarray:
        """
        Capture a simulated camera image from above the scene.
        Returns RGB image as numpy array.
        """
        # Camera parameters
        width, height = 640, 480
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[0.5, 0.0, 1.5],
            cameraTargetPosition=[0.5, 0.0, 0.6],
            cameraUpVector=[0, 1, 0]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height, nearVal=0.1, farVal=3.0
        )

        # Capture image
        _, _, rgb, _, _ = p.getCameraImage(
            width, height, view_matrix, proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        rgb_array = np.array(rgb, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))[:, :, :3]

        return rgb_array

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
        """Set object position (for debugging/testing)"""
        if self.object_id is not None:
            _, ori = p.getBasePositionAndOrientation(self.object_id)
            p.resetBasePositionAndOrientation(self.object_id, position, ori)

    def predict_grasp_pose(self, camera_image: np.ndarray) -> GraspPose:
        """
        Mock CNN Grasp Predictor
        Simulates AI-based grasp pose prediction without actual CNN training.
        Returns a plausible 6-DOF grasp pose for top-down grasping.
        """
        # Get current object position
        obj_pos = self.get_object_position()

        # Mock prediction: top-down grasp with slight randomization
        # Position: slightly above object center
        grasp_x = obj_pos[0] + np.random.uniform(-0.02, 0.02)
        grasp_y = obj_pos[1] + np.random.uniform(-0.02, 0.02)
        grasp_z = obj_pos[2] + 0.15  # 15cm above object

        grasp_position = [grasp_x, grasp_y, grasp_z]

        # Orientation: pointing downward (end-effector facing down)
        # Euler angles: roll=0, pitch=0, yaw=random
        roll = 0
        pitch = 0
        yaw = np.random.uniform(-np.pi/4, np.pi/4)

        euler_orientation = [roll, pitch, yaw]
        quaternion_orientation = p.getQuaternionFromEuler(euler_orientation)

        grasp_pose = GraspPose(
            position=grasp_position,
            orientation=quaternion_orientation,
            euler=euler_orientation
        )

        self.predicted_grasp = grasp_pose
        return grasp_pose

    def execute_pick_and_place(self) -> bool:
        """
        Initialize the IK trajectory for pick-and-place operation.
        Returns True if trajectory is successfully initialized.
        """
        if self.predicted_grasp is None:
            return False

        # Build 5-step trajectory waypoints
        grasp_pos = self.predicted_grasp.position
        grasp_orn = self.predicted_grasp.orientation

        # Step 1: Approach - 15cm above grasp point
        approach_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.15]

        # Step 2: Descent - at grasp point
        descent_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] - 0.05]

        # Step 3: Grasp - same as descent (constraint applied)
        grasp_final_pos = descent_pos.copy()

        # Step 4: Lift - 20cm above table
        lift_pos = [grasp_pos[0], grasp_pos[1], 0.85]

        # Step 5: Stop - maintain lift position
        stop_pos = lift_pos.copy()

        self.trajectory_waypoints = [
            ("Approach", approach_pos, grasp_orn, 60),
            ("Descent", descent_pos, grasp_orn, 60),
            ("Grasp", grasp_final_pos, grasp_orn, 30),
            ("Lift", lift_pos, grasp_orn, 80),
            ("Stop", stop_pos, grasp_orn, 40)
        ]

        self.trajectory_step = 0
        return True

    def step_trajectory(self) -> Tuple[bool, str]:
        """
        Execute one step of the trajectory.
        Returns (is_complete, status_message)
        """
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

        # Step simulation
        for _ in range(num_steps):
            p.stepSimulation()
            time.sleep(self.time_step)

        # Special action for grasp step: create constraint
        if step_name == "Grasp" and self.constraint_id is None:
            ee_state = p.getLinkState(self.robot_id, self.end_effector_index)
            ee_pos = ee_state[0]
            ee_orn = ee_state[1]

            # Create fixed constraint between end-effector and object
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

        # Move to next step
        self.trajectory_step += 1

        if self.trajectory_step >= len(self.trajectory_waypoints):
            return True, f"{step_name} complete. Grasping sequence complete."
        else:
            next_step = self.trajectory_waypoints[self.trajectory_step][0]
            return False, f"{step_name} complete. Next: {next_step}"

    def cleanup(self):
        """Disconnect from PyBullet"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


class RoboticGraspingGUI(QMainWindow):
    """
    Main PyQt6 GUI application for robotic grasping simulation.
    Provides controls and visualization for the pick-and-place workflow.
    """

    def __init__(self):
        super().__init__()

        # Initialize simulation
        self.simulation = PyBulletSimulation()
        self.state = SimulationState.UNINITIALIZED

        # Timers
        self.sim_timer = QTimer()
        self.sim_timer.timeout.connect(self.update_simulation)

        self.trajectory_timer = QTimer()
        self.trajectory_timer.timeout.connect(self.execute_trajectory_step)

        # Initialize UI
        self.init_ui()

        # Start simulation update loop (240Hz)
        self.sim_timer.start(int(1000 / 240))

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("AI-Driven Robotic Grasping Simulation")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left Panel: Simulation View
        left_panel = self.create_simulation_panel()
        splitter.addWidget(left_panel)

        # Right Panel: Controls and Output
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Top-Right: Controls
        controls = self.create_controls_panel()
        right_layout.addWidget(controls)

        # Bottom-Right: Output
        output = self.create_output_panel()
        right_layout.addWidget(output)

        splitter.addWidget(right_panel)

        # Set splitter proportions (70% simulation, 30% controls)
        splitter.setSizes([840, 360])

    def create_simulation_panel(self) -> QWidget:
        """Create the simulation view panel"""
        panel = QGroupBox("PyBullet Simulation View")
        layout = QVBoxLayout()

        info_label = QLabel(
            "PyBullet GUI window will open separately.\n"
            "Use the controls on the right to interact with the simulation."
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("padding: 20px; font-size: 12pt;")
        info_label.setWordWrap(True)

        layout.addWidget(info_label)
        panel.setLayout(layout)

        return panel

    def create_controls_panel(self) -> QWidget:
        """Create the controls panel with buttons and sliders"""
        panel = QGroupBox("Control Panel")
        layout = QVBoxLayout()

        # Section 1: Main Control Buttons
        button_group = QGroupBox("Workflow Controls")
        button_layout = QVBoxLayout()

        self.btn_initialize = QPushButton("1. Initialize Scene")
        self.btn_initialize.setMinimumHeight(40)
        self.btn_initialize.clicked.connect(self.on_initialize_scene)
        button_layout.addWidget(self.btn_initialize)

        self.btn_predict = QPushButton("2. Predict Grasp (CNN Mock)")
        self.btn_predict.setMinimumHeight(40)
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.on_predict_grasp)
        button_layout.addWidget(self.btn_predict)

        self.btn_execute = QPushButton("3. Execute Pick & Lift")
        self.btn_execute.setMinimumHeight(40)
        self.btn_execute.setEnabled(False)
        self.btn_execute.clicked.connect(self.on_execute_pick)
        button_layout.addWidget(self.btn_execute)

        button_group.setLayout(button_layout)
        layout.addWidget(button_group)

        # Section 2: Object Position Sliders
        slider_group = QGroupBox("Object Position Control (Testing)")
        slider_layout = QGridLayout()

        # X slider
        slider_layout.addWidget(QLabel("X Position:"), 0, 0)
        self.slider_x = QSlider(Qt.Orientation.Horizontal)
        self.slider_x.setMinimum(30)
        self.slider_x.setMaximum(70)
        self.slider_x.setValue(50)
        self.slider_x.valueChanged.connect(self.on_object_position_changed)
        slider_layout.addWidget(self.slider_x, 0, 1)
        self.label_x = QLabel("0.50 m")
        slider_layout.addWidget(self.label_x, 0, 2)

        # Y slider
        slider_layout.addWidget(QLabel("Y Position:"), 1, 0)
        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setMinimum(-20)
        self.slider_y.setMaximum(20)
        self.slider_y.setValue(0)
        self.slider_y.valueChanged.connect(self.on_object_position_changed)
        slider_layout.addWidget(self.slider_y, 1, 1)
        self.label_y = QLabel("0.00 m")
        slider_layout.addWidget(self.label_y, 1, 2)

        # Z slider
        slider_layout.addWidget(QLabel("Z Position:"), 2, 0)
        self.slider_z = QSlider(Qt.Orientation.Horizontal)
        self.slider_z.setMinimum(63)
        self.slider_z.setMaximum(80)
        self.slider_z.setValue(65)
        self.slider_z.valueChanged.connect(self.on_object_position_changed)
        slider_layout.addWidget(self.slider_z, 2, 1)
        self.label_z = QLabel("0.65 m")
        slider_layout.addWidget(self.label_z, 2, 2)

        slider_group.setLayout(slider_layout)
        layout.addWidget(slider_group)

        panel.setLayout(layout)
        return panel

    def create_output_panel(self) -> QWidget:
        """Create the output and visualization panel"""
        panel = QGroupBox("Output & Visualization")
        layout = QVBoxLayout()

        # CNN Output Display
        cnn_group = QGroupBox("Predicted 6-DOF Grasp Pose")
        cnn_layout = QVBoxLayout()

        self.grasp_table = QTableWidget(2, 3)
        self.grasp_table.setHorizontalHeaderLabels(["X", "Y", "Z"])
        self.grasp_table.setVerticalHeaderLabels(["Position (m)", "Orientation (°)"])
        self.grasp_table.setMaximumHeight(100)

        # Initialize with placeholder
        for i in range(2):
            for j in range(3):
                self.grasp_table.setItem(i, j, QTableWidgetItem("--"))

        cnn_layout.addWidget(self.grasp_table)
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group)

        # Status Log
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout()

        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMaximumHeight(200)
        self.status_log.setFont(QFont("Courier", 9))
        self.log_status("Application started. Click 'Initialize Scene' to begin.")

        status_layout.addWidget(self.status_log)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        panel.setLayout(layout)
        return panel

    def log_status(self, message: str):
        """Append message to status log"""
        timestamp = time.strftime("%H:%M:%S")
        self.status_log.append(f"[{timestamp}] {message}")

    def on_initialize_scene(self):
        """Handle Initialize Scene button click"""
        self.log_status("Initializing simulation scene...")

        # Get object position from sliders
        obj_pos = [
            self.slider_x.value() / 100.0,
            self.slider_y.value() / 100.0,
            self.slider_z.value() / 100.0
        ]

        success = self.simulation.initialize_scene(obj_pos)

        if success:
            self.state = SimulationState.READY
            self.log_status("✓ Scene initialized successfully.")
            self.log_status("  - Kuka IIWA robot loaded")
            self.log_status("  - Table and object placed")
            self.log_status("  - Ready for grasp prediction")

            # Enable next button
            self.btn_predict.setEnabled(True)
            self.btn_execute.setEnabled(False)
        else:
            self.log_status("✗ Failed to initialize scene.")

    def on_predict_grasp(self):
        """Handle Predict Grasp button click"""
        self.log_status("Capturing camera image...")

        # Capture camera image
        camera_image = self.simulation.get_camera_image()

        self.log_status("Running Mock CNN Grasp Predictor...")

        # Run mock CNN predictor
        grasp_pose = self.simulation.predict_grasp_pose(camera_image)

        # Update display
        self.update_grasp_display(grasp_pose)

        self.state = SimulationState.GRASP_PREDICTED
        self.log_status("✓ Grasp prediction complete!")
        self.log_status(f"  Position: [{grasp_pose.position[0]:.3f}, "
                       f"{grasp_pose.position[1]:.3f}, {grasp_pose.position[2]:.3f}] m")
        self.log_status(f"  Orientation: [{np.degrees(grasp_pose.euler[0]):.1f}°, "
                       f"{np.degrees(grasp_pose.euler[1]):.1f}°, "
                       f"{np.degrees(grasp_pose.euler[2]):.1f}°]")

        # Enable execute button
        self.btn_execute.setEnabled(True)

    def update_grasp_display(self, grasp_pose: GraspPose):
        """Update the grasp pose table with predicted values"""
        # Position row
        self.grasp_table.setItem(0, 0, QTableWidgetItem(f"{grasp_pose.position[0]:.3f}"))
        self.grasp_table.setItem(0, 1, QTableWidgetItem(f"{grasp_pose.position[1]:.3f}"))
        self.grasp_table.setItem(0, 2, QTableWidgetItem(f"{grasp_pose.position[2]:.3f}"))

        # Orientation row (Euler angles in degrees)
        self.grasp_table.setItem(1, 0, QTableWidgetItem(f"{np.degrees(grasp_pose.euler[0]):.1f}"))
        self.grasp_table.setItem(1, 1, QTableWidgetItem(f"{np.degrees(grasp_pose.euler[1]):.1f}"))
        self.grasp_table.setItem(1, 2, QTableWidgetItem(f"{np.degrees(grasp_pose.euler[2]):.1f}"))

    def on_execute_pick(self):
        """Handle Execute Pick & Lift button click"""
        self.log_status("Initializing pick-and-place trajectory...")

        success = self.simulation.execute_pick_and_place()

        if success:
            self.state = SimulationState.EXECUTING
            self.log_status("✓ Trajectory initialized. Executing...")
            self.log_status("  Step 1: Approach")

            # Disable buttons during execution
            self.btn_initialize.setEnabled(False)
            self.btn_predict.setEnabled(False)
            self.btn_execute.setEnabled(False)

            # Start trajectory execution timer
            self.trajectory_timer.start(100)  # Check every 100ms
        else:
            self.log_status("✗ Failed to initialize trajectory.")

    def execute_trajectory_step(self):
        """Execute one step of the pick-and-place trajectory"""
        is_complete, status_msg = self.simulation.step_trajectory()

        self.log_status(f"  {status_msg}")

        if is_complete:
            self.trajectory_timer.stop()
            self.state = SimulationState.COMPLETED
            self.log_status("✓ Grasping sequence complete!")
            self.log_status("  Object successfully grasped and lifted.")

            # Re-enable initialize button for new run
            self.btn_initialize.setEnabled(True)

    def on_object_position_changed(self):
        """Handle object position slider changes"""
        x = self.slider_x.value() / 100.0
        y = self.slider_y.value() / 100.0
        z = self.slider_z.value() / 100.0

        self.label_x.setText(f"{x:.2f} m")
        self.label_y.setText(f"{y:.2f} m")
        self.label_z.setText(f"{z:.2f} m")

        # Update object position if scene is initialized
        if self.state in [SimulationState.READY, SimulationState.GRASP_PREDICTED]:
            self.simulation.set_object_position([x, y, z])

    def update_simulation(self):
        """Update simulation loop (called by timer)"""
        if self.state != SimulationState.UNINITIALIZED:
            self.simulation.step_simulation()

    def closeEvent(self, event):
        """Handle application close event"""
        self.log_status("Shutting down simulation...")
        self.sim_timer.stop()
        self.trajectory_timer.stop()
        self.simulation.cleanup()
        event.accept()


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = RoboticGraspingGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
