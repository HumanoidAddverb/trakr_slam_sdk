# Isaac Sim Quadruped (`Trakr`)

This repository contains the necessary files and scripts to run a `trakr` quadruped robot simulation in NVIDIA Isaac Sim. The project is configured for reinforcement learning-based control, featuring environment setup, sensor simulation, and a client-server architecture for data streaming and remote control.

## Table of Contents

- [Isaac Sim Quadruped (`Trakr`)](#isaac-sim-quadruped-trakr)
  - [Table of Contents](#table-of-contents)
  - [1. Getting Started](#1-getting-started)
    - [Installation](#installation)
  - [2. Project Structure](#2-project-structure)
    - [Breakdown](#breakdown)
  - [3. Configuration Deep Dive](#3-configuration-deep-dive)
    - [`env.yaml`](#envyaml)
    - [`run_params.yaml`](#run_paramsyaml)
    - [`tcp_config.yaml`](#tcp_configyaml)
  - [4. Running the Simulation](#4-running-the-simulation)
  - [5. Robot and Policy](#5-robot-and-policy)
  - [6. Troubleshooting](#6-troubleshooting)

---

## 1. Getting Started

### Installation

1. **Download and install Isaac Sim Standalone:**
```bash
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip
mkdir ~/isaacsim
unzip "isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip" -d ~/isaacsim
```
2. **Clone this repository inside the Isaac Sim installation path:**

```bash
cd ~/isaacsim/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/
git clone https://github.com/HumanoidAddverb/trakr_slam_sdk.git
```
## 2. Project Structure

The project is organized into four main directories:

```
.
├── config/             # YAML files for simulation, environment, and network settings.
├── models/             # Contains the pre-trained neural network policy.
├── resources/          # Robot 3D models (USD, URDF) and meshes (STL).
└── scripts/            # Python scripts to run the simulation and client/server logic.
```

### Breakdown

*   **`config/`**:
    *   `env.yaml`: Defines the RL environment, including physics, observations, rewards, and terminations.
    *   `run_params.yaml`: High-level simulation parameters, asset paths, and sensor configurations.
    *   `tcp_config.yaml`: Network settings for the client-server communication.
*   **`models/`**:
    *   `policy.pt`: The trained PyTorch model that controls the quadruped's locomotion.
*   **`resources/`**:
    *   `meshes/`: Contains all the individual `.STL` files for the robot's links.
    *   `robot_description/`: Contains the robot's full description files, including `trakr.urdf` and the necessary USD files (`trakr_imu.usd`, `full_warehouse.usd`) for Isaac Sim.
*   **`scripts/`**:
    *   `trakr_standalone.py`: The main entry point. It launches the Isaac Sim application, sets up the environment, and runs the simulation. It also acts as the TCP server.
    *   `trakr.py`: Defines the `TrakrFlatTerrainPolicy` class, which loads the RL policy and computes joint torques based on observations.
    *   `server_data_transfer.py`: A simple TCP server class for sending data.
    *   `client_data_transfer.py`: A TCP client that receives and visualizes sensor data, and sends keyboard commands back to the simulation.

## 3. Configuration Deep Dive

### `env.yaml`

This is the configuration file, defining the reinforcement learning environment for training the policy.

*   **`sim`**: Contains core physics settings, including the physics `dt` (0.002s), gravity, and detailed PhysX engine parameters.
*   **`scene`**: Defines all assets in the simulation.
    *   **`robot`**: Specifies the robot's USD path, initial state (position, rotation, joint angles), actuator properties (PD controller gains), and collision settings.
    *   **`terrain`**: Configures the ground plane. It can generate various curricula of terrains, from flat to random rough patches.
    *   **`contact_forces`**: Sets up a sensor to detect contacts on the robot's body, used for termination conditions.
*   **`observations`**: Defines the observation vector fed to the policy network. It includes:
    *   Base linear and angular velocity.
    *   Projected gravity vector.
    *   Velocity commands.
    *   Joint positions and velocities.
    *   The previous action taken.
*   **`rewards`**: A collection of weighted reward functions that guided the policy's training. These include rewarding desired linear/angular velocity, proper gait, and foot clearance, while penalizing high torques, base motion, and unwanted contact.
*   **`terminations`**: Conditions that end an episode, such as the robot falling over (`body_contact`), going out of bounds, or timing out.
*   **`commands`**: Defines how target velocity commands are generated for the robot to follow.

### `run_params.yaml`

This file provides high-level parameters and asset paths used by the simulation scripts.

*   **Control Parameters**: `action_scale`, `p_gain`, `d_gain`, and `default_joint_pos` for the policy and joint controller.
*   **Asset Paths**: Specifies the exact paths to the `environment_usd_path` and `trakr_usd_path`. **You may need to edit these paths if your project is not in the default location.**
*   **Spawning Info**: Defines the robot's initial `trakr_spawn_position` and `trakr_spawn_orientation`.
*   **Sensor Intrinsics**: Contains camera parameters (`K`, `D`), resolution (`width`, `height`), and sensor-to-base_link transforms.

### `tcp_config.yaml`

This file configures the network settings for the client-server communication.

*   **`data_server` / `data_client`**: Sets the IP and port for the primary data stream (sensor data from sim to client, commands from client to sim).
*   **`teleop_server` / `teleop_client`**: Configures a secondary connection, likely intended for a separate teleoperation interface.

## 4. Running the Simulation

The entire project can be run using the trakr_standalone.py script, which launches the Isaac Sim environment and the data server. For visualization and control, you will then run the client script.

**1. Launch the Simulation**

Navigate to the `scripts` directory in your terminal and execute:
```bash
cd ~/isaacsim/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/trakr_simulator/scripts/
```

**2. Update the Paths**  
Before running the simulation, make sure to edit the following paths in your setup:

- In `config/run_params.yaml`, update these entries to match your local installation path:   
  
  ```yaml
  trakr_usd_path: "/home/<username>/isaacsim/.../trakr_simulator/resources/robot_description/trakr_imu.usd"
  policy_path:   "/home/<username>/isaacsim/.../trakr_simulator/models/policy.pt"
  env_config_path: "/home/<username>/isaacsim/.../trakr_simulator/config/env.yaml"
- In `scripts/trakr_standalone.py`, update the dataset output path:
  ```bash
  base_dir = "/home/<username>/your_workspace/dataset"
   ```
**3. Launch the Simulation**

Open a **second terminal**, navigate to the `scripts` directory, and run:

```bash
../../python.sh trakr_standalone.py
```

**4. Controlling the Robot**  
Once the simulation is running, open a new terminal and start the TCP client to connect and send control commands to the robot.
bash
```
cd ~/isaacsim/isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release/trakr_simulator/scripts/

../../python.sh client_data_transfer.py
```
This will:  
Receive RGB, Depth, LiDAR, and IMU data streams from the server.  
Display a 2×2 visualization panel (left camera, right camera, depth map, LiDAR scatter).  
Save IMU data to imu_data.csv
## 5. Robot and Policy

*   **Robot Model**: The `trakr` robot is defined by the `trakr.urdf` file for its kinematics and dynamics, and the `trakr_imu.usd` file is used to represent it visually and physically within Isaac Sim.
*   **Control Policy**: The robot's locomotion is not manually coded. Instead, it is driven by the neural network in `models/policy.pt`. This network was trained using reinforcement learning to map sensor observations to low-level joint actions to achieve stable walking and follow velocity commands.

## 6. Troubleshooting

*   **Path Errors:** If the script fails to find a USD or model file, double-check the paths in `run_params.yaml`. They may need to be absolute paths depending on your setup.
*   **TCP Connection Fails:**
    *   Ensure the server (`trakr_standalone.py`) is running *before* you start the client.
    *   Verify that the IP addresses in `tcp_config.yaml` are correct. Use `127.0.0.1` if running on the same machine. If on different machines, use the server's network IP and ensure no firewalls are blocking the port.
*   **Client Visuals Don't Appear:** Make sure you have a desktop environment and that `opencv-python` and `matplotlib` are installed correctly.
*   **Slow Performance:** This simulation is computationally intensive. If performance is poor, you can try:
    *   Running Isaac Sim in `headless` mode.
    *   Reducing the rendering resolution in `run_params.yaml`.
    *   Disabling advanced rendering features in `env.yaml`.
*   **Robot is Unstable:** The policy's performance is tied to the physics parameters in `env.yaml` and `run_params.yaml`. Changes to `p_gain`, `d_gain`, `physics_dt`, or mass properties can affect stability.