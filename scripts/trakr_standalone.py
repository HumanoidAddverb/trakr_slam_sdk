# importing the SimulationApp from Isaac Sim
from isaacsim import SimulationApp

# starting the simulation app
simulation_app = SimulationApp({"headless": False})

import carb

# Importing necessary modules from Isaac Sim
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from trakr import TrakrFlatTerrainPolicy
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.sensor import Camera
from isaacsim.sensors.physics import IMUSensor
from isaacsim.sensors.physx import RotatingLidarPhysX
import isaacsim.core.utils.numpy.rotations as rot_utils

# utils for the server 
from server_data_transfer import TCPServer

# common processing libraries
import multiprocessing
from scipy.spatial.transform import Rotation as R
import numpy as np
import yaml
import cv2
import os
from collections import deque

base_dir = "/home/<username>/your_workspace/dataset"
left_dir = os.path.join(base_dir, "image_0")
right_dir = os.path.join(base_dir, "image_1")
depth_dir = os.path.join(base_dir, "depth")
gt_poses_file = os.path.join(base_dir, "gt_poses.txt")

os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

#importing settings for rendering 
import carb.settings

def get_imu_data_from_buffer(imu_data_stream, t1, t2):
    """
    Retrieves IMU data from the buffer and returns it as a numpy array.
    This function checks if there is any IMU data in the buffer and returns it as a numpy array.
    If the buffer is empty, it returns None.
    Returns:
        np.ndarray or None: IMU data as a numpy array if available, otherwise None.    
    """
    imu_buffer_data =[]

    #Shortcut to deal with the first frame where t1 is 0
    #rendering starts at t2-8, so if t1 is 0, we set it to t2-8
    if t1==0:
        t1 = t2-8

    for e in imu_data_stream:
        if e > t1 and e <= t2:
            imu_buffer_data.append(imu_data_stream[e])

    return imu_buffer_data

def save_images(i, left_image, right_image):
    """
    Saves left and right images to disk with a specific naming convention.

    Args:
        i (int): Index for naming the images.
        left_image (np.ndarray): Left camera image.
        right_image (np.ndarray): Right camera image.
    """
    left_filename = os.path.join(left_dir, f"{i:06d}.png")
    right_filename = os.path.join(right_dir, f"{i:06d}.png")

    cv2.imwrite(left_filename, cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY))
    cv2.imwrite(right_filename, cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY))

def append_kitti_pose_to_file(position, orientation_quat, file_path):
    """
    Appends a single pose in KITTI format to a pose file.

    Args:
        position (list or np.ndarray): [x, y, z]
        orientation_quat (list or np.ndarray): [qx, qy, qz, qw]
        file_path (str): Path to the pose file to append to
    """
    # Convert quaternion to rotation matrix
    rotation = R.from_quat(orientation_quat)  # [qx, qy, qz, qw]
    rotation_matrix = rotation.as_matrix()    # 3x3

    # Form 3x4 pose matrix
    translation_vector = np.array(position).reshape(3, 1)
    pose_matrix = np.hstack((rotation_matrix, translation_vector))  # 3x4

    # Flatten and format for KITTI (12 elements per line)
    pose_kitti = pose_matrix.flatten()
    pose_str = ' '.join(['{:.12e}'.format(val) for val in pose_kitti])

    # Append to file
    with open(file_path, 'a') as f:
        f.write(pose_str + '\n')

#some rendering settings to be applied for smooth rendering 
def apply_render_settings():

    """
        Applies rendering settings to optimize the simulation for smooth and fast rendering.

        The settings control various aspects of the rendering pipeline, such as exposure, 
        tonemapping, anti-aliasing, and post-processing effects, to achieve consistent visuals 
        and faster rendering performance.

        Settings applied:
        - Disable auto exposure and set fixed exposure values.
        - Configure tonemapper settings like aperture, shutter speed, sensitivity, and gamma.
        - Enable FXAA for anti-aliasing.
        - Disable ambient occlusion, screen space reflections, motion blur, and depth of field.
        - Set rendering mode to 'rtx-realtime' for faster rendering.
        - Disable post-processing effects like bloom, color grading, film grain, and chromatic aberration.
        - Configure path tracing settings like samples per pixel (spp).
    """
    settings = carb.settings.get_settings()

    settings.set("/rtx/post/tonemapper/enableAutoExposure", False) # no adaptive brightness 
    settings.set("/rtx/post/tonemapper/exposure", 0.0)
    settings.set("/rtx/post/tonemapper/aperture", 2.8) # controls focus distance of objects in the simulation
    settings.set("/rtx/post/tonemapper/shutterSpeed", 1/60) # camera shutter speed 
    settings.set("/rtx/post/tonemapper/sensitivity", 100)
    settings.set("/rtx/post/tonemapper/type", 0)  # 0 = Linear tonemapping
    settings.set("/rtx/post/tonemapper/gamma", 2.2)  # force consistent gamma

    settings.set("/rtx/post/aa/op", 1)  # FXAA
    settings.set("/rtx/post/ambientOcclusion/enabled", False)   # No darkness in places like corners there are just as bright as other places 
    settings.set("/rtx/post/ssr/enabled", False) # no reflection enabled 
    settings.set("/rtx/post/motionBlur/enabled", False) # blur due to objects being in motion while rendering 

    settings.set("/rtx/rendermode", "rtx-realtime") # a faster rendering mode for the simulation
    settings.set("/rtx/pathtracing/denoiser/enabled", False) # disabled denoiser for faster rendering
    settings.set("/rtx/post/dof/enabled", False) # disabled depth of field for faster rendering
    settings.set("/rtx/pathtracing/spp", 256) # samples per pixel, higher values give better quality but slower rendering (Number of samples used to compute the final color per pixel.)

    # Disable other post effects
    settings.set("/rtx/post/bloom/enabled", False) # no bloom effect around bright areas
    settings.set("/rtx/post/colorGrading/enabled", False) # maintains true colors of the objects in the simulation
    settings.set("/rtx/post/filmGrain/enabled", False) # no film grain effect
    settings.set("/rtx/post/chromaticAberration/enabled", False) # simulates color fringing at high-contrast edges 

# apply_render_settings()

sensor_data_queue = multiprocessing.Queue(maxsize=1)  # Keeps only the latest sensor data
commands_queue = multiprocessing.Queue(maxsize=1)  # Keeps only the latest data

def compute_relative_pose(pose_a, pose_b):
    """
    Compute the relative pose from pose_a to pose_b.
    
    Args:
        pose_a: (translation_a, quaternion_a) — world to frame A
        pose_b: (translation_b, quaternion_b) — world to frame B
        
    Returns:
        relative_translation, relative_quaternion
    """
    t_a, q_a = pose_a
    t_b, q_b = pose_b

    # Convert quaternions to rotation objects
    rot_a = R.from_quat(q_a)
    rot_b = R.from_quat(q_b)

    # Compute relative rotation: q_rel = q_a^-1 * q_b
    rot_rel = rot_a.inv() * rot_b

    # Compute relative translation: R_a^T * (t_b - t_a)
    t_rel = rot_a.inv().apply(t_b - t_a)

    return t_rel, rot_rel.as_matrix()

class Trakr_runner(object):

    def __init__(self) -> None:

        """
            creates the simulation world with preset physics_dt and render_dt and creates an anymal robot inside the warehouse

            Argument:
            physics_dt {float} -- Physics downtime of the scene.
            render_dt {float} -- Render downtime of the scene.

        """
        self.load_params()
        self._world = World(stage_units_in_meters=1.0,
                            physics_dt=self.physics_dt,
                            rendering_dt=self.render_dt)

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")

        # spawn warehouse scene
        prim = define_prim("/World/Warehouse", "Xform")
        asset_path = assets_root_path + self.environment_usd_path
        prim.GetReferences().AddReference(asset_path)

        self.imu_data =None
        self.lidar_data = None
        self.left_camera_transform = None
        self.left_image = None
        self.right_image = None
        self.depth_image = None
        self.left_camera_gt_data = None

        self.trakr = TrakrFlatTerrainPolicy(
            prim_path="/World/trakr",
            name="Trakr",
            position=np.array(self.trakr_spawn_position),
            usd_path=self.trakr_usd_path,
            orientation = self.trakr_spawn_orientation,
        )

        self._base_command = np.zeros(3)

        # bindings for keyboard to command
        self._input_keyboard_mapping = {
            # forward command
            "NUMPAD_8": [1.0, 0.0, 0.0],
            "UP": [1.0, 0.0, 0.0],
            # back command
            "NUMPAD_2": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "NUMPAD_6": [0.0, -1.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            # right command
            "NUMPAD_4": [0.0, 1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }
        self.needs_reset = False
        self.first_step = True

        self.imu_buffer = dict()  # Buffer to store IMU data 

        self.previous_physics_step = 0

        self.current_physics_step = 0

    def load_params(self):

        """
            Loads simulation parameters from a YAML configuration file.

            This function reads the configuration file and initializes various parameters
            required for the simulation, including paths to USD files, spawn positions and
            orientations, sensor configurations, and rendering settings.

            Parameters loaded:
            - environment_usd_path: Path to the USD file for the environment.
            - trakr_usd_path: Path to the USD file for the Trakr robot.
            - trakr_spawn_position: Spawn position of the Trakr robot in the simulation.
            - trakr_spawn_orientation: Spawn orientation of the Trakr robot in the simulation.
            - velocity_magnitude: Magnitude of velocity for robot movement.
            - physics_dt: Physics timestep for the simulation.
            - render_dt: Rendering timestep for the simulation.
            - width, height: Resolution of the camera sensors.
            - K, D: Camera intrinsic matrix and distortion coefficients.
            - pixel_size: Size of a pixel in millimeters.
            - focus_distance: Distance at which the camera is focused.
            - f_stop: Aperture size of the camera lens.
            - transfer: Data transfer mode (e.g., TCP/IP).
            - Sensor translations and rotations: Relative positions and orientations of sensors
            (camera, IMU, LiDAR) with respect to the robot's base link.

            Raises:
            - FileNotFoundError: If the YAML configuration file is not found.
        """

        # opening the yaml file from the disk
        file_path = "../config/run_params.yaml"
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

        # initialising various parameters from the yaml files
        self.environment_usd_path = data["environment_usd_path"]
        self.trakr_usd_path = data["trakr_usd_path"]
        self.trakr_spawn_position = data["trakr_spawn_position"]
        self.trakr_spawn_orientation = data["trakr_spawn_orientation"]
        self.velocity_magnitude = data["velocity_magnitude"]
        self.physics_dt = data["physics_dt"]
        self.render_dt = data["render_dt"]
        self._image_width = data["width"]
        self._image_height = data["height"]
        self._K = data["K"]
        self._D = data["D"]
        self._pixel_size = data["pixel_size"]                    
        self._focus_distance = data["focus_distance"]
        self._f_stop = data["f_stop"]
        self._transfer = data["transfer"]

        # relative translation between the sensors and the base_link in Euler Angles 
        self._translation_body_left_cam = np.array(data["translation_camera_body_left_cam"])
        self._translation_body_right_cam = np.array(data["translation_camera_body_right_cam"])
        self._translation_body_imu = np.array(data["translation_camera_body_imu"])
        self._translation_body_lidar = np.array(data["translation_base_link_lidar"])

        # relative rotation between the sensors and the base_link in Euler Angles 
        self._rotation_body_left_cam = np.array(data["rotation_camera_body_left_cam"])
        self._rotation_body_right_cam = np.array(data["rotation_camera_body_right_cam"])
        self._rotation_body_imu = np.array(data["rotation_camera_body_imu"])
        self._rotation_body_lidar = np.array(data["rotation_base_link_lidar"])


    def setup_sensor_configs(self) -> None:

        """
            Configures and initializes the sensors used in the simulation.

            This function sets up the camera, LiDAR, and IMU sensors with their respective configurations,
            including intrinsic parameters, projection types, and data types. It ensures that the sensors
            are properly initialized and ready to provide data during the simulation.

            Sensor configurations:
            - Camera sensors:
                - Adds motion vectors and distance-to-image-plane data to the camera frames.
                - Sets intrinsic parameters such as focal length, focus distance, lens aperture, and clipping range.
                - Configures the projection type to "pinhole" for undistorted images.
            - LiDAR sensor:
                - Adds depth data and point cloud data to the LiDAR frames.
            - IMU sensor:
                - Initializes the IMU sensor for data collection.

            Raises:
            - ValueError: If any sensor fails to initialize.
        """

        #initializing the camera sensors
        for camera in [self._camera_left, self._camera_right]:
            camera.initialize()
            camera.add_motion_vectors_to_frame()
            camera.add_distance_to_image_plane_to_frame()

            # # Calculate the focal length and aperture size from the camera matrix
            (fx, _, cx, _, fy, cy, _, _, _) = self._K
            horizontal_aperture = self._pixel_size * 1e-3 * self._image_width
            vertical_aperture = self._pixel_size * 1e-3 * self._image_height
            focal_length_x = fx * self._pixel_size * 1e-3
            focal_length_y = fy * self._pixel_size * 1e-3
            focal_length = (focal_length_x + focal_length_y) / 2  # in mm

            # Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
            camera.set_focal_length(focal_length / 10.0)
            camera.set_focus_distance(self._focus_distance)
            camera.set_lens_aperture(self._f_stop * 100.0)
            camera.set_horizontal_aperture(horizontal_aperture / 10.0)
            camera.set_vertical_aperture(vertical_aperture / 10.0)
            camera.set_clipping_range(0.05, 1.0e5)

            # Right now Undistorted images are coming from the camera 
            camera.set_projection_type("pinhole")

        #initialzing the LIDAR sensor
        self._lidar_sensor.initialize()
        self._lidar_sensor.add_depth_data_to_frame() # setting the LIDAR sensor to give depth data
        self._lidar_sensor.add_point_cloud_data_to_frame() # setting the sensor to give pointcloud data 

        #initializing the IMU sensor
        self._imu_sensor.initialize()

    def setup(self) -> None:
        """
            Sets up the simulation environment and initializes sensors.

            This function performs the following tasks:
            - Creates and configures camera sensors (left and right) with their respective parameters.
            - Initializes the LiDAR sensor with rotation frequency, field of view, and resolution.
            - Initializes the IMU sensor with its frequency and orientation.
            - Configures all sensors using the `setup_sensor_configs` method.
            - Adds a physics callback to the simulation world for handling physics updates.

            Sensors initialized:
            - Camera sensors:
                - Left and right cameras with specified resolution, translation, and orientation.
            - LiDAR sensor:
                - Configured with rotation frequency, field of view, and resolution.
            - IMU sensor:
                - Configured with frequency and orientation.

            Physics callback:
            - Registers the `on_physics_step` method to handle physics updates during simulation.

            Raises:
            - ValueError: If any sensor fails to initialize.
        """

        # different prims for different sensors 
        self._camera_left = Camera(
            prim_path=
            "/World/trakr/trakr/base_link/Realsense/RSD455/Camera_OmniVision_OV9782_Left",
            resolution=(self._image_width, self._image_height),
            translation=self._translation_body_left_cam,
            orientation=rot_utils.euler_angles_to_quats(self._rotation_body_left_cam, degrees=True))

        self._camera_right = Camera(
            prim_path=
            "/World/trakr/trakr/base_link/Realsense/RSD455/Camera_OmniVision_OV9782_Right",
            resolution=(self._image_width, self._image_height),
            translation=self._translation_body_right_cam,
            orientation=rot_utils.euler_angles_to_quats(self._rotation_body_right_cam, degrees=True))

        self._lidar_sensor = RotatingLidarPhysX(
            prim_path="/World/trakr/trakr/base_link/Realsense/lidar",
            name="lidar",
            translation=self._translation_body_lidar,
            orientation=rot_utils.euler_angles_to_quats(self._rotation_body_lidar, degrees=True),
            rotation_frequency =0.0,
            fov = (360.0,0.0),
            resolution = (0.3,0.0))

        self._imu_sensor = IMUSensor(
            prim_path="/World/trakr/trakr/base_link/Realsense/RSD455/Imu_Sensor",
            name="imu",
            translation=self._translation_body_imu,
            orientation=rot_utils.euler_angles_to_quats(self._rotation_body_imu, degrees=True))
        
        #setting up all the sensors for the in built configs
        self.setup_sensor_configs()

        # adding the physics callback which gets called every physics step
        self._world.add_physics_callback("physics_step",
                                         callback_fn=self.on_physics_step)
        
    def on_physics_step(self, step_size) -> None:

        """
        Physics call back, initialize robot (first frame) and call robot advance function to compute and apply joint torque

        """

        #initializing the robot 
        if self.first_step:
            self.trakr.initialize()
            self.first_step = False

        elif self.needs_reset:
            self._world.reset(True)
            self.needs_reset = False
            self.first_step = True

        else:
            self.trakr.advance(step_size, self._base_command) #advancing trakr by one step during each physics time step
            imu_frame = self._imu_sensor.get_current_frame()

            imu_data = np.concatenate([
                            np.array([imu_frame['time']]),  # Time in seconds
                            imu_frame['lin_acc'],      # [ax, ay, az]
                            imu_frame['ang_vel'],      # [wx, wy, wz]
                            imu_frame['orientation']   # [qx, qy, qz, qw]
                            ])

            imu_data = imu_data.astype(np.float32)

            self.imu_buffer[imu_frame["physics_step"]] = imu_data

            if len(self.imu_buffer) > 20:
                self.imu_buffer.pop(next(iter(self.imu_buffer)))  # Keep the buffer size manageable

            if not commands_queue.empty():
                latest_command = commands_queue.get_nowait()  # Get the latest data without blocking
                latest_command[0] ,latest_command[1] = -latest_command[1] ,latest_command[0]
                self._base_command = latest_command
    
    def run(self) -> None:

        """
        Step simulation based on rendering downtime 

        """

        i=0

        try:
        
            while simulation_app.is_running():
                
                # rendering the simulation world (NOTE: camera and LIDAR can only simualte in this rate)
                self._world.step(render=True)
    
                if i >= 50:
                    # extracting all the sensor data from the robots                    
                    lidar_frame = self._lidar_sensor.get_current_frame()
                    left_frame = self._camera_left.get_current_frame()
                    right_frame = self._camera_right.get_current_frame()

                    current_time = np.array([lidar_frame['time']])

                    current_time_fp32 = current_time.astype(np.float32)

                    self.current_physics_step = lidar_frame['physics_step']

                    imu_data_stream = self.imu_buffer   
                    t1 =self.previous_physics_step
                    t2 =self.current_physics_step


                    imu_buffer_data = get_imu_data_from_buffer(imu_data_stream, t1, t2)

                    self.previous_physics_step = self.current_physics_step

                    # Retrieve RGB consistently from frames
                    left_image = left_frame['rgba'][..., :3]
                    right_image = right_frame['rgba'][..., :3]

                    # save_images(i, left_image, right_image)  # Save images to disk

                    # Retrieve depth
                    depth_image = left_frame['distance_to_image_plane']

                    # Retrieve lidar depth
                    lidar_data = lidar_frame['depth'][:,0]
                    lidar_data = lidar_data.reshape(1, 1200)

                    # Retrieve camera pose
                    left_camera_transform = self._camera_left.get_world_pose()

                    imu_data = np.concatenate(imu_buffer_data, axis=0)

                    # Prepare GT camera pose data
                    left_camera_transform_position = left_camera_transform[0]   # [x, y, z]
                    left_camera_transform_orientation = left_camera_transform[1]  # [qx, qy, qz, qw]
                    left_camera_gt_data = np.concatenate((left_camera_transform_position, left_camera_transform_orientation))
                    left_camera_gt_data = left_camera_gt_data.astype(np.float32)

                    # next time append relative poses to the text file
                    # append_kitti_pose_to_file(left_camera_transform_position, left_camera_transform_orientation, gt_poses_file)
                    
                    if self._transfer == "tcp_ip":

                        if not sensor_data_queue.full():
                            sensor_data_queue.put({
                                'time': current_time_fp32,
                                'left_image': left_image,
                                'right_image': right_image,
                                'depth_image': depth_image,
                                'lidar_data': lidar_data,
                                'imu_data': imu_data,
                                'left camera_gt_data': left_camera_gt_data
                            })

                if self._world.is_stopped():
                    self.needs_reset = True
                i += 1
            
        except KeyboardInterrupt:

            print("Shutting down gracefully...")
            simulation_app.stop()  # or cleanup here

          
        return
    
def run_tcp_data_server(data_queue, commands_queue):

    #initalizing the TCP server with the queues for data transfer
    tcp_server = TCPServer()

    # Start the TCP server and wait for a client to connect
    client_connected_flag = tcp_server.start()

    #start sending data if the client is connected
    while client_connected_flag:
        if not sensor_data_queue.empty():
            data = sensor_data_queue.get_nowait()

            # Serialize here
            current_time = data['time'].tobytes()
            left_image_bytes = data['left_image'].tobytes()
            right_image_bytes = data['right_image'].tobytes()
            depth_image_bytes = data['depth_image'].tobytes()
            lidar_data_bytes = data['lidar_data'].tobytes()
            imu_data_bytes = data['imu_data'].tobytes()
            left_camera_gt_data_bytes = data['left camera_gt_data'].tobytes()

            combined_data = b"".join([
                current_time,
                left_image_bytes,
                right_image_bytes,
                depth_image_bytes,
                lidar_data_bytes,
                imu_data_bytes,
                left_camera_gt_data_bytes
            ])
            # Send over TCP
            data_teleop = tcp_server.send_data(combined_data)

            # Receive and parse control data
            if data_teleop:
                arr = np.frombuffer(data_teleop, dtype=np.float32)
                if not commands_queue.full():
                    commands_queue.put(arr)

def main():

    """
    Parse arguments and instantiate the trakr runner

    """

    # initilises the trakr runner to create the runner 
    runner = Trakr_runner()

    # process for transfering data through TCP/IP
    tcp_server_process = multiprocessing.Process(target=run_tcp_data_server, args=(sensor_data_queue,commands_queue,), daemon=True)
    tcp_server_process.start()

    # Updating the simulation app and running the rendering loop
    simulation_app.update()
    runner._world.reset()
    simulation_app.update()
    runner.setup()
    simulation_app.update()
    runner.run()
    simulation_app.close()

if __name__ == "__main__":
    main()
