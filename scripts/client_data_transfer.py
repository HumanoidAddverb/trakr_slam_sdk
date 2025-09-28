import socket
import yaml
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import threading
# Define image dimensions (modify based on your camera resolution)
WIDTH, HEIGHT = 640, 480  # Example resolution
CHANNELS = 3  # RGB images

# Calculate sizes
left_image_size = WIDTH * HEIGHT * CHANNELS
right_image_size = WIDTH * HEIGHT * CHANNELS
depth_image_size = WIDTH * HEIGHT * 4  # Assuming 32-bit depth data
lidar_data_size = 1200 * 1 * 4  # Assuming LiDAR uses similar format
imu_data_size = 40  # Example: Quaternion (4 floats of 4 bytes each)

def save_lidar_heatmap(depth_array, filename="lidar_heatmap.png"):
    plt.figure(figsize=(10, 5))
    plt.imshow(depth_array, cmap='jet', aspect='auto')
    plt.colorbar(label="Depth")
    plt.xlabel("Width (Scan Index)")
    plt.ylabel("Height (Scan Line)")
    plt.title("LiDAR Depth Heatmap")

    # Save the figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    print(f"Heatmap saved as {filename}")

class TCPClient:
    """
    A TCP client that receives multiple images from the server and saves them dynamically.
    """

    def __init__(self, config_path="../config/tcp_config.yaml"):
        """
        Loads client configuration from YAML file.

        :param config                    print(data_size)
_path: Path to the YAML configuration file.
        """
        self.config = self.load_config(config_path)
        self.server_ip = self.config["data_client"]["server_ip"]
        self.port = self.config["data_client"]["port"]
        self.frame_count = 0
        self.scan_index = 0

        # Path to CSV file
        self.csv_file = 'imu_data.csv'

        # Write header (only once)
        with open(self.csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Qw', 'Qx', 'Qy', 'Qz'])

    def load_config(self, config_path):
        """Reads configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
        
    def start(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.server_ip, self.port))

        return True
    
    def stack_images_grid(self, img1, img2, img3, img4):
        img1 = cv2.resize(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), (200, 200))
        img2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), (200, 200))
        img3 = cv2.resize(img3, (200, 200))
        img4 = cv2.resize(img4, (200, 200))
        row1 = np.hstack((img1, img2))
        row2 = np.hstack((img3, img4))  # Padding with a black image

        combined = np.vstack((row1, row2))

        return combined
    
    def lidar_scatter_cv2(self, scan, image_size=(500, 500)):
        """
        Visualizes a single LiDAR scan as a scatter plot in OpenCV.
        
        Parameters:
            lidar_data: NumPy array of shape (240, 141)
            scan_index: Index of the scan to visualize
            image_size: Size of the output image (height, width)
        """
        height, width = image_size
        
        # Normalize depth values to fit in the image
        scan_normalized = cv2.normalize(scan, None, 0, height - 1, cv2.NORM_MINMAX).astype(int)

        # Create a blank black image
        scatter_image = np.zeros((height, width, 3), dtype=np.uint8)

        # X-coordinates: spread evenly across the width
        x_coords = np.linspace(50, width - 50, len(scan)).astype(int)

        
        # Y-coordinates: Use normalized depth values (inverted for OpenCV image axis)
        y_coords = height - 1 - scan_normalized  # Invert to match OpenCV's (0,0) at top-left

        # Draw scatter points
        for x, y in zip(x_coords, y_coords):

            cv2.circle(scatter_image, (x, y[0]), 3, (0, 255, 255), -1)  # Yellow points

        return scatter_image


    def receive(self):
        """
        Connects to the server, receives multiple images, and saves them dynamically.
        Measures the data rate (bytes per second).
        """
        data_size = 3076840

        # Start timing
        start_time = time.time()

        # Receive the actual data
        received_data = b""
        while len(received_data) < data_size:
            packet = self.client_socket.recv(data_size - len(received_data))
            if not packet:
                break
            received_data += packet

        # End timing
        end_time = time.time()

        # Calculate elapsed time and data rate
        elapsed_time = end_time - start_time
        data_rate = len(received_data) / elapsed_time  # Bytes per second
        frequency = 1 / elapsed_time  # Frequency in Hz


        print(f"Received {len(received_data)} bytes of data in {elapsed_time:.4f} seconds")
        print(f"Data rate: {data_rate:.2f} bytes/second")
        print(f"Frequency: {frequency:.2f} Hz")
        # Extract each component
        offset = 0

        left_image_bytes = received_data[offset : offset + left_image_size]
        offset += left_image_size

        right_image_bytes = received_data[offset : offset + right_image_size]
        offset += right_image_size

        depth_image_bytes = received_data[offset : offset + depth_image_size]
        offset += depth_image_size

        lidar_data_bytes = received_data[offset : offset + lidar_data_size]
        offset += lidar_data_size

        imu_data_bytes = received_data[offset : offset + imu_data_size]

        # Convert image data to NumPy arrays
        left_image = np.frombuffer(left_image_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))
        right_image = np.frombuffer(right_image_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, CHANNELS))

        # Convert depth & LiDAR data (assuming 16-bit unsigned integers)
        depth_image = np.frombuffer(depth_image_bytes, dtype=np.float32).reshape((HEIGHT, WIDTH))
        lidar_data = np.frombuffer(lidar_data_bytes, dtype=np.float32).reshape((1, 1200))

        # IMU Data (Example: Quaternion - adjust based on format)
        imu_data = np.frombuffer(imu_data_bytes, dtype=np.float32).tolist()

        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)  # Convert to 8-bit for display

        # Apply a colormap for better visualization
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        scan = lidar_data[self.scan_index,:]  # Select one scan (a single row)
        print(left_image[0][0])

        scan_scatterd_2d = self.lidar_scatter_cv2(scan)

        combined = self.stack_images_grid(left_image, right_image, depth_colored, scan_scatterd_2d)

        unix_timestamp = time.time()

        print("client_side", unix_timestamp, imu_data)  # Print IMU data for verification

        # Append data to CSV progressively
        with open(self.csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([unix_timestamp] + imu_data)

        # Save images
        cv2.imshow("Combined Image", combined)
        cv2.waitKey(10)

        # Function to get command from the keyboard asynchronously
        def get_command():
            command_mapping = {
            'w': [0.4, 0.0, 0.0],  # Forward
            's': [-0.4, 0.0, 0.0], # Backward
            'a': [0.0, 0.4, 0.0],  # Left
            'd': [0.0, -0.4, 0.0], # Right
            'm': [0.0, 0.0, 0.4],  # Up
            'n': [0.0, 0.0, -0.4], # Down
            'x': [0.0, 0.0, 0.0]   # Stop
            }

            while True:
                user_input = input("Enter command (w/s/a/d/q/e/x): ").strip().lower()
                if user_input in command_mapping:
                    return np.array(command_mapping[user_input], dtype=np.float32)
                else:
                    print("Invalid input. Please enter one of the following: w, s, a, d, q, e, x.")

        # Start a thread to get the command
        command_thread = threading.Thread(target=get_command, daemon=True)
        command_thread.start()

        # Wait for the user to input a command
        command = get_command()
        command = np.array([0.4, 0.0, 0.0], dtype=np.float32)
        self.client_socket.sendall(command.tobytes())  # send as raw byte

        return
# Run the client
if __name__ == "__main__":
    client = TCPClient(config_path="../config/tcp_config.yaml")
    client.receive()
