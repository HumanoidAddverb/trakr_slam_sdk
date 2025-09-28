import socket
import yaml
import threading

class TCPServer:
    """
    A TCP server that sends all images from a specified directory to a client.
    """

    def __init__(self, config_path="../config/tcp_config.yaml"):
        """
        Loads server configuration from a YAML file.

        :param config_path: Path to the configuration YAML file
        """
        self.config = self.load_config(config_path)
        self.host = self.config["data_server"]["ip"]
        self.port = self.config["data_server"]["port"]
        self.image = None 
        self.conn = None
        self.lock = threading.Lock()


    def load_config(self, config_path):
        """Reads configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def start(self) -> bool:
        """
        Starts the server, listens for clients, and sends multiple images.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)  # Allow one client at a time
        print(f"Server listening on {self.host}:{self.port}...")
        self.conn, addr = self.server_socket.accept()
        print(f"Client connected from {addr}")
        
        return True

    def send_data(self,data):
        try:
        
            self.conn.sendall(data)
            command = self.conn.recv(12)
            return command
        except (socket.error, ConnectionResetError, BrokenPipeError):
            self.conn = None 
        

    def __del__(self):
        if self.conn:
            self.conn.close()

# Run the server
if __name__ == "__main__":
    server = TCPServer(config_path="tcp_data_transfer/config.yaml")
 