import socket, json, struct
import numpy as np
from micrograd.multi_dim_engine import Tensor

class TensorConnection :
    def __init__(self, socket):
        self.socket = socket

    def send_tensor(self, tensor) :
        tensor_info , tensor_data_bytes = self.tensor_info(tensor)
        self._send_msg(tensor_info)
        self.socket.sendall(tensor_data_bytes)

    def recv_tensor(self) :
        tensor_info = self._recv_msg()
        tensor = self.construct_tensor(tensor_info)
        return tensor

    def _recv_exact(self, n):
        full_msg = b''
        while len(full_msg) < n :
            chunk = self.socket.recv(n - len(full_msg))
            if not chunk : #safety check
                raise ConnectionError("Connection problem while receiving data !!")
            full_msg += chunk
        return full_msg

    def _send_msg(self, msg) :
        json_data = json.dumps(msg).encode('utf-8')
        len_data = struct.pack('!I', len(json_data))
        self.socket.sendall(len_data)
        self.socket.sendall(json_data)

    def tensor_info(self, tensor) :
        tensor_data_bytes = tensor.data.tobytes()
        tensor_data_length = len(tensor_data_bytes)
        tensor_dict = {
            "dtype" : str(tensor.data.dtype),
            "shape" : tensor.data.shape,
            "length" : tensor_data_length
        }
        return (tensor_dict, tensor_data_bytes)

    def construct_tensor(self, tensor_dict) :
        tensor_length = tensor_dict["length"]
        tensor_data_bytes = self._recv_exact(tensor_length)
        tensor_data = np.frombuffer(tensor_data_bytes, dtype=tensor_dict["dtype"])
        tensor = Tensor(tensor_data.reshape(tensor_dict["shape"]))
        return tensor

    def _recv_msg(self) :
        info_length = self._recv_exact(4)
        length_json_info = struct.unpack('!I', info_length)[0]
        json_info = self._recv_exact(length_json_info).decode('utf-8')
        msg = json.loads(json_info)
        return msg