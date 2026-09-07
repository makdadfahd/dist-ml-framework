import socket, json, struct
import numpy as np
from micrograd.multi_dim_engine import Tensor

class TensorConnection :
    def __init__(self, socket):
        self.socket = socket

    def send_tensor(self, tensor) :
        #transforming tensor data into bytes , getting its length , and packing its information in a dict
        arr_bytes = tensor.data.tobytes()
        arr_length = len(arr_bytes)
        arr_info = {
            "dtype" : str(tensor.data.dtype),
            "shape" : tensor.data.shape,
            "length" : arr_length
        }

        #preparing the json info about the array and getting the length of the json info
        json_info = json.dumps(arr_info).encode('utf-8')
        len_json_info = struct.pack('!I', len(json_info))

        #sending information through socket
        self.socket.sendall(len_json_info)
        self.socket.sendall(json_info)
        self.socket.sendall(arr_bytes)

    def recv_tensor(self) :
        #receive the length of the json information
        info_length = self._recv_exact(4) 

        #unpacking the length of the json info
        len_json_info = struct.unpack('!I', info_length)[0]

        #receiving the json info :
        json_info = self._recv_exact(len_json_info).decode('utf-8')
        
        #loading the json info and getting the array length
        arr_info = json.loads(json_info)
        arr_length = arr_info["length"]

        arr_bytes = self._recv_exact(arr_length)

        array_data = np.frombuffer(arr_bytes, dtype= arr_info["dtype"])
        array_data = array_data.reshape(arr_info["shape"])

        return Tensor(array_data)

    #defining a helper function to help us receive the exact number of bytes we need instead of writing loops each time

    def _recv_exact(self, n):
        full_msg = b''
        while len(full_msg) < n :
            chunk = self.socket.recv(n - len(full_msg))
            # a small safety check so we don't receive empty bytes in case of a connection problem
            if not chunk :
                raise ConnectionError("Connection problem while receiving data !!")
            full_msg += chunk
        return full_msg
