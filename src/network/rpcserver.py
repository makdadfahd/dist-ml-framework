import socket , struct , json
from network.tensorconnection import TensorConnection
from micrograd.multi_dim_engine import Tensor
def add(*args) :
    total = 0
    for arg in args :
        total += arg
    return total

class RPCServer() :   
    def __init__(self, socket) :
        self.conn = TensorConnection(socket)
        self.functions = {"add" : add}

    def dispatch(self, function_name , args = []) :
        try :
            #trying if the function actually exists
            func = self.functions[function_name]

            try :
                #after we made sure it exists we call it with arguments passed
                result  = func(*args)
                response = {
                    'status' : True ,
                    'result' : result
                }
            except Exception as e :
                #if it crashes we tell the user why it crashed
                response = {
                    'status' : False ,
                    'message' : str(e)
                }

        except Exception :
            #if it doesn't exist we return an immediate response 
            response = {
                'status' : False ,
                'message' : 'function is not defined'
            }
        return response

    def serve_request(self) :
        request = self.conn._recv_msg()
        func = request["function_name"]
        args = request["args"]
        new_args = []
        for arg in args :
            if arg == "TENSOR" :
                tensor = self.conn.recv_tensor()
                new_args.append(tensor)
            else :
                new_args.append(arg)

        response = self.dispatch(func , new_args)

        if response["status"] :
            if isinstance(response["result"], Tensor) :
                tensor = response["result"]
                response["result"] = "TENSOR"
                self.conn._send_msg(response)
                self.conn.send_tensor(tensor)
            else :
                self.conn._send_msg(response)
        else :
            self.conn._send_msg(response)

    def call(self, function_name, arguments) :
        #sending request
        new_arguments = []
        tensor_list = []

        for arg in arguments :
            if isinstance(arg, Tensor) :
                new_arguments.append("TENSOR")
                tensor_list.append(arg)
            else :
                new_arguments.append(arg)

        request = {
            "function_name" : function_name,
            "args" : new_arguments
        }

        self.conn._send_msg(request)
        for tensor in tensor_list :
            self.conn.send_tensor(tensor)

        #receiving response
        response = self.conn._recv_msg()
        status = response["status"]

        #checking status 
        if status : 
            if response["result"] == "TENSOR" :
                tensor = self.conn.recv_tensor()
                return tensor
            else :
                return response["result"]
        else :
            raise RuntimeError(response["message"])

    def handle_client(self) :
        while True :
            try :
                self.serve_request()
            except ConnectionError :
                break
