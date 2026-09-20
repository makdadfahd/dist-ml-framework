import socket , struct , json
from network.tensorconnection import TensorConnection
from micrograd.multi_dim_engine import Tensor

class RPCClient() :
    def __init__(self, socket) :
        self.conn = TensorConnection(socket)

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

        real_response = []
        #checking status 
        if status : 
            for i in range(len(response["result"])) :
                if response["result"][i] == "TENSOR" :
                    tensor = self.conn.recv_tensor()
                    real_response.append(tensor)
                else :
                    real_response.append(response["result"][i])
            return real_response

        else :
            raise RuntimeError(response["message"])

class RPCServer() :   
    def __init__(self, socket, params , optimzer ) :
        self.conn = TensorConnection(socket)
        self.functions = {"get_weights" : self.get_weights,
                          "push_grads" : self.push_grads}
        self.params = params
        self.optimizer = optimzer

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
        tensors_list = []

        if response["status"] :
            for i in range(len(response["result"])) :
                if isinstance(response["result"][i], Tensor) :
                    tensor = response["result"][i]
                    response["result"][i] = "TENSOR"
                    tensors_list.append(tensor)

            self.conn._send_msg(response["result"])

            if len(tensors_list) > 0 :
                for tensor in tensors_list:
                    self.conn.send_tensor(tensor)
        else :
            self.conn._send_msg(response)

    

    def handle_client(self) :
        while True :
            try :
                self.serve_request()
            except ConnectionError :
                break

    def get_weights(self) :
        params = []
        for param in self.params :
            params.append(param)
        return params

    def push_grads(self, gradients) :
        for i , gradient in enumerate(gradients) :
            self.params[i].grad = gradient

        self.optimizer.step()
        self.optimizer.zero_grad()
        