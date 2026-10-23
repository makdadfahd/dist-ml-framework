import socket , struct , json
from network.tensorconnection import TensorConnection
from micrograd.multi_dim_engine import Tensor

class RPCClient() :
    def __init__(self, socket) :
        self.conn = TensorConnection(socket)

    def call(self, function_name, arguments=[]) :
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
        if tensor_list :
            self.conn.send_tensor(tensor_list)

        #receiving response
        response = self.conn._recv_msg()
        status = response["status"]

        real_response = []
        #checking status 
        if status : 
            result_list = response["result"]
            tensor_count = result_list.count("TENSOR")
            tensor_list = []
            if tensor_count > 0 :
                tensor_list = self.conn.recv_tensor()
            j = 0    
            for i in range(len(response["result"])) :
                if response["result"][i] == "TENSOR" :
                    real_response.append(tensor_list[j])
                    j += 1
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
        tensor_count = args.count("TENSOR")

        tensors = []
        j = 0

        if tensor_count > 0 :
            tensors = self.conn.recv_tensor()
            
        for arg in args :
            if arg == "TENSOR"  :
                new_args.append(tensors[j])
                j += 1
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

            self.conn._send_msg(response)

            if len(tensors_list) > 0 :
                self.conn.send_tensor(tensors_list)
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

    def push_grads(self, *gradients) :
        for i , gradient in enumerate(gradients) :
            self.params[i].grad = gradient.data
        
        self.optimizer.step()
        
        self.optimizer.zero_grad()
        return "gradients pushed"
        