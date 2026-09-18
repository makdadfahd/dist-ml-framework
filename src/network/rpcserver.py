import socket , struct , json
from network.tensorconnection import TensorConnection

def add(*args) :
    return sum(args)

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
        #we receive the json info length 
        info_length = self.conn._recv_exact(4)
        len_json_info = struct.unpack('!I', info_length)[0]

        #we call to receive the exact bytes number received in the length of the json info
        request_json = self.conn._recv_exact(len_json_info).decode('utf-8')

        #we load the request of the client, we get the function name and args (if they exist), and we dispatch them to give a response
        request = json.loads(request_json)
        func = request["function_name"]
        args = request["args"]
        response = self.dispatch(func , args)

        #we send the response back 
        response_json = json.dumps(response).encode('utf-8')
        self.conn._send_msg(response_json)


    def call(self, function_name, arguments) :
        #we send the request
        request = {
            "function_name" : function_name,
            "args" : arguments
        }
        request_json = json.dumps(request).encode('utf-8')
        self.conn._send_msg(request_json)

        #we receive the response 
        response_length = self.conn._recv_exact(4)
        len_response_json = struct.unpack('!I', response_length)[0]

        response_json = self.conn._recv_exact(len_response_json).decode('utf-8')
        response = json.loads(response_json)

        status = response["status"]

        #we check the response status , if true we get the response if not we get the alert message as an error
        if status : 
            return response["result"]
        else :
            raise RuntimeError(response["message"])

    def handle_client(self) :
        while True :
            try :
                self.serve_request()
            except ConnectionError :
                break
