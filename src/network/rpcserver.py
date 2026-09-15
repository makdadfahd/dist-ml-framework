import socket , struct , json
from network.tensorconnection import TensorConnection

class RPCServer() :   
    def __init__(self) :
        self.conn = TensorConnection(socket)
        self.functions = {}

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
        json_info = self.conn._recv_exact(len_json_info).decode('utf-8')

        #we load the request of the client, we get the function name and args (if they exist), and we dispatch them to give a response
        request = json.loads(json_info)
        func = request["function_name"]
        args = request["args"]
        response = self.dispatch(func , args)

        #we send the response back to the client

    def call(self, function_name, arguments) :
        pass