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
