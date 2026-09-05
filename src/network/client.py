import socket 
import threading


#implement the constants :
dest_address = socket.gethostbyname(socket.gethostname())
dest_port = 5050
encoder = 'utf-8'
name = input('your name please : ')
bytes_number = 1024

#intialize the client socket, and connect to the server :
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((dest_address, dest_port))

#def some functions :
def receive_message():
    while True :
        global bytes_number , client_socket , encoder
        message = client_socket.recv(bytes_number).decode(encoder)
        if message == 'allias' :
            client_socket.send(name.encode(encoder))
        else :
            print(message)

def send_message() :
    while True :
        global client_socket , encoder 
        message = input()
        client_socket.send(message.encode(encoder))

#now we move to threading :
t1 = threading.Thread(target=receive_message)
t2 = threading.Thread(target=send_message)
t1.start()
t2.start()