import socket 
import threading


#implement the constants :
server_address = socket.gethostbyname(socket.gethostname())
port = 5050
encoder = 'utf-8'
bytes_number = 1024

#initialise the server socket, bind it to (ADDRESS, PORT) and make the server waiting for connection :
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_address, port))
server_socket.listen()


#make a list for client and their sockets :
clients = []
clients_sockets = []


#functions the make communication easier :
def handle_client() :
    global bytes_number
    while True :
        print('server is listenning and waiting for connection...')
        client_socket , client_address = server_socket.accept()
        global clients_sockets
        clients_sockets.append(client_socket)
        client_socket.send('allias'.encode(encoder))
        allias = client_socket.recv(bytes_number).decode(encoder)
        global clients
        clients.append(allias)
        print(f'connection established with {allias}')
        broadcast(f'{allias} has joined the chat')
    

def broadcast(message) :
    global clients_sockets
    for client_socket in clients_sockets :
        client_socket.send(message.encode(encoder))


def receive_message() :
    while True :
        global clients_sockets , clients , bytes_number
        for client , client_socket in zip(clients, clients_sockets) :
            message  = client_socket.recv(bytes_number).decode(encoder)
            if message == 'leave' :
                index = clients_sockets.index(client_socket)
                client_socket.send('you have left the server'.encode(encoder))
                clients_sockets.remove(client_socket)
                allias = clients[index]
                clients.remove(allias)
                broadcast(f'{allias} has left the chat')
            else :
                broadcast(f'{client} : {message}')
        
            


#threading so all functions can work at the same time :
t1 = threading.Thread(target=handle_client)
t2 = threading.Thread(target=receive_message)
t1.start()
t2.start()
    