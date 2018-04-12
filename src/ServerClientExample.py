#!/usr/bin/env python3

#SERVER

import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = ('192.168.0.12', 10000)
print('starting up on %s port %s' % server_address)
sock.bind(server_address)

sock.listen(1)

while True:
    print('waiting for connection')
    connection, client_address = sock.accept()
    try:
        print('connection from', client_address)
        while True:
            data = connection.recv(160)
            print('received "%s"' % data)
            if data:
                print("responding to client")
                connection.sendall(data)
            else:
                print("no more data from client", client_address)
                break
    finally:
        connection.close()

#CLIENT

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect the socket to the port where the server is listening
server_address = ('192.168.0.13', 10000)
print('connecting to %s port %s' % server_address)
sock.connect(server_address)
try:
    
    # Send data
    message = 'This is the message.  It will be repeated.'
    print('sending "%s"' % message)
    sock.sendall(message.encode())

    # Look for the response
    amount_received = 0
    amount_expected = len(message)
    
    while amount_received < amount_expected:
        data = sock.recv(160)
        amount_received += len(data)
        print('received "%s"' % data.decode())

finally:
    print('closing socket')
    sock.close()
