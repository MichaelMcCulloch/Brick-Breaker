#!/usr/bin/env python3

import json
import socket
import sys

if __name__ == '__main__':
    server = str(sys.argv[1])
    port = int(sys.argv[2])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = (server, port)
    print('connecting to %s port %s' % server_address)
    sock.connect(server_address)
    try:
        message = json.dumps('This is the message. It will be repeated.')
        print('sending %s' % message)
        sock.sendall(message.encode())

        amount_received = 0
        amount_expected = len(message)
        recv_message=""
        while amount_received < amount_expected:
            data = sock.recv(16)
            recv_message += data.decode()
            amount_received += len(data)

        print('received "%s"' % json.loads(recv_message))
    finally:
        sock.close()

        


