#!/usr/bin/env python3

from threading import Thread
import socket
import sys
import json
import csv

class ClientThread(Thread):
    def __init__(self, host, port):
        Thread.__init__(self)
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))

    def run(self):
        self.sock.listen(5)
        while True:
            client, address = self.sock.accept()
            client.settimeout(3600)
            
    def listen_to_client(self, client, address):
        size = 1024
        while True:
            try:
                data = client.recv(size)
                if data:
                    # Set the response to echo back the recieved data 
                    response = data
                    client.send(response)
                else:
                    break
            except:
                client.close()
                return False

    
    


if __name__ == '__main__':
    config = json.load(open("config.json"))
    port_start = config['Port_Start']

    for i, client in enumerate(config['Client_List']):
        print(i, client)
        #start a thread for that client and send it the data for it to begin processing



    while True:
        port_num = input("Port? ")
        try:
            port_num = int(port_num)
            break
        except ValueError:
            pass

    ThreadedServer('',port_num).listen()
    print("HERE")
