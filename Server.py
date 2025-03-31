import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import pickle
import numpy as np

update_model = {}
running = True

def startCentralizedServer():
    class UpdateModel(Thread): 
 
        def __init__(self,ip,port): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port 
            print('Request received from Client IP : '+ip+' with port no : '+str(port)+"\n") 
 
        def run(self): 
            data = conn.recv(1000000)
            data = pickle.loads(data)
            encodings = data[0]
            names = data[1]
            blinks = data[2]
            np.save("global/encoding", encodings)
            np.save("global/names", names)
            np.save("global/blinks", blinks)
            print("Global model successfully updated")
            conn.send(str('Model updated to server successfully').encode())         
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 2222))
    print("Centralized Server Started & waiting for incoming connections\n\n")
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = UpdateModel(ip,port) 
        newthread.start() 
    
def startServer():
    Thread(target=startCentralizedServer).start()

startServer()

