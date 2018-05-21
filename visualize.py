import os
import struct
import socket
import subprocess
from numpy import *

def recvFloats(sock, nFloats):
    nBytesRemains = nFloats * 4
    buf = bytearray(0)
    while nBytesRemains > 0:
        bufRecv = sock.recv(nBytesRemains)
        nBytesRemains -= len(bufRecv)
        buf = buf + bufRecv
    data = frombuffer(buf, float32)
    assert data.size == nFloats
    return data

class Display:
    def __init__(self, nx, ny, portno=13888):
        path = os.path.dirname(os.path.abspath(__file__))
        cmd = [os.path.join(path, 'server.exe'), str(nx), str(ny), str(portno)]
        self.server = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        while self.server.stdout.readline().strip() != "READY": pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", portno))
        self.nx = nx
        self.ny = ny

    def refresh(self, x0, x1, y0, y1):
        data = getbuffer(array([x0, x1, y0, y1], float32))
        self.sock.sendall(data)
        data = recvFloats(self.sock, self.nx * self.ny * 3)
        self.data = data.reshape([self.nx, self.ny, 3])
        img = log(self.data[:,:,0]) / log(self.data.max())
        imshow(self.data[:,:,0], cmap='hot',
                norm=matplotlib.colors.LogNorm(vmin=1, vmax=self.data.max()))
        gca().set_facecolor('black')
        colorbar()

    def shutdown(self):
        self.sock.sendall(bytearray(16))
        self.server.wait()

    def __del__(self):
        self.shutdown()

d = Display(1024, 1024)
d.refresh(1, 15, -0.02, 0.12)
