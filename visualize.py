import os
import time
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

def sendBytes(sock, data):
    assert isinstance(data, bytes)
    sock.sendall(struct.pack('I', len(data)))
    sock.sendall(data)

def connectServer(portno):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        time.sleep(0.1)
        try:
            sock.connect(("127.0.0.1", portno))
            return sock
        except:
            continue

class MouseSelect:
    def __init__(self, callback):
        self.callback = callback

    def press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.callback(event)

class Display:
    def __init__(self, nx, ny, nU, portno=10088):
        path = os.path.dirname(os.path.abspath(__file__))
        cmd = [os.path.join(path, 'server.exe')]
        cmd.extend([str(nx), str(ny), str(nU), str(portno)])
        self.server = subprocess.Popen(cmd)
        self.sock = connectServer(portno)

        self.nx = nx
        self.ny = ny
        self.nU = nU
        self.nGrids = 160

        self.exitMsg = struct.pack('II', 4, 0)

        fig = figure(1)
        self.mouse = MouseSelect(self.mouseRelease)
        fig.canvas.mpl_connect('button_press_event', self.mouse.press)
        fig.canvas.mpl_connect('button_release_event', self.mouse.release)

    def refresh(self, x0, x1, y0, y1):
        self.viewport = [x0, x1, y0, y1]
        sendBytes(self.sock, struct.pack('Iffff', 1, x0, x1, y0, y1))
        data = recvFloats(self.sock, self.nx * self.ny * 3)
        self.data = data.reshape([self.nx, self.ny, 3])
        figure(1)
        clf()
        imshow(self.data[:,:,0], cmap='hot', origin='lower',
                norm=matplotlib.colors.LogNorm(vmin=.5, vmax=self.data.max()),
                extent=self.viewport)
        gca().set_facecolor('black')
        gca().set_aspect('auto')
        colorbar()
        draw()

    def mouseRelease(self, event):
        x0 = min(self.mouse.x0, self.mouse.x1)
        x1 = max(self.mouse.x0, self.mouse.x1)
        y0 = min(self.mouse.y0, self.mouse.y1)
        y1 = max(self.mouse.y0, self.mouse.y1)
        self.drawProfiles(x0, x1, y0, y1)

    def drawProfiles(self, x0, x1, y0, y1):
        sendBytes(self.sock, struct.pack('Iffff', 2, x0, x1, y0, y1))
        pdata = recvFloats(self.sock, self.nU * self.nGrids)
        self.pdata = pdata.reshape([self.nU, self.nGrids])
        print(self.pdata.max())
        figure(2)
        clf()
        imshow(self.pdata.T, cmap='hot', origin='lower',
                norm=matplotlib.colors.LogNorm(vmin=.5, vmax=self.pdata.max()),
                extent=[-0.5, 1.5, 0, 1.0])
        gca().set_facecolor('black')
        gca().set_aspect('auto')
        colorbar()
        draw()

    def shutdown(self):
        self.sock.sendall(self.exitMsg)
        self.server.kill()

    def __del__(self):
        self.shutdown()

d = Display(1024, 1024, 512)
#d.refresh(-1, 2, -4, 8)
d.refresh(-1, 2, 0, 1.2)
xlabel(r"$H'$")
ylabel(r"$\tau$")
grid()
