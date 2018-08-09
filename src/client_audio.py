import pyaudio
import socket
from threading import Thread

frames = []

def tcpStream():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    
    tcp.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    tcp.connect(("192.168.0.108",30002))
    while True:
        if len(frames) > 0:
            try:
                tcp.send(frames.pop(0))
                s = tcp.recv(1024)
#                print(f"Received {s}",end="\r")
            except KeyboardInterrupt as e:  
                tcp.close()

def record(stream, CHUNK):    
    while True:
        frames.append(stream.read(CHUNK))

if __name__ == "__main__":
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK,
                    )

    Tr = Thread(target = record, args = (stream, CHUNK,))
    Ts = Thread(target = tcpStream)
    Tr.start()
    Ts.start()
