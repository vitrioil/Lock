import wave
import time
import socket
import pyaudio
import numpy as np
from queue import Queue
from model import TModel
from sample import Sample
from threading import Thread

class Listen:
    host = ""
    port = 30002
    def __init__(self,p,stream,form=pyaudio.paInt16,chunk=1024*2,channels=2,rate=44100,shift_bytes=275):
        self.p = p
        self.stream = stream
        self.form = form
        self.chunk = chunk
        self.channels = channels
        self.rate = rate
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((self.host, self.port))
        self.s.listen(1)
        self.q = Queue()
        self.frames = []
        self.saved = False
        self.shift_bytes = shift_bytes
        self.closed = False
        self._connect()

    def _connect(self):
        print("Waiting for a connection request")
        self.con,self.addr = self.s.accept()
        print(f"Connected to {self.addr}")

    def tcpStream(self):
        count = 0
        while not self.closed:
            try:
                soundData = self.con.recv(self.chunk)
                self.con.send(f"Got{count}!".encode())
                if len(soundData)>0:
                    self.q.put(soundData)
                    self.frames.append(soundData)
                #print(f"Got a packet:{count}!")
                count += 1
            except ConnectionResetError as e:
                print(str(e))
                self._close()

            except Exception as e:
                print("Exception in tcpStream()",str(e))
                self._close()
                self.save()

    
    def _close(self):
        print("Closing the socket")
        self.closed = True
        self.s.close()

    def play(self):
        while True:
            if not self.q.empty():
                self.stream.write(self.q.get(), self.chunk)

    def save(self,remove=True,filename="test.wav"):
        if self.saved or len(self.frames) == 0:
            return
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.form))
            wf.setframerate(self.rate)
            print("Saving bytes {}".format(len(self.frames)),end="\r")
            wf.writeframes(b''.join(self.frames))
        if remove:
            self.frames = self.frames[self.shift_bytes:]
        print(f"Frames length is now {len(self.frames)}")
        self.saved=True


    def reset_save(self):
        self.saved = False
        
    def stop(self):
        '''
        To stop the server with ^C
        To do: Improvise
        '''
        last_time = time.time()
        while True:
            try:
                if time.time() - last_time > 10:
                    last_time = time.time()
                    self.reset_save()
            except KeyboardInterrupt as e:
                print("Exception in stop()",str(e))
                self.close()
                self.save()
                self.s.close()

class Evaluate:
    def __init__(self,listen_object):
        self.sample = Sample(None,None,None,saved=True)
        Tx,freq = 5511,101 
        self.model = TModel(Tx,freq,*self.sample.load_dataset())
        self.model.load_model()
        self.listen_object = listen_object

    def continuously_analyze(self):
        while len(self.listen_object.frames) != 0:
            predictions = self.model.detect_triggerword("test.wav")
            if predictions is None:
                self.listen_object.reset_save()
                                                # if client is disconnected start removing frames
                self.listen_object.save(remove=self.listen_object.closed)
            else:
                print("Got enough data resetting save",end="\r")
                self.listen_object.reset_save()
                self.listen_object.save()
                predictions = np.squeeze(predictions)
                continuous = 0
                print("Total ones",np.sum(predictions))
                for i in predictions:
                    '''
                    To do hyperparameter this to 50/75 contiguous ones to predict
                    '''
                    if i == 1:
                        continuous += 1
                    else:
                        continuous = 0
                    if continuous >= 50:
                        print("Did you say lock?",end="\r")
                        continuous = 0
            time.sleep(2)

def start_threads(l,e):
    tTCP = Thread(target = l.tcpStream)
    tPlay = Thread(target = l.play)
    tStop = Thread(target=l.stop)
    tAnalyze = Thread(target=e.continuously_analyze)
    tTCP.start()
    tPlay.start()
    tStop.start()
    tAnalyze.start()

if __name__ == "__main__":
    
    form,channels,rate,chunk = pyaudio.paInt16,2,44100,1024*2
    p = pyaudio.PyAudio()

    stream = p.open(format=form,
                    channels = channels,
                    rate = rate,
                    output = True,
                    frames_per_buffer = chunk,
                    )
    l = Listen(p,stream,form,chunk,channels,rate)
    e = Evaluate(l)
    start_threads(l,e)