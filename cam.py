import time
import cv2
import threading as thr
import logging
import argparse
import sys
import queue
import keyboard
import numpy as np

logname = "./log/logs.txt"
logging.basicConfig(filename=logname,
                    filemode='a+',
                    format='TIME: %(asctime)s,%(msecs)d || %(name)s || %(levelname)s || %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

delays = [0.01, 0.1, 1]

event = thr.Event()
event.clear()

qus = []
for i in range(len(delays) + 1):
    qus.append(queue.Queue())

def exiter():
    event.set()

def whileSensor(sensorx, index):
    while(1):
        qus[index].put(sensorx.get())
        if event.is_set():
            break

class Sensor:
    """Sensor - some quazi-abstract class"""

    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")

class SensorX(Sensor):
    """Sensor X - just timer"""

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0
        logging.info("SensorX with delay " + str(delay) + " initialized")

    def get(self) -> int:
        time.sleep(self._delay)
        self._data = self._data + 1
        return self._data

class SensorCam(Sensor):
    """Sensor Cam - just camera"""

    def __init__(self, cam_path : str, resolution):
        self._data = 0
        self._cam_path = cam_path

        try:
            # self.cam = cv2.VideoCapture(cam_path, cv2.CAP_DSHOW)
            self.cam = cv2.VideoCapture(int(cam_path), cv2.CAP_DSHOW)   # for Windows, just int
        except:
            logging.error("Bad cam name: " + self._cam_path)
            exiter()
        if not self.cam.isOpened():
            logging.error("No camera with such name: " + self._cam_path)
            exiter()
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
        logging.info(str(cam_path) + " camera initialized")

    def get(self):
        check, frame = self.cam.read()
        if not check:
            logging.critical("No signal from camera: " + self._cam_path)
            
        return frame
    
    def __del__(self):
        self.cam.release()

class WindowImage:
    def __init__(self, winName):
        self.winName = winName
        logging.info(winName + " window initialized")

    def show(self, frame):
        self.windowImage = cv2.imshow(self.winName, frame)
    
    def __del__(self):
        cv2.destroyWindow(self.winName)

class OurWindowImage(WindowImage):
    def subShow(self, results):
        frame = results[len(results) - 1]
        frame_copy = frame.copy()
        for i in range(len(results) - 1):
            cv2.putText(frame_copy, str(results[i]), (50, (i + 1)*50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        self.show(frame_copy)

if __name__ == "__main__":
    logging.info("Program start")
    parser = argparse.ArgumentParser()
    parser.add_argument('camName', type = str)
    parser.add_argument('resolution', type = str)
    parser.add_argument('frequency', type = float)

    args = parser.parse_args()
    camName = args.camName
    resolution = args.resolution
    frequency = args.frequency

    fir = []
    sec = []
    for i in range(len(resolution)):
        if not resolution[i].isdigit():
            fir = resolution[0:i]
            sec = resolution[(i+1)::]
            break
    
    if fir == [] or sec == [] or not fir.isnumeric() or not sec.isnumeric():
        logging.error("Bad resolution: " + resolution)
        sys.exit()
    resolution = [int(fir), int(sec)]

    # freq_index = -1
    # for i in range(len(delays)):
    #     if delays[i] == frequency:
    #         freq_index = i
    #         break
    # if freq_index == -1:
    #     logging.error("Bad frequency: " + str(frequency))
    #     sys.exit()

    if frequency < min(delays):
        logging.error("Too small frequency: " + str(frequency))
        sys.exit()

    threads = []
    for i in range(len(delays)):
        threads.append(thr.Thread(target=whileSensor, args=(SensorX(delays[i]), i), daemon=True))
    
    sensorCam = SensorCam(camName, resolution)
    threads.append(thr.Thread(target=whileSensor, args=(sensorCam, len(delays)), daemon=True))

    windowImage = OurWindowImage("cam")

    results = []
    for i in range(len(threads)):
        threads[i].start()
        results.append(0)
    
    num_prev_frames = 0
    need_new_frame = 0
    buffer_cam = np.uint8([[[1, 1, 1] for i in range(resolution[1])] for j in range(resolution[0])])
    cv2.integral(buffer_cam)
    freqs = [i/min(delays) for i in delays]
    frequency = frequency/min(delays)
    while not event.is_set():
        try:
            if keyboard.is_pressed('q'):  # if key 'q' is pressed
                event.set()
                logging.info("Manual closing by pressing 'q'")
                break
        except:
            pass
        
        for i in range(len(qus)):
            if not qus[i].empty():
                if len(qus) - 1 == i:
                    buffer_cam = qus[i].get()

                else:
                    results[i] = qus[i].get()
                    if freqs[i] * results[i] / frequency > num_prev_frames + 1:
                        results[len(qus) - 1] = buffer_cam
                        num_prev_frames = num_prev_frames + 1
        windowImage.subShow(results)
        cv2.waitKey(1)

    for threadx in threads:
        threadx.join()
    del sensorCam
    del windowImage
    logging.info("Safely closed")
