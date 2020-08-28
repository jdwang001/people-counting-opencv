# import the necessary packages
from threading import Thread
import sys
import cv2
from wp.log import Log
import imutils
import time

log = Log(__name__).getlog()

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        # self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)
        self.stopped = False
        # self.queuesize = queueSize
        # self.path = path

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        self.framecount = 0
        self.stopthread = False

    def start(self,width='',height=''):
        log.info("Get video begain")
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=(width,height))
        t.daemon = True
        t.start()
        return self

    def update(self,width,height):
        log.debug("begain get video stream")
        # keep looping infinitely
        try:
            while True:
                if self.stopthread:
                    log.debug("Exit thread,reconnect")
                    return
                # if the thread indicator variable is set, stop the
                # thread
                if self.stopped:
                    log.debug("exit the thread.")
                    return

                self.framecount += 1

                #if totalFrames % args["skip_frames"] == 0:

                # otherwise, ensure the queue has room in it
                if not self.Q.full():
                    # read the next frame from the file
                    frame = self.stream.read()

                    # if the `grabbed` boolean is `False`, then we have
                    # reached the end of the video file
                    if not frame:
                        self.stop()
                        log.debug("Can't get frame ")
                        continue

                    # add the frame to the queue
                    #resizeframe = cv2.resize(frame, (width, height)
                    #self.Q.put(resizeframe)
                    #resizeframe = cv2.resize(frame, (width, height)
                    #if self.framecount % 6 == 0:
                    #    self.Q.put(frame)
                    self.Q.put(frame)
                else:
                    log.info("frame queue is full")
                    time.sleep(1)

        except Exception as exc:
            log.error('FileVideoStream exception %s ',exc)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def release(self):
        self.stopthread = True
        self.stream.release()