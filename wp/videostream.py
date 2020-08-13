# import the necessary packages
from threading import Thread
import sys
import cv2

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
	#self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
	#self.stream.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self,width='',height=''):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=(width,height))
        t.daemon = True
        t.start()
        return self

    def update(self,width,height):
        # keep looping infinitely
        try:
            while True:
                # if the thread indicator variable is set, stop the
                # thread
                if self.stopped:
                    return

                # otherwise, ensure the queue has room in it
                if not self.Q.full():
                    # read the next frame from the file
                    frame = self.stream.read()

                    # if the `grabbed` boolean is `False`, then we have
                    # reached the end of the video file
                    if not frame:
                        self.stop()

                    # add the frame to the queue
                    #resizeframe = cv2.resize(frame, (width, height)
                    #self.Q.put(resizeframe)
                    #resizeframe = cv2.resize(frame, (width, height)
                    self.Q.put(frame)
                else:
                    print("video frame is full")
        except Exception as exc:
            print ('FileVideoStream exception ',exc)

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
        self.stream.release()