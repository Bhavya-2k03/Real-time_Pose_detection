from tkinter import Frame
import cv2
import mediapipe as mp
import time

class posedetector():
    def __init__(self, mode=False, complexity=1, smooth=True, segmentation = False, smooth_segmentation = True ,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)

    def findPose(self, frame, draw=True):
        frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(frame_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(frame, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return frame

    def findPosition(self,frame,draw=True):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(frame,(cx,cy),5,(255,0,0),cv2.FILLED)
        return lmList

def main():
    cam =cv2.VideoCapture(0)
    detector=posedetector()
    ptime=0
    while True:
         reg,frame=cam.read()
         frame=detector.findPose(frame)
         lmList=detector.findPosition(frame)
         ctime=time.time()
         fps=1/(ctime-ptime)
         ptime=ctime
         print(lmList)
         cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0),3,)
         cv2.imshow("My_Image",frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

if __name__ =="__main__":
    main()