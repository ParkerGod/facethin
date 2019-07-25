# -*- coding:UTF-8 -*-
import numpy as np
import cv2
import math
import facethin
from imutils import face_utils, resize
try:
    from dlib import get_frontal_face_detector, shape_predictor
except ImportError:
    raise



predictor = shape_predictor(
            "shape_predictor_68_face_landmarks.dat")  # 面部分析器
detector = get_frontal_face_detector()  # 面部识别器


def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray,0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x,p.y] for p in predictor(img_gray,rects[i]).parts()])
        land_marks.append(land_marks_node)

    return land_marks

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    #cap.set(3,640) #设置分辨率
    #cap.set(4,480)
    
    fps =cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    ret , frame = cap.read()
    while(ret):
        ret , frame = cap.read()
        landmarks = landmark_dec_dlib_fun(frame)
        #如果未检测到人脸关键点，就不进行瘦脸
        if len(landmarks) == 0:
            print("没有脸")
        else:
            for landmarks_node in landmarks:
                left_landmark= landmarks_node[4]
                left_landmark_down=landmarks_node[8]
        
                right_landmark = landmarks_node[11]
                right_landmark_down = landmarks_node[15]
        
                endPt = landmarks_node[30]
        
        
                #计算第4个点到第6个点的距离作为瘦脸距离
                r_left=math.sqrt((left_landmark[0,0]-left_landmark_down[0,0])*(left_landmark[0,0]-left_landmark_down[0,0])+
                                 (left_landmark[0,1] - left_landmark_down[0,1]) * (left_landmark[0,1] - left_landmark_down[0, 1]))
        
                # 计算第14个点到第16个点的距离作为瘦脸距离
                r_right=math.sqrt((right_landmark[0,0]-right_landmark_down[0,0])*(right_landmark[0,0]-right_landmark_down[0,0])+
                                (right_landmark[0,1] -right_landmark_down[0,1]) * (right_landmark[0,1] -right_landmark_down[0, 1]))
            hi = facethin.face_data(frame,left_landmark[0,0],left_landmark[0,1],endPt[0,0],endPt[0,1],r_left,right_landmark[0,0],right_landmark[0,1],r_right,1920,1080)
            print(hi)
            print("已改")
            cv2.imwrite("frame.jpg",frame)
        cv2.imshow('frame',frame)
        
        if cv2.waitKey(1) &0xFF ==ord('q'):  #按q键退出
        	break
    #when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()