from darkflow.net.build import TFNet
from common_func import isPosiInFrame, gen_area
import cv2
import numpy as np
import time
import os
import datetime
import multiprocessing as mp
from multiprocess1 import foo, lifo_put, lifo_get
#import redis


class Wizvideo:
    def __init__(self, video_url):
        self.options = {}
        self.video_url = video_url
        self.color = (255, 0, 0)
        self.areas = []
        self.api_contral= None
        cap = cv2.VideoCapture(self.video_url)
        self.frame_size = ()
        while cap.isOpened():
            
            _, f1 = cap.read()
            self.frame_size = f1.shape
            break
        print('the size of frame is: ', self.frame_size)
        cap.release()

    def config(self, user_options, color=(255, 0, 0)):
        self.options = user_options
        self.color = color

    def show_areas(self, area_id, time = 100):
            #points
            #[[(0.01, 0.01), (0.5, 0.01), (0.5, 0.7), (0.4, 0.7), (0.01, 0.5)],
            # [(0.01, 0.01), (0.5, 0.01), (0.5, 0.7), (0.4, 0.7), (0.01, 0.5)]]
        num = 0
        cap = cv2.VideoCapture(self.video_url)
        while cap.isOpened():
            ret, frame = cap.read()
            if len(self.areas) == 0:
                cv2.putText(frame, 'there is not available area:',
                            (int(self.frame_size[0] / 3), int(self.frame_size[1] / 3)), cv2.FONT_HERSHEY_SIMPLEX,
                            self.frame_size[1]/800,
                            (0, 0, 255), int(self.frame_size[1]/500))
            else:
                for i in range(0, len(self.areas)):
                    if self.areas[i]['area_id']==area_id:
                        a = np.array([self.areas[i]['points']], dtype=np.int32)
                        cv2.polylines(frame, a, 1, 255)
                        cv2.putText(frame, 'area_name:' + self.areas[i]['area_name'],
                                    (10 + int(self.areas[i]['points'][0][0]), 25 + int(self.areas[i]['points'][0][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    self.frame_size[1] / 1280,
                                    (0, 0, 255), 2)
                        cv2.putText(frame, 'area_type:' + self.areas[i]['area_type'],
                                    (10 + int(self.areas[i]['points'][0][0]), 51 + int(self.areas[i]['points'][0][1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    self.frame_size[1] / 1280,
                                    (0, 0, 255), 2)

            cv2.imshow('frame', frame)
            num = num + 1
            if cv2.waitKey(1) & 0xFF == ord('q') or num >= time:
                break
        cap.release()
        cv2.destroyAllWindows()

    def add_area(self, area_type, points, area_name, area_id):
        content = {'area_type': area_type,
                   'points': gen_area(points, self.frame_size[0], self.frame_size[1]),
                   'area_name': area_name,
                   'area_id': area_id}
        self.areas.append(content)

        print('Now, we have ', len(self.areas), 'areas')
        print(self.areas)

    def run_dectect(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        tfnet = TFNet(self.options)
        colors = [self.color for _ in range(10)]
        capture = cv2.VideoCapture(self.video_url)
        count_frame=0
        stime = time.time()
        while True:
            
            
            ret, frame = capture.read()
        
            if ret:
                num_person = 0
                if count_frame==100:
                    
                    count_frame=0
                    
                    print('FPS {:.1f}'.format(100 / (time.time() - stime)))
                    print(time.time() - stime)
                else:
                    print(count_frame)
                    count_frame=count_frame+1
                # 视频上的文字
                if len(self.areas) == 0:
                    cv2.putText(frame, 'there is not available area:',
                                (int(self.frame_size[0] / 3), int(self.frame_size[1] / 3)), cv2.FONT_HERSHEY_SIMPLEX,
                                self.frame_size[1] / 800,
                                (0, 0, 255), int(self.frame_size[1] / 500))
                else:
                    for i in range(0, len(self.areas)):
                        a = np.array([self.areas[i]['points']], dtype=np.int32)

                        # Display the resulting frame
                        cv2.polylines(frame, a, 1, 255)
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        cv2.putText(frame, 'area_name:' + self.areas[i]['area_name'],
                                    (10 + int(self.areas[i]['points'][0][0]),
                                     25 + int(self.areas[i]['points'][0][1])), font, 1,
                                    (0, 0, 255), 2)
                        cv2.putText(frame, 'area_type:' + self.areas[i]['area_type'],
                                    (10 + int(self.areas[i]['points'][0][0]),
                                     49 + int(self.areas[i]['points'][0][1])), font, 1,
                                    (0, 0, 255), 2)
                #开始预测，预测bbox的位置以及类别
                if True:
                    results = tfnet.return_predict(frame)
                    for color, result in zip(colors, results):
                        label = result['label']
                        if label == 'person':
                            tl = (result['topleft']['x'], result['topleft']['y'])
                            br = (result['bottomright']['x'], result['bottomright']['y'])
                            print('tl:', tl)
                            print('br:', br)
                            core = [int((tl[0]+br[0])/2), br[1]]
                            frame = cv2.rectangle(frame, tl, br, color, 3)
                            # label = result['label']
                            if len(self.areas) != 0:
                                for area in self.areas:
                                    if isPosiInFrame(core, area['points']) is True:
                                        cv2.circle(frame, (int((tl[0] + br[0]) /2), int(br[1]/0.9)), 63, (0, 0, 255), -1)

                                        num_person = num_person + 1

                                        pwd = os.getcwd()
                                        os.chdir(pwd + '/detect_people')
                                        cv2.imwrite(
                                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(
                                                num_person) + ".jpg", frame)
                                        os.chdir(pwd)

                            confidence = result['confidence']
                            text = '{}: {:.0f}%'.format(label, confidence * 100)
                            frame = cv2.putText(
                                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
                    cv2.putText(frame, 'num_person:' + str(num_person),
                                (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                                (255, 255, 255), 2)
                    cv2.imshow('frame', frame)
                    
                    # print(car_num)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()

