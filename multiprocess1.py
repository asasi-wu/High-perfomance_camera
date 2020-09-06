from queue import LifoQueue             #加载一个栈
from darkflow.net.build import TFNet    #加载深度学习模型的库

import multiprocessing as mp            #多进程的库
import threading                        #多线程的库
#import redis                            #python控制redis的库
import cv2                              #opencv
import numpy as np
import os
import datetime
import time
import socket
import json
from common_func import isPosiInFrame, gen_area
#栈，用来获取最新帧
def lifo_put(queue, cap, name,index):
    #print(cap)
    time.sleep(int(index)/10)            #这句话是要同一个进程内的不同线程分时间启动，以防造成死锁
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    ret=True
    count = 0
    start_time=time.time()
    while ret:
        count += 1
        ret, frame = cap.read()
        #cv2.imshow('frame', frame)
        

        if count == 100:
       
            
            count = 0
            #print(name,'  detectet is alive',name, threading.currentThread())
            #print("frame_size{}{}".format(frame.shape[0],frame.shape[1]))
            
            if type(frame) == type(np.array([1, 2, 3])) and frame.shape == (1080, 1920, 3):
                queue.put([frame, name])
                if queue.qsize() >= 3:
                    queue.clear()           #这里修改了一下queue库里面的源码,　clear()函数不会真正的清空栈，而会留下栈顶元素
            print('FPS {:.1f}'.format(100 / (time.time() - start_time)))

'''
运动检测函数
'''
def lifo_get(queue, q):
    knnSubtractor = cv2.createBackgroundSubtractorKNN(100, 400, True)
    while True:
        frame_tmp = queue.get()
        name = frame_tmp[1]
        frame = frame_tmp[0]                        #frame是一个1920×1080的矩阵
        #frame = queue.get()
        #print('get name ',name)
        res = frame
        frame = cv2.resize(frame, (320, 200))       #resize减小计算量，frama变成320×200的矩阵
        knnMask = knnSubtractor.apply(frame)        #运动检测算法, 一直不动的地方矩阵值为0,　运动的地方为255
        fg_mask = cv2.GaussianBlur(knnMask, (33, 33), 0)            #高斯模糊, 去噪,　就是对一个像素点的周围像素点取平均, (33,33)是取平均的半径,　之后矩阵的值在0-255之间
        ret, th = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)     #把数值为0-255的矩阵按按某个阈值分成两份, 小于阈值的为0, 大于阈值的为255
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)    #发散, 其实可以省略这一步
        img = np.asarray(dilated)
        check_zores = np.where(img == 255)          #下面开始检验矩阵是否为全零矩阵, 不是全零矩阵的话, 该帧将被放入阻塞队列

        if check_zores[0].shape[0] == 0:
            pass
            #print('this is a zeros matrix')
        else:
            #print('this is not a zeros matrix')
            if q.full() is False:
                q.put_nowait([res, name])


# 这个函数就是获得网络摄像头地址的,　然后开启线程
def foo(q, url, num, location_name):
    #url是一个数组，包含了所有的url
    #num是一个int型整数，只开启的线程个数，同时控制帧率
    cap =[]
    thread = []
    url_queue = []
    thread_knn = []
    for i in range(num):
        cap.append(cv2.VideoCapture(url[i]))

    for i in range(num):
        url_queue.append(LifoQueue(maxsize=500))

    for i in range(num):
        thread.append(threading.Thread(target=lifo_put, args=(url_queue[i], cap[i], location_name[i],i)))
        thread_knn.append(threading.Thread(target=lifo_get, args=(url_queue[i], q, )))
        thread[i].start()
        thread_knn[i].start()
    for thread in thread:
        thread.join()
    for thread in thread_knn:
        thread.join()


def add_area( area_type, points, area_name, area_id,frame_size0, frame_size1):
    content = {'area_type': area_type,
                'points': gen_area(points, frame_size0, frame_size1),
                'area_name': area_name,
                'area_id': area_id}
    #self.areas.append(content)

    #print('Now, we have ', len(self.areas), 'areas')
    #print(self.areas)
    return content



if __name__ == "__main__":
    #mp.set_start_method('spawn')
    # 配置网络摄像头的地址
    url = ['/home/asasi/Desktop/obj_detect/test.mp4',
            '/home/asasi/Desktop/obj_detect/test1.mp4'
           ]
    #area=add_area(area_name='area11', area_type='warnning', points=[(0.55, 0.38), (0.68, 0.4), (0.5, 0.8), (0, 0.75)], area_id=1,frame_size0=frame.shpe[0],frame_size1=frame.shape[1])
    

    # 设置阻塞队列的缓存大小
    q = mp.Queue(100)

    count = 0
    # 要加载的模型的地址,　以及参数
    options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'threshold': 0.2,
        'gpu': 1.0
    }

    # 加载深度学习模型
    tfnet = TFNet(options)
    # 这里的名字要对应上面的地址
    location_name = ['outside', 'home']
    locations_id={'outside':1,'home':2}
    locations_points={'outside':[(0., 0.38), (0., 0.4), (0.5, 0.8), (0, 0.75)],'home':[(0.55, 0.38), (0.68, 0.4), (0.5, 0.8), (0, 0.75)]}
    # 开启一个jin程,　并传入相关参数,　这个线程就是负责从摄像头拿到数据, 并做运动分析
   
    p = mp.Process(target=foo, args=(q, url, len(url),location_name))
    p.start()
    color = [(255, 0, 0) for _ in range(10)]
    while True:
        try:

            # 主线程, 从阻塞队列中获得数据帧，队列为空的话跳过跳过此循环
            frame_tmp = q.get_nowait()
            name = frame_tmp[1]
            frame = frame_tmp[0]
            
            #print('get name ', name)
            #frame = q.get_nowait()
            # 预测结果
            results = tfnet.return_predict(frame)
            #cv2.imshow('frame', frame)
            #print('buffer queue size ', q.qsize())
            for result in results:
                label = result['label']
                if label == 'person':
                    tl = (result['topleft']['x'], result['topleft']['y'])
                    br = (result['bottomright']['x'], result['bottomright']['y'])
                    #print(tl, br, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    core = [int((tl[0]+br[0])/2), br[1]]
                    area=add_area(area_name=name, area_type='warnning', points=locations_points[name], area_id=locations_id[name],frame_size0=frame.shape[0],frame_size1=frame.shape[1])
                    
                    a = np.array([area['points']], dtype=np.int32)
                    print(a)
                    # Display the resulting frame
                    cv2.polylines(frame, a, 1, 255)
                    if isPosiInFrame(core, area['points']) is True:
                        print("true")
                        cv2.circle(frame, (int((tl[0] + br[0]) /2), int(br[1]/0.9)), 63, (0, 0, 255), -1)

                        

                        pwd = os.getcwd()
                        os.chdir(pwd + '/detect_people/test4')
                        cv2.imwrite(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'|' + str(tl) + '|' + str(br) +
                                '.jpg', frame)
                        os.chdir(pwd)

                    """
                    phone = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    phone.connect(('127.0.0.1', 10001))  # 拨通电话
                    sendData = {'tl': tl, 'br': br, 'name': name,
                                'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    message = json.dumps(sendData)
                    phone.send(message.encode('utf-8'))  # 发消息
                    phone.close()
                    """


        except:
            pass
