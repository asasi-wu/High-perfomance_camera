from common_func import isPosiInFrame, gen_area
import os
import datetime
import pickle


class Wizvideo:
    def __init__(self, options):
        self.options = options
        self.areas = {}
        if os.path.exists('dic_pickle') is True:
            with open('dic_pickle', 'rb') as f:
                self.areas = pickle.load(f)
                print(self.areas)
        else:
            open('dic_pickle', 'rb')

    def add_areas(self, cap_url, cap_name, cap_id, area_type, points, area_id, area_name, ):
        if self.areas.get(cap_url) is None:
            self.areas[cap_url] = []
        area_desc = {'cap_url': cap_url,
                     'cap_name': cap_name,
                     'cap_id': cap_id,
                     'area_type': area_type,
                     'points': gen_area(points, self.frame_size[0], self.frame_size[1]),
                     'area_name': area_name,
                     'area_id': area_id}
        self.areas[cap_url].append(area_desc)
        with open('dic_pickle', 'wb') as f:
            pickle.dump(self.areas, f)

    def delete_areas(self, cap_url, area_name):
        if self.areas.get(cap_url) is not None:
            count = 0
            for i in self.areas[cap_url]:
                count += 1
                if i['area_name'] == area_name:
                    temp = count
                    self.areas[cap_url].pop(temp - 1)
                    print(self.areas)
                    with open('dic_pickle', 'wb') as f:
                        pickle.dump(self.areas, f)
                    break

    def clear_areas(self):
        self.areas = {}
        with open('dic_pickle', 'wb') as f:
            pickle.dump(self.areas, f)

    def get_areas(self):
        list_points = []
        for i in self.areas.values():
            for j in i:
                list_points.append(j)
        return list_points

    def show_config(self):
        print(self.options)
        print(self.areas)
        print(self.frame_size)


if __name__ =='__main__':
    w = Wizvideo()
    '''def run_detect(self):
        url = []
        for i in self.areas.keys():
            url.append(i)

        list_points = []
        for i in self.areas.values():
            for j in i:
                list_points.append(j)

        q = mp.Queue(self.sizeofQueue)
        pool = redis.ConnectionPool(host='localhost', port=6379, db=1)
        r = redis.Redis(connection_pool=pool)
        count = 0
        tfnet = TFNet(self.options)
        p = mp.Process(target=foo, args=(q, url, len(url),))
        p.start()
        color = [(255, 0, 0) for _ in range(10)]
        while True:
            try:
                frame_tmp = q.get_nowait()
                name = frame_tmp[1]
                frame = frame_tmp[0]
                print('get name ', name)
                # frame = q.get_nowait()
                results = tfnet.return_predict(frame)
                # print('buffer queue size ', q.qsize())
                for result in results:
                    label = result['label']
                    if label == 'person':
                        tl = (result['topleft']['x'], result['topleft']['y'])
                        br = (result['bottomright']['x'], result['bottomright']['y'])
                        print(tl, br, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                        r.set(str(count),
                              name + str(tl) + str(br) + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        count += 1

                        pwd = os.getcwd()
                        os.chdir(pwd + '/detect_people/test4')
                        cv2.imwrite(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'|' + str(tl) + '|' + str(br) +
                                    '.jpg', frame)
                        os.chdir(pwd)
            except:
                pass

if len(self.areas) != 0:
            for i in list_points:
                if isPosiInFrame(core, i['points']) is True:
                    print('warning')
                    print(i['area_id'])'''
