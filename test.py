

def delete_areas(areas, cap_name, area_name):
    if areas.get(cap_name) is not None:
        count = 0
        for i in areas[cap_name]:
            count += 1
            if i['area_name'] == area_name:
                temp = count
                areas[cap_name].pop(temp-1)
                print(areas)
                break
        return areas


if __name__ == '__main__':
    list1 = {'rtsp://admin:zhihe000@192.168.1.181:554/h264/ch1/main/av_stream': [{'area_type': 'warning', 'points': [(0.425, 0.3), (0.625, 0.3), (0.725, 0.6), (0.325, 0.6)], 'area_name': 'area1', 'area_id': 1}],
             'rtsp://admin:zhihe000@192.168.1.182:554/h264/ch1/main/av_stream': [{'area_type': 'error', 'points': [(0.425, 0.3), (0.625, 0.3), (0.725, 0.6), (0.325, 0.6)], 'area_name': 'area1', 'area_id': 1}, {'area_type': 'error', 'points': [(0.425, 0.3), (0.625, 0.3), (0.725, 0.6), (0.325, 0.6)], 'area_name': 'area2', 'area_id': 2},
                                                                                 {'area_type': 'error', 'points': [(0.425, 0.3), (0.625, 0.3), (0.725, 0.6), (0.325, 0.6)], 'area_name': 'area3', 'area_id': 3}, {'area_type': 'error', 'points': [(0.425, 0.3), (0.625, 0.3), (0.725, 0.6), (0.325, 0.6)], 'area_name': 'area4', 'area_id': 4}]
             }
    for i in list1.values():
        print(i)