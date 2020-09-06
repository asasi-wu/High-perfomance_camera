def gen_area(points = None, axis_y = 720, axis_x = 1280):
    a = [[i * axis_x, j * axis_y] for (i, j) in points]
    return a


def isRayIntersect(point, begin, end):
    if begin[1] == end[1]:
        return False
    if min(begin[1], end[1]) >= point[1]:
        return False
    if max(begin[1], end[1]) <= point[1]:
        return False
    if max(begin[0], end[0]) <= point[0]:
        return False
    intersecter = end[0] - ((end[0] - begin[0]) / (end[1] - begin[1])) * (end[1] - point[1])
    if intersecter <= point[0]:
        return False

    return True

def isPosiInFrame(point, frames):
    nums = 0
    for i in range(0,len(frames)):
        if i < len(frames)-1:
            if isRayIntersect(point,frames[i], frames[i+1]) == True:
                nums = nums + 1
        else:
            if isRayIntersect(point,frames[i], frames[0]) == True:
                nums = nums + 1
    if nums%2 == 0:
        return False
    if nums%2 != 0:
        return True