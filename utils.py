def read_classes(path:str):
    with open(path, "r") as f:
        classes = f.readlines()
    labels = [_cls.strip() for _cls in classes]
    return labels

def unconvert(class_id:int, width:int, height:int, x:float, y:float, w:float, h:float) -> tuple:
    xmax = int((x*width) + (w * width)/2.0)
    xmin = int((x*width) - (w * width)/2.0)
    ymax = int((y*height) + (h * height)/2.0)
    ymin = int((y*height) - (h * height)/2.0)
    class_id = int(class_id)
    return (class_id, xmin, xmax, ymin, ymax)

def read_ann(filename:str) -> list:
    __xml = []
    with open(filename, "r") as f:
        data = f.readlines()
    for dt in data:
        class_id, x, y, w, h = dt.split()
        box = x, y, w, h
        x, y, w, h = list(map(float, box))
        __xml.append([class_id, x, y, w, h])
    return __xml

def convert(width:int, height:int, bbox:list) -> tuple:
    dw = 1./width
    dh = 1./height
    x = (bbox[0] + bbox[1])/2.0 - 1
    y = (bbox[2] + bbox[3])/2.0 - 1
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
