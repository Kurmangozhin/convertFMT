from glob import glob
from tqdm import tqdm
import os, cv2, argparse
from pascal_voc_writer import Writer
from multiprocessing import Pool
from utils import read_classes, unconvert, read_ann, convert
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser("yolo_voc_format")
parser.add_argument("-path", "--path_dir", type = str, default = "", help="", required = True)
parser.add_argument('--yolo_voc', action ='store_true', help = 'yolo to voc (*.xml)', required = False)
parser.add_argument('--voc_yolo', action ='store_true', help = 'voc to yolo (*.txt)', required = False)

__CLASSES = read_classes("classes.txt")

def __yolo_to_voc(filename:str) -> None:
    img_src = cv2.imread(filename)
    height, width, _ = img_src.shape
    __xmlvoc = read_ann(filename.replace(".jpg", ".txt"))
    writer = Writer(filename, width, height)
    for class_id, x, y, w, h in __xmlvoc:
        class_id, xmin, xmax, ymin, ymax = unconvert(class_id, width, height, x, y, w, h)
        writer.addObject(__CLASSES[class_id], xmin, ymin, xmax, ymax)
    writer.save(filename.replace(".jpg", ".xml"))


def __voc_to_yolo(filename:str) -> None:
    tree = ET.parse(filename.replace(".jpg", ".xml"))
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    with open(filename.replace(".jpg", ".txt"), "w") as f:
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in __CLASSES or int(difficult)==1: continue
            cls_id = __CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            bbox = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bbox = convert(width, height, bbox)
            f.write(str(cls_id) + " " + " ".join([str(a) for a in bbox]) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    nmcpu = os.cpu_count()
    filesJpg = sorted(glob(f"{args.path_dir}/*.jpg"))
    msg = f"[INFO]: No files (*.jpg) in dir: {args.path_dir}"
    assert len(filesJpg)!= 0, msg
    print(f"\n[INFO]: {args.path_dir = }")

    if args.yolo_voc:
        with Pool(processes=nmcpu) as p:
            max_ = len(filesJpg)
            with tqdm(total=max_, colour="blue", desc="[INFO]: Yolo -> Pascal XML") as pbar:
                for _ in p.imap_unordered(__yolo_to_voc, filesJpg):
                    pbar.update()

    if args.voc_yolo:
        with Pool(processes=nmcpu) as p:
            max_ = len(filesJpg)
            with tqdm(total=max_, colour="yellow", desc="[INFO]: Pascal XML -> Yolo") as pbar:
                for _ in p.imap_unordered(__voc_to_yolo, filesJpg):
                    pbar.update()

    if not args.yolo_voc and not args.voc_yolo:
        print("\n[INFO]: Please arguments cmd --yolo_voc | --voc_yolo")
  
    """
    cmd:
        python3 main.py --path dir --yolo_voc // *.txt --> *.xml
        python3 main.py --path dir --voc_yolo // *.xml --> *.txt
    """



