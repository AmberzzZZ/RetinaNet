from lxml.etree import Element, SubElement, tostring
import cv2
import os


def yolo2voc(txt_file, folder_name, png_dir, xml_dir):

    file_name = txt_file.split('/')[-1].split('.txt')[0]

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = folder_name
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = file_name + '.png'
    node_filename = SubElement(node_root, 'path')
    node_filename.text = os.path.join(png_dir, file_name+'.png')

    img = cv2.imread(os.path.join(png_dir, file_name+'.png'), 0)
    shape = img.shape

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img.shape[1])
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img.shape[0])
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '1'

    name_list = ['gzzs', 'xmsjj', 'hzrdgh']

    f = open(txt_file, 'r')
    for line in f.readlines():
        if len(line) < 3:
            continue
        clsid, xc, yc, w, h = map(float, line.strip().split(' '))
        clsid = int(clsid)
        x_min = int(shape[1] * (xc - w/2.))
        x_max = int(shape[1] * (xc + w/2.))
        y_min = int(shape[0] * (yc - h/2.))
        y_max = int(shape[0] * (yc + h/2.))

        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = name_list[int(clsid)]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(x_min)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(y_min)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(x_max)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(y_max)
    f.close()

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    xml_name = file_name + '.xml'
    xml_file = open(os.path.join(xml_dir, xml_name), 'wb')
    xml_file.write(xml)
    xml_file.close()


if __name__ == "__main__":

    yolo2voc("data/tux_hacking.txt", 'test', "data/", "data/")




