import os
import struct
import numpy as np
from PIL import Image
from skimage import io, transform


def load_mnist(path, kind):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

'''
def load_usps(path):

    dirs = [str(i) for i in range(10)]

    images = []
    labels = []

    for i in range(10):
        tpath = path + "/" + dirs[i]
        file_list = os.listdir(tpath)
        t_labels = []
        t_images = []
        for j in file_list:
            if '.png' != j[-4:]:
                continue
            else:
                img = io.imread(tpath + "/" + j)
                img = img[:,:,0]
                img = 255 - img
                img1 = transform.resize(img, (46, 46))
                img1 *= 255
                img1 = img1.astype(int)
                img1 = transform.rescale(img1, 28/46)
                img1.shape = (784,)
                t_images.append(img1)
                t_labels.append(i)

        images += t_images
        labels += t_labels
    print("Reading USPS is done")
    images = np.array(images)
    labels = np.array(labels)

    return images, labels
'''

def load_usps(path):

    dirs = [str(i) for i in range(10)]

    images = []
    labels = []

    for i in range(10):
        tpath = path + "/" + dirs[i]
        file_list = os.listdir(tpath)
        t_labels = []
        t_images = []
        for j in file_list:
            if '.png' != j[-4:]:
                continue
            else:
                img = Image.open(tpath + "/" + j).convert('L').resize((28, 28), Image.ANTIALIAS)
                img_data = list(img.getdata())
                t_images.append(img_data)
                t_labels.append(i)

        images += t_images
        labels += t_labels
    print("Reading USPS is done")
    images = np.divide(np.subtract(255, np.array(images)), 255)
    labels = np.array(labels)

    return images, labels


'''
def load_usps_data(filename, height=28, width=28):
    extract_path = "usps_data"
    img_names = []
    img_data_list = []
    labels = []

    with zipfile.ZipFile(filename, 'r') as zip:
        zip.extractall(extract_path)

    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)

        if "Numerals" in path:
            files_images = [fname for fname in files if fname.find(".png") >= 0]
            for fil in files_images:
                labels.append(int(path[-1]))
                img_names.append(os.path.join(*path, fil))  # * splat operator

    for idx, img_fname in enumerate(img_names):
        img = Image.open(img_fname).convert('L').resize((height, width), Image.ANTIALIAS)
        img_data = list(img.getdata())
        img_data_list.append(img_data)

    img_data_array = np.divide(np.subtract(255, np.array(img_data_list)), 255)
    usps_labels = one_hot_array(np.array(labels), 10)

    return img_data_array, usps_labels
'''



