import os

import h5py
import numpy as np
import scipy.misc
import torch

np.random.seed(123)

# loading data from .h5
class DataLoaderH5(object):
    def __init__(self, **kwargs):
        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']

        # read data info from lists
        f = h5py.File(kwargs['data_h5'], "r")
        self.im_set = np.array(f['images'])
        self.lab_set = np.array(f['labels'])

        self.num = self.im_set.shape[0]
        assert self.im_set.shape[0]==self.lab_set.shape[0], '#images and #labels do not match!'
        assert self.im_set.shape[1]==self.load_size, 'Image size error!'
        assert self.im_set.shape[2]==self.load_size, 'Image size error!'
        print('# Images found:', self.num)

        self.shuffle()
        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3))
        labels_batch = np.zeros(batch_size, dtype=np.double)
        
        for i in range(batch_size):
            image = self.im_set[self._idx]
            image = image.astype(np.float32)/255. - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] = image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.lab_set[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
                if self.randomize:
                    self.shuffle()
        
        # Switch to NCHW ordering and convert to torch FloatTensor
        images_batch = torch.from_numpy(images_batch.swapaxes(2, 3).swapaxes(1, 2)).float()
        labels_batch = torch.from_numpy(labels_batch).float()
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

    def shuffle(self):
        perm = np.random.permutation(self.num)
        self.im_set = self.im_set[perm] 
        self.lab_set = self.lab_set[perm]

# Loading data from disk
class DataLoaderDisk(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        self.list_lab = []
        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path, lab =line.rstrip().split(' ')
                self.list_im.append(os.path.join(self.data_root, path))
                self.list_lab.append(int(lab))
        self.list_im = np.array(self.list_im, np.object)
        self.list_lab = np.array(self.list_lab, np.int64)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        # permutation
        perm = np.random.permutation(self.num) 
        self.list_im[:, ...] = self.list_im[perm, ...]
        self.list_lab[:] = self.list_lab[perm, ...]

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3), )
        labels_batch = np.zeros(batch_size, dtype=np.double)
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            labels_batch[i, ...] = self.list_lab[self._idx]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        # Switch to NCHW ordering and convert to torch FloatTensor
        images_batch = torch.from_numpy(images_batch.swapaxes(2, 3).swapaxes(1, 2)).float()
        labels_batch = torch.from_numpy(labels_batch).long()
        return images_batch, labels_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0

# Loading test data sequentially from disk
class DataLoaderDiskTest(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []
        for im_name in sorted(os.listdir(os.path.join(self.data_root, 'test'))):
            self.list_im.append(os.path.join(self.data_root, 'test', im_name))
        self.list_im = np.array(self.list_im, np.object)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)

        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, 3, self.fine_size, self.fine_size))
        paths_batch = []
        
        for i in range(batch_size):
            paths_batch.append(self.list_im[self._idx])
            
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            
            # Switch to NCHW ordering
            images_batch[i] = image.swapaxes(1, 2).swapaxes(0, 1)
            
            self._idx += 1
            self._idx %= self.size()
        
        # Convert to torch FloatTensor
        images_batch = torch.from_numpy(images_batch).float()
        
        return images_batch, paths_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
        
def load_objects():
    # def load_xml():
    object_files = glob.glob("../../data/objects/train/*/*/*.xml")
    object_files.sort()

    objects = []

    for path in object_files:
        with open(path, "r") as f:
            xml = f.read()
            xml = """<?xml version="1.0"?>
            <base>
            """ + xml + """
            </base>
            """

            tree = ET.fromstring(xml)

            path = os.path.join("../../data/images/train", tree.find("folder").text, tree.find("filename").text)

            objs = []
            for obj in tree.findall("objects"):
                bndbox = obj.find("bndbox")
                objdata = {
                    "class": int(obj.find("class").text),
                    # "polygon": obj.find("polygon"),  # ignoring polygon for now
                    "bndbox": (
                        (
                            int(bndbox.find("xmin").text),
                            int(bndbox.find("xmax").text)
                        ),
                        (
                            int(bndbox.find("ymin").text),
                            int(bndbox.find("ymax").text)
                        )
                    )
                }
                objs.append(objdata)

            data = {
                "path": path,
                "class": int(tree.find("class").text),
                "objects": objs,
            }
            objects.append(data)
    return objects
