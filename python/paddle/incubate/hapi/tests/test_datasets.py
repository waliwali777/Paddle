# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import os
import numpy as np
import tempfile
import shutil
import cv2

from paddle.incubate.hapi.datasets import *
from paddle.incubate.hapi.datasets.utils import _check_exists_and_download


class TestFolderDatasets(unittest.TestCase):
    def setUp(self):
        self.data_dir = tempfile.mkdtemp()
        self.empty_dir = tempfile.mkdtemp()
        for i in range(2):
            sub_dir = os.path.join(self.data_dir, 'class_' + str(i))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            for j in range(2):
                fake_img = (np.random.random((32, 32, 3)) * 255).astype('uint8')
                cv2.imwrite(os.path.join(sub_dir, str(j) + '.jpg'), fake_img)

    def tearDown(self):
        shutil.rmtree(self.data_dir)

    def test_dataset(self):
        dataset_folder = DatasetFolder(self.data_dir)

        for _ in dataset_folder:
            pass

        assert len(dataset_folder) == 4
        assert len(dataset_folder.classes) == 2

        dataset_folder = DatasetFolder(self.data_dir)
        for _ in dataset_folder:
            pass

    def test_folder(self):
        loader = ImageFolder(self.data_dir)

        for _ in loader:
            pass

        loader = ImageFolder(self.data_dir)
        for _ in loader:
            pass

        assert len(loader) == 4

    def test_transform(self):
        def fake_transform(img):
            return img

        transfrom = fake_transform
        dataset_folder = DatasetFolder(self.data_dir, transform=transfrom)

        for _ in dataset_folder:
            pass

        loader = ImageFolder(self.data_dir, transform=transfrom)
        for _ in loader:
            pass

    def test_errors(self):
        with self.assertRaises(RuntimeError):
            ImageFolder(self.empty_dir)
        with self.assertRaises(RuntimeError):
            DatasetFolder(self.empty_dir)

        with self.assertRaises(ValueError):
            _check_exists_and_download('temp_paddle', None, None, None, False)


class TestMNISTTest(unittest.TestCase):
    def test_main(self):
        mnist = MNIST(mode='test')
        self.assertTrue(len(mnist) == 10000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 1)
            self.assertTrue(image.shape[1] == 28)
            self.assertTrue(image.shape[2] == 28)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)


class TestMNISTTrain(unittest.TestCase):
    def test_main(self):
        mnist = MNIST(mode='train', chw_format=False)
        self.assertTrue(len(mnist) == 60000)

        for i in range(len(mnist)):
            image, label = mnist[i]
            self.assertTrue(image.shape[0] == 784)
            self.assertTrue(label.shape[0] == 1)
            self.assertTrue(0 <= int(label) <= 9)


class TestFlowersTrain(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='train')
        self.assertTrue(len(flowers) == 6149)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 6149)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersValid(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='valid')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestFlowersTest(unittest.TestCase):
    def test_main(self):
        flowers = Flowers(mode='test')
        self.assertTrue(len(flowers) == 1020)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1020)
        image, label = flowers[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(image.shape[2] == 3)
        self.assertTrue(label.shape[0] == 1)


class TestCifarTrain10(unittest.TestCase):
    def test_main(self):
        cifar = Cifar(mode='train10')
        self.assertTrue(len(cifar) == 50000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 50000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(data.shape[0] == 3072)
        self.assertTrue(0 <= int(label) <= 9)


class TestCifarTest10(unittest.TestCase):
    def test_main(self):
        cifar = Cifar(mode='test10')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(data.shape[0] == 3072)
        self.assertTrue(0 <= int(label) <= 9)


class TestCifarTrain100(unittest.TestCase):
    def test_main(self):
        cifar = Cifar(mode='train100')
        self.assertTrue(len(cifar) == 50000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 50000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(data.shape[0] == 3072)
        self.assertTrue(0 <= int(label) <= 99)


class TestCifarTest100(unittest.TestCase):
    def test_main(self):
        cifar = Cifar(mode='test100')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(data.shape[0] == 3072)
        self.assertTrue(0 <= int(label) <= 99)


class TestVOC2012Train(unittest.TestCase):
    def test_main(self):
        voc2012 = VOC2012(mode='train')
        self.assertTrue(len(voc2012) == 2913)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 2913)
        image, label = voc2012[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 3)


class TestVOC2012Valid(unittest.TestCase):
    def test_main(self):
        voc2012 = VOC2012(mode='valid')
        self.assertTrue(len(voc2012) == 1449)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1449)
        image, label = voc2012[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 3)


class TestVOC2012Test(unittest.TestCase):
    def test_main(self):
        voc2012 = VOC2012(mode='test')
        self.assertTrue(len(voc2012) == 1464)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1464)
        image, label = voc2012[idx]
        self.assertTrue(len(image.shape) == 3)
        self.assertTrue(len(label.shape) == 3)


class TestConll05st(unittest.TestCase):
    def test_main(self):
        conll05st = Conll05st()
        self.assertTrue(len(conll05st) == 5267)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 5267)
        sample = conll05st[idx]
        self.assertTrue(len(sample) == 9)
        for s in sample:
            self.assertTrue(len(s.shape) == 1)


class TestImdbTrain(unittest.TestCase):
    def test_main(self):
        imdb = Imdb(mode='train')
        self.assertTrue(len(imdb) == 25000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 25000)
        data, label = imdb[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(int(label) in [0, 1])


class TestImdbTest(unittest.TestCase):
    def test_main(self):
        imdb = Imdb(mode='test')
        self.assertTrue(len(imdb) == 25000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 25000)
        data, label = imdb[idx]
        self.assertTrue(len(data.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(int(label) in [0, 1])


class TestImikolovTrain(unittest.TestCase):
    def test_main(self):
        imikolov = Imikolov(mode='train', data_type='NGRAM', window_size=2)
        self.assertTrue(len(imikolov) == 929589)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 929589)
        data  = imikolov[idx]
        self.assertTrue(len(data) == 2)


class TestImikolovTest(unittest.TestCase):
    def test_main(self):
        imikolov = Imikolov(mode='test', data_type='NGRAM', window_size=2)
        self.assertTrue(len(imikolov) == 82430)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 82430)
        data  = imikolov[idx]
        self.assertTrue(len(data) == 2)


class TestSentimentTrain(unittest.TestCase):
    def test_main(self):
        sentiment = Sentiment(mode='train')
        self.assertTrue(len(sentiment) == 1600)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1600)
        data  = sentiment[idx]
        self.assertTrue(len(data) == 2)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(int(data[1]) in [0, 1])


class TestSentimentTest(unittest.TestCase):
    def test_main(self):
        sentiment = Sentiment(mode='test')
        self.assertTrue(len(sentiment) == 400)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 400)
        data  = sentiment[idx]
        self.assertTrue(len(data) == 2)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(int(data[1]) in [0, 1])


class TestUCIHousingTrain(unittest.TestCase):
    def test_main(self):
        uci_housing = UCIHousing(mode='train')
        self.assertTrue(len(uci_housing) == 404)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 404)
        data  = uci_housing[idx]
        self.assertTrue(len(data) == 2)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(data[0].shape[0] == 13)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(data[1].shape[0] == 1)


class TestUCIHousingTest(unittest.TestCase):
    def test_main(self):
        uci_housing = UCIHousing(mode='test')
        self.assertTrue(len(uci_housing) == 102)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 102)
        data  = uci_housing[idx]
        self.assertTrue(len(data) == 2)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(data[0].shape[0] == 13)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(data[1].shape[0] == 1)


class TestWMT14Train(unittest.TestCase):
    def test_main(self):
        wmt14 = WMT14(mode='train', dict_size=50)
        self.assertTrue(len(wmt14) == 191155)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 191155)
        data  = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT14Test(unittest.TestCase):
    def test_main(self):
        wmt14 = WMT14(mode='test', dict_size=50)
        self.assertTrue(len(wmt14) == 5957)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 5957)
        data  = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT14Gen(unittest.TestCase):
    def test_main(self):
        wmt14 = WMT14(mode='gen', dict_size=50)
        self.assertTrue(len(wmt14) == 3001)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 3001)
        data  = wmt14[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT16Train(unittest.TestCase):
    def test_main(self):
        wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50, lang='en')
        self.assertTrue(len(wmt16) == 29000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 29000)
        data  = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT16Test(unittest.TestCase):
    def test_main(self):
        wmt16 = WMT16(mode='test', src_dict_size=50, trg_dict_size=50, lang='en')
        self.assertTrue(len(wmt16) == 1000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1000)
        data  = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


class TestWMT16Val(unittest.TestCase):
    def test_main(self):
        wmt16 = WMT16(mode='val', src_dict_size=50, trg_dict_size=50, lang='en')
        self.assertTrue(len(wmt16) == 1014)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 1014)
        data  = wmt16[idx]
        self.assertTrue(len(data) == 3)
        self.assertTrue(len(data[0].shape) == 1)
        self.assertTrue(len(data[1].shape) == 1)
        self.assertTrue(len(data[2].shape) == 1)


if __name__ == '__main__':
    unittest.main()
