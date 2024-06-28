import cv2
import numpy as np
import albumentations as A
import tensorflow as tf
from tensorflow.data import AUTOTUNE
import os

class COLON_data(tf.keras.utils.Sequence):
    def __init__(self, batchSize, imgSize, inputImgPaths, labelsImgPaths,data_type):
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.inputImgPaths = inputImgPaths
        self.labelsImgPaths = labelsImgPaths
        self.data_type = data_type


    def __len__(self):
        return len(self.inputImgPaths) // self.batchSize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchSize
        batchInputImgPaths = self.inputImgPaths[i: i + self.batchSize]
        batchLabelsImgPaths = self.labelsImgPaths[i: i + self.batchSize]

        x = np.zeros((self.batchSize,) + self.imgSize + (3,), dtype="float32")
        z = np.zeros((self.batchSize,) + self.imgSize + (1,), dtype="float32")
        for j, (input_image, input_label) in enumerate(zip(batchInputImgPaths, batchLabelsImgPaths)):
            img = cv2.imread(input_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))

            lbl = cv2.imread(input_label, cv2.IMREAD_GRAYSCALE)
            lbl = cv2.resize(lbl, (224, 224))
            lbl = np.expand_dims(lbl, -1)

            x[j] = img.astype("float32") / 255.0
            z[j] = lbl.astype("float32") / 255.0

        return x, z



def getDataLoader(batch_size):
    """ Create dataloader and return dataloader object which can be used with 
        model.fit
    """
    TrainIMAGES_DIR = "/kaggle/input/new-dataset/EBHI-SEG/Train/image"
    TrainMASKS_DIR = "/kaggle/input/new-dataset/EBHI-SEG/Train/label"
                      
    TestIMAGES_DIR = "/kaggle/input/new-dataset/EBHI-SEG/Test/image"
    TestMASKS_DIR = "/kaggle/input/new-dataset/EBHI-SEG/Test/label"


    AdenoImg ="/kaggle/input/new-one/EBHI-SEG/Adenocarcinoma/image"
    AdenoMask ="/kaggle/input/new-one/EBHI-SEG/Adenocarcinoma/label"

    NorImg ="/kaggle/input/new-one/EBHI-SEG/Normal/image"
    NorMask ="/kaggle/input/new-one/EBHI-SEG/Normal/label"

    PolypImg ="/kaggle/input/new-one/EBHI-SEG/Polyp/image"
    PolypMask ="/kaggle/input/new-one/EBHI-SEG/Polyp/label"

    SerratedImg ="/kaggle/input/new-one/EBHI-SEG/Serrated adenoma/image"
    SerratedMask ="/kaggle/input/new-one/EBHI-SEG/Serrated adenoma/label"

    HgradeImg ="/kaggle/input/new-one/EBHI-SEG/High-grade IN/image"
    HgradeMask ="/kaggle/input/new-one/EBHI-SEG/High-grade IN/label"

    LgradeImg ="/kaggle/input/new-one/EBHI-SEG/Low-grade IN/image"
    LgradeMask ="/kaggle/input/new-one/EBHI-SEG/Low-grade IN/label"

    IMAGE_SIZE = (224, 224)

    """
    inputImgPaths = sorted([os.path.join(IMAGES_DIR, x) for x in os.listdir(IMAGES_DIR)])
    targetImgPaths = sorted([os.path.join(MASKS_DIR, x) for x in os.listdir(MASKS_DIR)])
    """

    X_train = sorted([os.path.join(TrainIMAGES_DIR, x) for x in os.listdir(TrainIMAGES_DIR)])
    X_test = sorted([os.path.join(TestIMAGES_DIR, x) for x in os.listdir(TestIMAGES_DIR)])

    Y_train = sorted([os.path.join(TrainMASKS_DIR, x) for x in os.listdir(TrainMASKS_DIR)])
    Y_test = sorted([os.path.join(TestMASKS_DIR, x) for x in os.listdir(TestMASKS_DIR)])

    

    AdenoX_test = sorted([os.path.join(AdenoImg, x) for x in os.listdir(AdenoImg)])
    AdenoY_test = sorted([os.path.join(AdenoMask, x) for x in os.listdir(AdenoMask)])

    NorX_test = sorted([os.path.join(NorImg, x) for x in os.listdir(NorImg)])
    NorY_test = sorted([os.path.join(NorMask, x) for x in os.listdir(NorMask)])

    PolypX_test = sorted([os.path.join(PolypImg, x) for x in os.listdir(PolypImg)])
    PolypY_test = sorted([os.path.join(PolypMask, x) for x in os.listdir(PolypMask)])

    SerratedX_test = sorted([os.path.join(SerratedImg, x) for x in os.listdir(SerratedImg)])
    SerratedY_test = sorted([os.path.join(SerratedMask, x) for x in os.listdir(SerratedMask)])

    HgradeX_test = sorted([os.path.join(HgradeImg, x) for x in os.listdir(HgradeImg)])
    HgradeY_test = sorted([os.path.join(HgradeMask, x) for x in os.listdir(HgradeMask)])

    LgradeX_test = sorted([os.path.join(LgradeImg, x) for x in os.listdir(LgradeImg)])
    LgradeY_test = sorted([os.path.join(LgradeMask, x) for x in os.listdir(LgradeMask)])
    
    trainGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = X_train, labelsImgPaths = Y_train,data_type = "Train")
    testGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = X_test, labelsImgPaths = Y_test,data_type = "Test")

    
    AdenoTestGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = AdenoX_test, labelsImgPaths = AdenoY_test,data_type = "Test")
    NormalTestGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = NorX_test, labelsImgPaths = NorY_test,data_type = "Test")
    PolypTestGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = PolypX_test, labelsImgPaths = PolypY_test,data_type = "Test")
    SerratedTestGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = SerratedX_test, labelsImgPaths = SerratedY_test,data_type = "Test")
    HgradeTestGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = HgradeX_test, labelsImgPaths = HgradeY_test,data_type = "Test")
    LgradeTestGen = COLON_data(batchSize = batch_size, imgSize = IMAGE_SIZE, inputImgPaths = LgradeX_test, labelsImgPaths = LgradeY_test,data_type = "Test")

    return trainGen, testGen, AdenoTestGen, NormalTestGen, PolypTestGen, SerratedTestGen, HgradeTestGen, LgradeTestGen


