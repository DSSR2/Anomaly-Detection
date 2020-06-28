import cv2
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, SpatialDropout2D, ReLU, LeakyReLU, Dense, Reshape, Flatten, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, model_from_json, Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adadelta
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm

class AD():
    def __init__(self, model_name="AD1", train_path="Dataset/", gray_scale=True,save_path="Models/", img_size=(200,200), patch_size=40):
        self.img_size = img_size
        self.model = get_model(model_name, self.img_size)
        self.train_path = train_path
        self.save_path = save_path
        self.patch_size = patch_size
        self.batch_size = 16
        self.gray_scale = gray_scale
        self.load_data(train_path)

    def train(self, epochs):
        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.model.compile(optimizer, loss="binary_crossentropy", metrics=['mse', "mae"])
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, min_lr=1e-8, min_delta=0.0003)
        val_split = 0.3
        data_idx = np.random.permutation(len(self.data))
        train_idx = data_idx[:int((1-val_split)*len(self.data))]
        valid_idx = data_idx[int((1-val_split)*len(self.data)):]
        train_data = self.data[train_idx]
        valid_data = self.data[valid_idx]
        print("train on ", len(train_data), " images")
        print("validate on ", len(valid_data), " images")

        self.hist = self.model.fit(train_data, train_data, validation_data=(valid_data, valid_data), batch_size=self.batch_size, epochs=epochs, verbose=1, callbacks=[reduce_lr])
        save_model(self.model, self.save_path+"/model")


    def test_img(self, img_path):
        if(self.gray_scale):
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, self.img_size)
            img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        else: 
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_size)
        
        fname = img_path.split("\\")[-1]
        print(img.shape)
        img = np.reshape(img, (1, img.shape[0], img.shape[1], img.shape[2]))
        pred = self.model.predict(img/255)
        pred = tf.squeeze(pred)
        pred = pred.numpy()
        pred = np.round(pred*255)
        pred = pred.astype("uint8")
        img = img.reshape((img.shape[1], img.shape[2], img.shape[3]))
        print(img.shape, pred.shape)
        im_h = cv2.hconcat([img, pred])
        plt.imshow(im_h, cmap="gray")
        plt.show()

    def predict(self, path, save_path="Output/", show=False, save=True, patch_size=0, thresh=0.8, hardcore=False):
        if(patch_size):
            self.patch_size = patch_size

        sing_f = False
        if(os.path.isfile(path)):
            sing_f = True
            files = [path]
        else:
            files = glob(path+"/*.jpg") + glob(path+"/*.bmp") + glob(path+"/*.png") + glob(path+"/*.jpeg")
            
        for i in tqdm(files):
            fname = i.split("\\")[-1]
            if(self.gray_scale):
                og = cv2.imread(i, 0)
                og = cv2.resize(og, self.img_size)
                og = np.reshape(og, (og.shape[0], og.shape[1], 1))
            else:
                og = cv2.imread(i)
                og = cv2.resize(og, self.img_size)
            
            img = np.reshape(og, (1, og.shape[0], og.shape[1], og.shape[2]))
            pred = self.model.predict(img)
            pred = tf.squeeze(pred)
            pred = pred.numpy()
            pred = pred.astype("uint8")
            if(self.gray_scale):
                img = np.reshape(img, (img.shape[1], img.shape[2]))
            else:
                img = img.reshape((img.shape[1], img.shape[2], img.shape[3]))
            ret = self.test(img, pred, thresh, hardcore)

            if(show):
                plt.figure()
                plt.imshow(ret)
            if(save):
                cv2.imwrite(save_path+"/"+fname, ret)

    def test(self, img, pred, thresh=0.8, hardcore=False):
        img_patches = split_image(img, self.patch_size)
        pred_patches = split_image(pred, self.patch_size)
        
        errs = []
        
        for i in range(len(img_patches)):
            errs.append((abs(pred_patches[i]-img_patches[i])))
        errs = np.asarray(errs)

        val = errs.mean() + thresh*errs.std()

        op_arr = []
        for i in range(len(errs)):
            if(errs[i].mean()>=val):
                e = errs[i]
                e[e<=val] = img_patches[i][e<=val]
                if(hardcore):
                    op_arr.append(np.full((self.patch_size, self.patch_size), 1))
                else: 
                    op_arr.append(errs[i])
                
            else:
                op_arr.append(img_patches[i])
        img = img
        rec = combine_image(op_arr)
        if(self.gray_scale):
            fin = np.stack((img, img, rec))
            fin = np.moveaxis(fin, 0, -1)
        else: 
            fin = cv2.addWeighted(img, 0.7, rec, 0.2)      
        return fin

    def load_data(self, path):
        self.train_path = path
        files = glob(self.train_path+"/*.jpg") + glob(self.train_path+"/*.jpeg") + glob(self.train_path+"/*.png") + glob(self.train_path+"/*.bmp")
        self.data = []
        for i in tqdm(files):
            if(self.gray_scale):
                img = cv2.imread(i, 0)
            else: 
                img = cv2.imread(i)
            img = cv2.resize(img, self.img_size)
            img = img/255
            self.data.append(img)

        self.data = np.asarray(self.data)
        print(len(files), " files found and loaded.")

def load_model(name):
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(name+".h5")
    print("Model loaded successfully!")
    return model

def save_model(model, name):
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(name+".h5")
    print("Model Saved")

def find_diff(img, pred):
    return 0

def get_model(model_name, img_size):
    if(os.path.exists(model_name+".h5")):
        print("Model files found!")
        return load_model(model_name)
    elif(model_name=="AD0"):
        return make_model(input_shape=(200,200,3))
    elif(model_name=="AD1"):
        return make_model(input_shape=(200,200,1))
    elif(model_name=="AD2"):
        return make_model(input_shape=(256,256,1))
    else:
        print(model_name, " Invalid, loading AD1")
        return make_model(input_shape=(200,200,1))




def make_model(input_shape = (200,200,1)):
    input_img = Input(shape=input_shape)
    x = Conv2D(8, (5, 5), padding='same')(input_img)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(8, (5, 5), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(4, (5, 5), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    bflat = MaxPooling2D((2, 2), padding='same')(x)
    
    flat = Flatten()(bflat)
    
    x = Dense(1024)(flat)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Dense(flat.shape[1])(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Reshape((bflat.shape[1], bflat.shape[2], bflat.shape[3]))(x)
    
    x = Conv2D(4, (5, 5), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(8, (5, 5), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(8, (5, 5), padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = UpSampling2D((2, 2))(x)
    
    x = Conv2D(input_shape[2], (5,5), padding="same", activation="sigmoid")(x)
    
    model = Model(input_img, x)

    return model

def split_image(img, window_size):
    patches = []
    ctr = 0
    for r in range(0, img.shape[0], window_size):
        for c in range(0, img.shape[0], window_size):
            patches.append(img[r:min(img.shape[0],r+window_size), c:min(img.shape[0], c+window_size)])
            ctr+=1
    patches = np.asarray(patches)
    return patches

def combine_image(patches):
    rows = cols = int(np.sqrt(len(patches)))
    img = np.asarray([])
    for i in range(cols):
        row = patches[i*cols]
        for j in range (1, rows):
            row = np.concatenate((row, patches[i*cols+j]), axis = 1)
        img = np.vstack([img, row]) if img.size else row
    return img