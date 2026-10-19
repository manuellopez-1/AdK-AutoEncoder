import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import sys

from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, Reshape, Flatten


def alphacarboner(ct):
    output = np.zeros((214,3))
    count = 0
    for line in ct:
        if 'CA' in line:
            s = [entry for entry in line.split() if entry]
            x = float(s[-6]);y = float(s[-5]); z = float(s[-4])
            output[count,:] = [x,y,z]
            count +=1
    return output

def distance(arr1, arr2):
    out = 0
    for i in range(len(arr1)):
        out += (arr1[i]-arr2[i])**2
    return np.sqrt(out)

def func(val):
    return np.exp(-val/.8) #idk i like .8 for AdK
def nodemaker(matrix):
    output = np.zeros(( len(matrix[:,0]), len(matrix[:,0]) ))
    for i in range(len(matrix[:,0])):
        for I in range(i+1, len(matrix[:,0])):
            output[i,I]=output[I,i] = func(distance(matrix[i,:], matrix[I,:]))
    return output

if os.path.isfile('graphs.npz')==False:
    p = '/gpfs/gibbs/pi/balou/mal258/AdKGrofiles/gro_files_'
    outputs = []
    MS=[]
    cual = []
    for i in range(11):
        for f in os.listdir(p + str(i) ):
            if '.gro' in f:
                fil = open(p + str(i) + '/' +f,'r' )
                ct = fil.readlines()
                fil.close()
                graph = nodemaker(alphacarboner(ct))
                outputs.append(graph)
                MS.append(i)
                cual.append(int(f.split('.')[0]))
    outputs = np.array(outputs)
    np.savez('graphs.npz',outputs=outputs,MS=MS,cual=cual)
else:
    A=np.load('graphs.npz')
    outputs = A['outputs']
    MS = A['MS']
    cual = A['cual']


triangle_indices = np.triu_indices(outputs.shape[1], k=1)

compressedoutputs = outputs[:,triangle_indices[0],triangle_indices[1]]

compressedoutputs = compressedoutputs.astype(np.float32)

MS = np.asarray(MS, dtype=np.int32)


print(outputs.shape)
print(compressedoutputs.shape)


class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder,**kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def encoder(inputsize, LS):
    siz = inputsize
    L = []
    sizes = [512,128]
    for siz in sizes:
        #let's start with leakyrelu and go from there
        L.append(layers.Dense(siz, activation=layers.LeakyReLU(alpha=0.2)))
        L.append(layers.Dropout(.1))
    L.append(layers.Dense(LS, activation = None))
    return tf.keras.Sequential(L, name="encoder")


def decoder(outputsize, LS):
    siz = LS
    L = [layers.Input(shape=(LS,))]
    sizes = [128,512]
    for siz in sizes:
        L.append(layers.Dense(siz, activation=layers.LeakyReLU(alpha=0.2)))
        L.append(layers.Dropout(.1))
    L.append(layers.Dense(outputsize, activation="sigmoid"))
    return tf.keras.Sequential(L, name="decoder")



ae_model = Autoencoder(encoder(compressedoutputs.shape[1],2),decoder(compressedoutputs.shape[1],2 ), name='autoencoder')

ae_model.build(input_shape=(None, compressedoutputs.shape[1]))

ae_model.summary()
ae_model.encoder.summary()
ae_model.decoder.summary()

def mse_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    #bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return mse_loss




import matplotlib.pyplot as plt
from PIL import Image
import io
import imageio

class ImageVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, model, sample_inputs,MS, every =100, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.sample_inputs = sample_inputs
        self.MS = MS
        self.imgs = []
        self.every = 10
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1)%self.every != 0: #save every 10 
            return
        encoding = self.model.encoder(self.sample_inputs,training=False)
        if isinstance(encoding, (list, tuple)):
            encoding = encoding[0]
        encoding = encoding.numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(encoding[:,0], encoding[:,1], c = self.MS, cmap = 'coolwarm', s= 1)
        ax.set_xlabel("CV1")
        ax.set_ylabel("CV2")
        ax.set_title('Epoch ' + str(epoch+1))
        if os.path.isdir('imgs')==False:
            os.mkdir('imgs')
        plt.savefig('imgs/'+str(epoch) + '.png')

        self.imgs.append(Image.open('imgs/'+str(epoch) + '.png').copy())
        plt.close(fig)
    def save_gif(self, filename='LatentspaceProgression', loop=True, duration=500):
        imgs = self.imgs
        self.imgs[0].save(filename+'.gif', save_all=True, append_images=self.imgs[1:],loop=loop, duration=duration)
imgviz_callback = ImageVisualizer(ae_model, compressedoutputs,MS)
ae_model.compile(optimizer   = tf.keras.optimizers.Adam(learning_rate=0.0001),loss= mse_loss, metrics = [tf.keras.metrics.MeanSquaredError(),tf.keras.metrics.BinaryCrossentropy()])

rng = np.random.RandomState(42)
indices = rng.permutation(len(compressedoutputs))

X_val = compressedoutputs[indices[:1000]]
X_train = compressedoutputs[indices[1000:]]

ae_model.fit(X_train, X_train,epochs=10000,batch_size=64,validation_data=(X_val, X_val),callbacks=[imgviz_callback])

imgviz_callback.save_gif()
ae_model.save_weights("autoencoder_final_weights.h5")
