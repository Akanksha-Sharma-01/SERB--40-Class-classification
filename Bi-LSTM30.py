import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, initializers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

keras.utils.set_random_seed(12)
tf.config.experimental.enable_op_determinism()

TrainLabel = np.load('/storage/akanksha/NewData/Data/Train/label.npy')
TrainData = np.load('/storage/akanksha/NewData/Data/Train/eeg.npy')
ValLabel = np.load('/storage/akanksha/NewData/Data/Val/label.npy')
TestLabel = np.load('/storage/akanksha/NewData/Data/Test/label.npy')
ValData = np.load('/storage/akanksha/NewData/Data/Val/eeg.npy')
TestData = np.load('/storage/akanksha/NewData/Data/Test/eeg.npy')

def preprocess(eeg_data, label):
	data = []
	labels = []
	for i in range(np.shape(eeg_data)[0]):
		for j in range(11):
			data.append(eeg_data[i,:,j*20:200+(1+j)*20])
			labels.append(label[i])
	labels = np.array(labels)
	data = np.array(data)
	return data,labels

TrainData, TrainLabel = preprocess(TrainData, TrainLabel)
ValData, ValLabel = preprocess(ValData, ValLabel)
TestData, TestLabel = preprocess(TestData, TestLabel)

print(np.shape(TrainData))
TrainData = np.reshape(TrainData, [TrainData.shape[0], 128,220,1])
TrainLabel = np.reshape(TrainLabel, [TrainLabel.shape[0],])
ValData = np.reshape(ValData, [ValData.shape[0],128,220,1])
ValLabel = np.reshape(ValLabel, [ValLabel.shape[0],])
TestData = np.reshape(TestData, [TestData.shape[0], 128,220,1])
TestLabel = np.reshape(TestLabel, [TestLabel.shape[0],])
maxlen = 16      # Only consider 3 input time points
embed_dim_1 = 128  # Features of each time point
embed_dim2 = 25
num_heads = 8   # Number of attention heads 25
ff_dim = 64     # Hidden layer size in feed forward network inside transformer

def positionalEmbedding(input,embed_dim):
	position = tf.range(start=0, limit=maxlen, delta=1)
	emm = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)(position)
	input = tf.reshape(input, [-1, maxlen, embed_dim])
	output = input + emm
	return output

def transformer(input, embed_dim, rate, ff_dim):
	att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(input,input)
	att = layers.Dropout(rate)(att,training = True)
	normal1 = layers.LayerNormalization(epsilon=1e-6)(att+input)
	ff = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])(normal1)
	ff = layers.Dropout(rate)(ff, training = True)
	normal2 = layers.LayerNormalization(epsilon=1e-6)(ff+normal1)
	return normal2

def encoder(input):
	x = tf.keras.layers.Conv2D(kernel_size=(1, 35),   filters=25, strides = (1,2), activation = 'relu', kernel_initializer = "RandomNormal")(input)
	x = tf.keras.layers.Conv2D(kernel_size=(1, 35),   filters=25, strides = (1,2), activation = 'relu', kernel_initializer = "RandomNormal" )(x)
	x = tf.keras.layers.Conv2D(kernel_size=(128, 1), filters=25 ,strides = (1, 1), activation = 'sigmoid', kernel_initializer = "RandomNormal")(x)
	x = tf.keras.layers.Reshape((30,25))(x)
	x = tf.keras.layers.Bidirectional(layers.LSTM(30, return_sequences=True),input_shape=(30, 25))(x)
	x = tf.keras.layers.Bidirectional(layers.LSTM(30))(x)
	#x = transformer(x, embed_dim2, 0.5, ff_dim)
	x = tf.keras.layers.Flatten()(x)
	return x

def classifier(input):
	x = layers.Dense(70, activation="sigmoid", kernel_initializer = "RandomNormal")(input)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.5)(x)
	output = layers.Dense(39, activation="softmax", kernel_initializer = "RandomNormal")(x)
	return output

def Model():
	Input = layers.Input(shape=(128,220,1))
	Encoding = encoder(Input)
	Output = classifier(Encoding)
	Mmodel = keras.Model(inputs=Input, outputs=Output)
	return Mmodel

with tf.device('/GPU:0'):
    model = Model()
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3,100,t_mul=2.0,m_mul=1.0,alpha=1e-5,name=None)
    #my_callbacks=[keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.0001,patience=5 )]
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler),loss="SparseCategoricalCrossentropy",metrics=['accuracy'])
    model.summary()
    history = model.fit(TrainData, TrainLabel, batch_size=40, epochs=19, validation_split=0, validation_data=(ValData, ValLabel),verbose = 2)#, callbacks=my_callbacks)
    model.save("/home/akanksha/Store/40ClassModel3Bi_LSTM30.h5")

    AccTr = history.history['accuracy']
    AccVal = history.history['val_accuracy']
    LossTr = history.history['loss']
    LossVal = history.history['val_loss']
    plt.subplot(2,2,1)
    plt.plot(AccTr, label = "Accuracy")
    plt.plot(LossTr, label = "loss")
    plt.legend()
    plt.grid()
    plt.title("Training")

    plt.subplot(2,2,2)
    plt.plot(AccVal, label = "Accuracy")
    plt.plot(LossVal, label = "Loss")
    plt.legend()
    plt.grid()
    plt.title("Validation")

    plt.subplot(2,2,3)
    plt.plot(AccTr, label = "Training")
    plt.plot(AccVal, label = "Validation")
    plt.legend()
    plt.grid()
    plt.title("Accuracy")

    plt.subplot(2,2,4)
    plt.plot(LossTr, label = "Training")
    plt.plot(LossVal, label = "Validation")
    plt.legend()
    plt.grid()
    plt.title("Loss")
    plt.imsave("Bi-LSTM30.png")

    TrainAcc = model.evaluate(TrainData,TrainLabel)
    TestAcc = model.evaluate(TestData,TestLabel)

    print("Training Accuracy",TrainAcc)
    print("Test Accuracy",TestAcc)
