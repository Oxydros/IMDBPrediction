
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import SGD, RMSprop, Adam

def SequentialCNNModel(Xtrain, Ytrain, Xtest, Ytest):
    model = Sequential()
    nbFeatures = len(Xtrain[0])
    epochs = 100
    batch_size = 1024

    # Ytrain = [1 if i > 50 else 0 for i in Ytrain]
    # Ytest = [1 if i > 50 else 0 for i in Ytest]
    # Ytrain = keras.utils.to_categorical(Ytrain)
    # Ytest = keras.utils.to_categorical(Ytest)

    print("Number of features: %d"%nbFeatures)

    model.add(Dense(nbFeatures, activation='relu',
                    input_shape=(nbFeatures,)))

    ##
    # model.add(Dropout(0.2))
    # model.add(Dense((nbFeatures + 1), activation='relu'))
    # model.add(Dropout(0.2))
    ##

    model.add(Dense(nbFeatures * 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nbFeatures * 4, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(nbFeatures * 2, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))

    print(model.summary())

    # optimizer = RMSprop(lr=0.001)
    optimizer = Adam()

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    history = model.fit(Xtrain, Ytrain, epochs=epochs,
              batch_size=batch_size,
              verbose=1,
              validation_data=(Xtest, Ytest))
    score = model.evaluate(Xtest, Ytest, batch_size=batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('CNN model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Accuracy_CNN.png")
    plt.clf()

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Loss_CNN.png")
    plt.clf()
