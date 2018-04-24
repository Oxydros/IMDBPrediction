import numpy
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import SGD, RMSprop, Adam


def SequentialRNNModel(Xtrain, Ytrain, Xtest, Ytest):

    model = Sequential()
    nbFeatures = len(Xtrain[0])
    epochs = 300
    batch_size = 512

    Xtrain = Xtrain.reshape(len(Xtrain), 1, nbFeatures)
    Xtest = Xtest.reshape(len(Xtest), 1, nbFeatures)

    # Ytrain = keras.utils.to_categorical(Ytrain)
    # Ytest = keras.utils.to_categorical(Ytest)

    nbOutput = 1#len(Ytrain[0])

    Ytrain = numpy.asarray(Ytrain).reshape(len(Ytrain), 1, nbOutput)
    Ytest = numpy.asarray(Ytest).reshape(len(Ytest), 1, nbOutput)

    model.add(Dense(nbFeatures, activation='relu',
                    input_shape=(1, nbFeatures)))
    model.add(Dropout(0.5))
    model.add(LSTM(batch_size, return_sequences=True))
    model.add(Dropout(0.7))
    model.add(LSTM(batch_size, return_sequences=True))
    model.add(Dropout(0.7))
    model.add(Dense(nbOutput, activation='relu'))

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

    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('RNN model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Accuracy_RNN.png")
    plt.clf()

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('RNN model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Loss_RNN.png")
    plt.clf()
