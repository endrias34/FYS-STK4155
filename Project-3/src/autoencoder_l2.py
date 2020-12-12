import keras.losses
from keras.layers import InputLayer, Conv2D, add
from keras.layers import MaxPooling2D, UpSampling2D
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras import regularizers

def denoise_network_train2(input_shape, filters, filter_sizes, AF, last_layer_AF,
                           lmbd, eta, loss, optimizer):

    weights = RandomUniform(minval=-0.04, maxval=0.04, seed=None)
    model = Sequential()

    model.add(InputLayer(input_shape=input_shape))
    model.add(Conv2D(filters[0], filter_sizes[0], activation = AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters[1], filter_sizes[1], activation = AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters[2], filter_sizes[2], activation = AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(filters[3], filter_sizes[3], activation = AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(filters[4], filter_sizes[4], activation = AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(filters[5], filter_sizes[5], activation = AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(UpSampling2D(size = (2, 2)))
    model.add(Conv2D(filters[6], filter_sizes[6], activation = AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))
    model.add(Conv2D(1, (3, 3), activation=last_layer_AF, input_shape=input_shape, padding = 'same',
                   kernel_initializer = weights, kernel_regularizer=regularizers.l2(lmbd)))

    if optimizer == 'RMSprop':
        optimizer = RMSprop(lr=eta)

    model.compile(loss=loss, optimizer=optimizer)

    return model
