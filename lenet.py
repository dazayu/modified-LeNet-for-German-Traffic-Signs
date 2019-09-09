#from keras.models import Sequential
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
##from keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dense
#from keras import backend as K
##from keras import regularizers
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import regularizers
#from keras import backend as K


########################################################################

########################################################################
class LeNet:
    @staticmethod
    def build(numChannels,imgRows,imgCols,numClasses,weightsPath=None,
              dropoutR=0.5,regPara=0.01):

        #initialize the model
        model = Sequential()
#        inputShape = (imgRows,imgCols,numChannels)

        # if we are using "channel first ", update the input shape


        #Layer 1
        model.add(Conv2D(filters = 6, 
                         kernel_size = (5,5), 
                         strides = 1, 
                         padding = "same",
                         input_shape = (imgCols,imgRows,numChannels)))
        model.add(Activation(activation = "tanh"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    
        #Layer 2
        model.add(Conv2D(filters = 16, 
                         kernel_size = (5,5),
                         strides = 1,
                         padding = "same",
                         input_shape = (14,14,18)))
        model.add(Activation(activation = "tanh"))
        model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
    

        model.add(Flatten())
        model.add(Dense(units = 360, activation = 'tanh',kernel_regularizer=regularizers.l2(regPara)))
        model.add(Dropout(dropoutR))
        model.add(Dense(units = 180, activation = 'tanh'))
        model.add(Dropout(dropoutR))

        model.add(Dense(units = numClasses, activation = 'softmax'))

        if weightsPath is not None:
            model.load_weights(weightsPath)
    
        return model
