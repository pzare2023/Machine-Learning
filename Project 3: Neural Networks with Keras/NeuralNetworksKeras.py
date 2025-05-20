
import matplotlib.pyplot as plt
from tensorflow import keras 
import numpy as np

"""
PART 1: Implementing a neural network using Keras

"""

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 


plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray') 
    plt.axis('off')  
plt.show()



# Scale the input feature down to 0-1 values, by dividing them by 255.0 
x_train = x_train / 255.0 
x_test = x_test / 255.0 


model = keras.Sequential() 


model.add(keras.layers.Flatten(input_shape=[28, 28])) 

# Build the first hidden layer to the model. 

model.add(keras.layers.Dense(300, activation= 'relu')) 

# Build a second hidden layer to the model. 
# For this, use a 'Dense layer' with 100 neurons, also using the ReLU activation function.
model.add(keras.layers.Dense(100, activation= 'relu')) 

# Build an output layer to the model.

model.add(keras.layers.Dense(10, activation= 'softmax')) 



model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 



model.fit(x_train, y_train, epochs=20) 


# Testing the model 
plt.close('all')
y_pred = model.predict(x_test[0:10]) 
plt.figure(figsize=(5,2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title('Predicted label: ' + str(np.argmax(y_pred[i]))) 
    plt.imshow(x_test[i], cmap='gray') 
    plt.axis('off')  
plt.show()


