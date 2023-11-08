#!/usr/bin/env python
# coding: utf-8

# In[26]:


import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import random 
import numpy as np


# In[27]:


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)= mnist.load_data()


# In[28]:


#used to scale the pixel values to a range between 0 and 1.
#to make data consistent.

x_train= x_train/255
y_train= y_train/255


# In[40]:


#print('size of input image:', x_train[0].shape[0])
(x_train.shape)


# In[30]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"), #arbitary values
    keras.layers.Dense(10, activation="softmax"),
])

#Relu is used for hidden layer
#softmax for output layer


# In[31]:


model.summary()


# In[32]:


model.compile(optimizer="sgd",
loss="sparse_categorical_crossentropy", #measures dissimilarity in data
metrics=['accuracy']) #Tells the performance of our model


# In[33]:


history=model.fit(x_train,
y_train,validation_data=(x_test,y_test),epochs=10) 

#epochs are iterations


# In[34]:


test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f" %test_loss)
print("Accuracy=%.3f" %test_acc)

#testing loss and accuracy


# In[35]:


n=random.randint(0,9999) #to print random int value
plt.imshow(x_test[n]) #n is any random no. 
plt.show() 


# In[36]:


predicted_value=model.predict(x_test)
plt.imshow(x_test[n])
plt.show()

print(predicted_value[n])


# In[37]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[39]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# In[ ]:




