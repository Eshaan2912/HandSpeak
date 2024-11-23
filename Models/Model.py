#!/usr/bin/env python
# coding: utf-8

# ### Importing the Libraries

# In[3]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[4]:


tf.__version__ 


# ### Part 1 - Data Preprocessing

# #### Generating images for the Training set

# In[5]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# #### Generating images for the Test set

# In[6]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# ### Creating the Training set

# In[7]:


training_set = train_datagen.flow_from_directory('S:/Projects/Sign Language To Text Conversion/dataSet/trainingData',                                
                                                 target_size = (128, 128),
                                                 batch_size = 10,
                                                 color_mode = 'grayscale',                                
                                                 class_mode = 'categorical')


# In[8]:


test_set = test_datagen.flow_from_directory('S:/Projects/Sign Language To Text Conversion/dataSet/testingData',
                                            target_size = (128, 128),                                  
                                            batch_size = 10,        
                                            color_mode = 'grayscale',
                                            class_mode = 'categorical')


# ### Part 2 - Building the CNN

# #### Initializing the CNN

# In[9]:


classifier = tf.keras.models.Sequential()


# #### Step 1 - Convolution

# In[10]:


classifier.add(tf.keras.layers.Conv2D(filters=32,
                                     kernel_size=3, 
                                     padding="same", 
                                     activation="relu", 
                                     input_shape=[128, 128, 1]))


# #### Step 2 - Pooling

# In[11]:


classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, 
                                         strides=2, 
                                         padding='valid'))


# #### Adding a second convolutional layer

# In[12]:


classifier.add(tf.keras.layers.Conv2D(filters=32, 
                                      kernel_size=3, 
                                      padding="same", 
                                      activation="relu"))

classifier.add(tf.keras.layers.MaxPool2D(pool_size=2, 
                                         strides=2, 
                                         padding='valid'))


# #### Step 3 - Flattening

# In[13]:


classifier.add(tf.keras.layers.Flatten())


# #### Step 4 - Full Connection

# In[15]:


classifier.add(tf.keras.layers.Dense(units=128, 
                                     activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=96, activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.40))
classifier.add(tf.keras.layers.Dense(units=64, activation='relu'))
classifier.add(tf.keras.layers.Dense(units=27, activation='softmax')) # softmax for more than 2


# ### Part 3 - Training the CNN

# #### Compiling the CNN

# In[16]:


classifier.compile(optimizer = 'adam', 
                   loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])


# #### Training the CNN on the Training set and evaluating it on the Test set

# In[17]:


classifier.summary()


# In[25]:


classifier.fit(training_set,
                  epochs = 5,
                  validation_data = test_set)


# #### Saving the Model

# In[26]:


model_json = classifier.to_json()
with open("model_new.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model_new.h5')
print('Weights saved')


# In[ ]:




