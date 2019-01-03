
# coding: utf-8

# In[1]:


#nvcc --version


# In[2]:


import tensorflow as tf
tf.__version__


# In[3]:


import os
os.getcwd()


# In[4]:


tf.test.gpu_device_name()


# In[5]:


import deeplabcut


# In[6]:


#create the project, set working directory for videos, and find videos
deeplabcut.create_new_project('CeVICHE','Rachel StClair', ['/home/mpcr/Desktop/Rachel/CeVICHE/avi/001/ceviche-001.avi'], working_directory='/home/mpcr/Desktop/Rachel/CeVICHE',copy_videos=False)


# In[7]:


path_config_file = '/home/mpcr/Desktop/Rachel/CeVICHE/CeVICHE-Rachel StClair-2018-12-06/config.yaml'
os.stat(path_config_file)


# In[8]:


#data selection
deeplabcut.extract_frames(path_config_file, 'automatic', 'uniform')


# In[9]:


videofile_path = ['/home/mpcr/Desktop/Rachel/CeVICHE/CeVICHE-Rachel StClair-2018-12-06/config.yaml'] #Enter the list of videos to analyze.


# In[10]:


deeplabcut.label_frames(path_config_file)


# In[11]:


deeplabcut.create_training_dataset(path_config_file, Shuffles=1)

