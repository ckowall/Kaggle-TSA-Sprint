
# coding: utf-8

# In[1]:


import numpy as np
from random import shuffle
import pandas as pd
import os
import cv2
import scipy.stats as stats
import csv
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


# In[2]:


NUM_ZONES = 17
zone1 = [[141, 45], [141, 45], [141, 45], None, None, None, None, [141, 340], [141, 331], [141, 331], [150, 300], [150, 300], [140, 200], [140, 200], [140, 100], [140, 100]] #right upper arm
zone2 = [[0, 0], [0, 0], [0, 100], None, None, None, None, [0, 340], [0, 331], [0, 331], [0, 331], [0, 300], [0, 200], [0, 200], [0, 120], [0, 100]] #right lower arm
zone3 = [[141, 331], [141, 331], [150, 300], [150, 300], [140, 200], [140, 200], [140, 100], [140, 100], [141, 45], [141, 45], [141, 45], None, None, None, None, [141, 340]] #left upper arm
zone4 = [[0, 331], [0, 331], [0, 331], [0, 300], [0, 200], [0, 200], [0, 120], [0, 100], [0, 0], [0, 0], [0, 100], None, None, None, None, [0, 340]] #left lower arm
zone5 = [[208, 180], [208, 160], [200, 200], None, None, None, None, None, None, None, None, None, [200, 300], [200, 300], [200, 200], [200, 200]] #chest
zone6 = [[280, 104], [280, 104], None, None, None, None, None, [280, 260], [280, 265], [280, 265], [280, 200], [280, 200], [250, 200], [250, 200], [250, 200], [250, 140]] #right belly
zone7 = [[280, 265], [280, 265], [280, 200], [280, 200], [250, 200], [250, 200], [250, 200], [250, 140], [280, 104], [280, 104], None, None, None, None, None, [280, 260]] #left belly
zone8 = [[360, 80], [360, 80], None, None, None, None, None, [360, 280], [360, 300], [360, 300], [360, 260], [360, 200], [340, 200], [340, 200], [330, 200], [330, 150]] #right hip
zone9 = [[360, 180], [360, 180], [360, 200], [360, 200], None, [360, 260], [350, 250], [340, 200], [360, 180], [360, 180], [360, 200], [360, 200], None, [360, 260], [350, 250], [340, 200]] #crotch
zone10 = [[360, 300], [360, 300], [360, 260], [360, 200], [340, 200], [340, 200], [330, 200], [330, 150], [360, 80], [360, 80], None, None, None, None, None, [360, 280]] #left hip
zone11 = [[450, 100], [450, 100], [450, 150], [450, 200], [450, 200], [450, 250], [450, 300], [450, 300], [450, 260], [450, 200], [450, 180], [450, 150]] #right knee
zone12 = [[450, 260], [450, 260], [450, 260], [450, 260], [450, 200], [450, 180], [450, 150], [450, 100], [450, 100], [450, 150], [450, 200], [450, 200], [450, 250], [450, 300]] #left knee
zone13 = [[500, 100], [500, 100], [500, 150], [500, 200], None, [500, 300], [500, 300], [500, 300], [500, 260], [500, 260], [500, 260], [500, 260], [500, 200], [500, 180], [500, 150], [500, 100]] #right calf
zone14 = [[500, 260], [500, 260], [500, 260], [500, 260], [500, 200], [500, 200], [500, 180], [500, 150], [500, 100], [500, 100], [500, 150], [500, 200], None, [500, 300], [500, 300], [500, 300]] #left calf
zone15 = [[600, 100], [600, 100], [600, 150], [600, 200], None, [600, 200], [600, 300], [600, 150], [600, 260], [600, 260], [600, 300], [600, 300], [600, 200], [600, 200], [600, 180], [600, 150]] #right foot
zone16 = [[600, 260], [600, 260], [600, 300], [600, 300], [600, 200], [600, 200], [600, 180], [600, 150], [600, 100], [600, 100], [600, 150], [600, 200], None, [600, 200], [600, 300], [600, 150]] #left foot
zone17 = [None, None, None, None, [200, 300], [200, 300], [200, 200], [200, 200], [208, 180], [208, 160], [200, 200], None, None, None, None, None] #back
zones = [zone1, zone2, zone3, zone4, zone5, zone6, zone7, zone8, zone9, zone10,
         zone11, zone12, zone13, zone14, zone15, zone16, zone17]

#for i in range(len(zones)):
#    print("length of zone " + repr(i) + " " + repr(len(zones[i])))
# In[3]:


BROAD_MODULE_FOLDER = '/home/correy/Desktop/broad/'
STAGE1_LABELS = BROAD_MODULE_FOLDER + '/stage1_labels.csv'
APS_LOC = BROAD_MODULE_FOLDER + 'aps/'
APS_FILE_NAME = APS_LOC + '00360f79fd6e02781457eda48f85da90.aps'
TRAIN_LOC = BROAD_MODULE_FOLDER + 'train/'
TEST_LOC = BROAD_MODULE_FOLDER + 'test/'
NPY_FILE_NAME = TRAIN_LOC + '00360f79fd6e02781457eda48f85da90.npy'
MOD_LOC = BROAD_MODULE_FOLDER + 'modules/'
IMAGE_DIM = 161
NUM_VIEWS_PER_APS = 16
NUM_EPOCHS = 10
CYCLES = 10
COLORMAP = 'pink'

NUM_MODULES = 12

LEARNING_RATE = 1e-3

TRAIN_PATH = '/home/correy/Desktop/network/train/'
MODEL_PATH = '/home/correy/Desktop/network/model/'
MODEL_NAME = ('tsa-{}-lr-{}-{}-{}-tz'.format('alexnet-v0.1', LEARNING_RATE, IMAGE_DIM, 
                                                IMAGE_DIM ))


# In[4]:


def get_negative_subjects(zone_num):
    zone_index = {0: 'Zone1', 1: 'Zone2', 2: 'Zone3', 3: 'Zone4', 4: 'Zone5', 5: 'Zone6', 
                  6: 'Zone7', 7: 'Zone8', 8: 'Zone9', 9: 'Zone10', 10: 'Zone11', 11: 'Zone12', 
                  12: 'Zone13', 13: 'Zone14', 14: 'Zone15', 15: 'Zone16',
                  16: 'Zone17'
                 }
    negative_subjects = []
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Subject', 'Zone', 'Probability']]
    #print("This is the df['Subject'] \n" + repr(df['Subject']))
    subjects = df['Subject'].unique()
    #print("number of subjects " + repr(len(subjects)) )
    for subject in subjects:
        query_string = 'Subject==' + repr(subject)
        sub_df = df.query(query_string)
        key = zone_index.get(zone_num)
        if sub_df.loc[sub_df['Zone'] == key]['Probability'].values[0] == 0:
            negative_subjects.append(subject)
    return negative_subjects


# In[5]:


def get_positive_subjects(zone_num):
    zone_index = {0: 'Zone1', 1: 'Zone2', 2: 'Zone3', 3: 'Zone4', 4: 'Zone5', 5: 'Zone6', 
                  6: 'Zone7', 7: 'Zone8', 8: 'Zone9', 9: 'Zone10', 10: 'Zone11', 11: 'Zone12', 
                  12: 'Zone13', 13: 'Zone14', 14: 'Zone15', 15: 'Zone16',
                  16: 'Zone17'
                 }
    positive_subjects = []
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Subject', 'Zone', 'Probability']]
    #print("This is the df['Subject'] \n" + repr(df['Subject']))
    subjects = df['Subject'].unique()
    #print("number of subjects " + repr(len(subjects)) )
    for subject in subjects:
        query_string = 'Subject==' + repr(subject)
        sub_df = df.query(query_string)
        key = zone_index.get(zone_num)
        if sub_df.loc[sub_df['Zone'] == key]['Probability'].values[0] == 1:
            positive_subjects.append(subject)
    return positive_subjects


# In[6]:


def get_submission_subjects():
    df = pd.read_csv(STAGE1_LABELS)
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    LABELED_SUBJECT_LIST = df['Subject'].unique()
    SUBJECT_LIST = [os.path.splitext(subject)[0] for subject in os.listdir(REST_LOC)]
    
    submission_list = list(set(SUBJECT_LIST) - set(LABELED_SUBJECT_LIST))

    return submission_list


# In[7]:


def read_header(infile):
    # declare dictionary
    h = dict()
    
    with open(infile, 'r+b') as fid:

        h['filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['parent_filename'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 20))
        h['comments1'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['comments2'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 80))
        h['energy_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['config_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['file_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['trans_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['date_modified'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 16))
        h['frequency'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['mat_velocity'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_polarization_channels'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare00'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['adc_min_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_max_voltage'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['band_width'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare01'] = np.fromfile(fid, dtype = np.int16, count = 5)
        h['polarization_type'] = np.fromfile(fid, dtype = np.int16, count = 4)
        h['record_header_size'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['word_precision'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['min_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['max_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['avg_data_value'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_scale_factor'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['data_units'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['surf_removal'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['edge_weighting'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['x_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['y_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['z_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['t_units'] = np.fromfile(fid, dtype = np.uint16, count = 1)
        h['spare02'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_return_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['scan_orientation'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scan_direction'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['data_storage_order'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['x_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_inc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['num_x_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_y_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_z_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['num_t_pts'] = np.fromfile(fid, dtype = np.int32, count = 1)
        h['x_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_speed'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_acc'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_motor_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_encoder_res'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['date_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['time_processed'] = b''.join(np.fromfile(fid, dtype = 'S1', count = 8))
        h['depth_recon'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['elevation_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['roll_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_max_travel'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['azimuth_offset_angle'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['adc_type'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['spare06'] = np.fromfile(fid, dtype = np.int16, count = 1)
        h['scanner_radius'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['x_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['y_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['z_offset'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['t_delay'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_start'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['range_gate_end'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['ahis_software_version'] = np.fromfile(fid, dtype = np.float32, count = 1)
        h['spare_end'] = np.fromfile(fid, dtype = np.float32, count = 10)

    return h


# In[8]:


def read_data(infile):
    
    # read in header and get dimensions
    h = read_header(infile)
    nx = int(h['num_x_pts'])
    ny = int(h['num_y_pts'])
    nt = int(h['num_t_pts'])
    
    extension = os.path.splitext(infile)[1]
    
    with open(infile, 'rb') as fid:
          
        # skip the header
        fid.seek(512) 

        # handle .aps and .a3aps files
        if extension == '.aps' or extension == '.a3daps':
        
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)

            elif(h['word_type']==4): 
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor'] 
            data = data.reshape(nx, ny, nt, order='F').copy()

        # handle .a3d files
        elif extension == '.a3d':
              
            if(h['word_type']==7):
                data = np.fromfile(fid, dtype = np.float32, count = nx * ny * nt)
                
            elif(h['word_type']==4):
                data = np.fromfile(fid, dtype = np.uint16, count = nx * ny * nt)

            # scale and reshape the data
            data = data * h['data_scale_factor']
            data = data.reshape(nx, nt, ny, order='F').copy() 
            
        # handle .ahi files
        elif extension == '.ahi':
            data = np.fromfile(fid, dtype = np.float32, count = 2 * nx * ny * nt)
            data = data.reshape(2, ny, nx, nt, order='F').copy()
            real = data[0,:,:,:].copy()
            imag = data[1,:,:,:].copy()

        if extension != '.ahi':
            return data
        else:
            return real, imag


# In[9]:


def convert_to_grayscale(img):
    # scale pixel values to grayscale
    base_range = np.amax(img) - np.amin(img)
    rescaled_range = 255 - 0
    img_rescaled = (((img - np.amin(img)) * rescaled_range) / base_range)

    return np.uint8(img_rescaled)


# In[10]:


def normalize(image):
    MIN_BOUND = 0.0
    MAX_BOUND = 255.0
    
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


# In[11]:


def zero_center(image):
     
    PIXEL_MEAN = 0.014327
    
    image = image - PIXEL_MEAN
    return image


# In[12]:


def get_single_aps_image(infile, nth_image):

    # read in the aps file, it comes in as shape(512, 620, 16)
    img = read_data(infile)
    
    # transpose so that the slice is the first dimension shape(16, 620, 512)
    img = img.transpose()
    
    img = img[nth_image]
    
    img = convert_to_grayscale(img)
    img = normalize(img)
    img = zero_center(img)
    #img = np.rot90(img)
    
    return np.flipud(img)


# In[13]:


def pad_to(image, size):
    x_size, y_size = image.shape
    print(" in pad to image shape ", repr(image.shape))
    x_pad = size - x_size
    y_pad = y_size - size
    if x_pad >= 0 and y_size >= 0:
        x_padding = np.zeros((x_pad, y_size), dtype=image.dtype)
        padded_array = np.concatenate((image, x_padding), axis=0)
    else:
        padded_array = image
    if size >= 0 and y_pad > 0:
        y_padding = np.zeros((size, y_pad), dtype=image.dtype)
        print(" SHape of padding " + repr(y_padding.shape), " shape of padded array " + repr(padded_array.shape))
        padded_array = np.concatenate((padded_array, y_padding), axis=1)
    else:
        print(" y pad is negative " + repr(y_pad))
    return padded_array


# In[14]:


def copy_crop(image, x_start, y_start):
    image_copy = image.copy()
    length, width = image.shape
    if length < y_start + IMAGE_DIM:
        image_copy = image_copy[y_start: y_start + IMAGE_DIM, x_start: x_start + IMAGE_DIM]
        print("padded out" + repr(image_copy.shape))
        return pad_to(image_copy, IMAGE_DIM)
    else:
        return image_copy[y_start: y_start + IMAGE_DIM, x_start: x_start + IMAGE_DIM]


# In[15]:


def create_broad_folders(start):
    for i in range(NUM_ZONES):
        if i >= start:
            os.mkdir(TRAIN_LOC + repr(i))
            positive_subjects = get_positive_subjects(i)
            negative_subjects = get_negative_subjects(i)
            for j in range(len(positive_subjects)):
                subject = positive_subjects[j]
                targets = zones[i]
                infile = APS_LOC + subject + '.aps'
                for k in range(len(targets)):
                    if targets[k] != None:
                        img = get_single_aps_image(infile, k)
                        resized_image = copy_crop(img, targets[k][1], targets[k][0])
                        target_file_name = TRAIN_LOC + repr(i) + '/' + subject + '_' + repr(k) + '.npy'
                        np.save(target_file_name, resized_image)
                        print("creating file " + target_file_name + " shape " + repr(resized_image.shape))
                        min_dif = 100000
                        min_neg = np.array([])
                        for l in range(len(negative_subjects)):
                            neg_name = negative_subjects[l]
                            neg_path = APS_LOC + neg_name + '.aps'
                            img = get_single_aps_image(neg_path, k)
                            neg_image = copy_crop(img, targets[k][1], targets[k][0])
                            dif_image = np.subtract(resized_image, neg_image)
                            dif_image = np.absolute(dif_image)
                            dif = np.sum(dif_image)
                            if dif < min_dif:
                                min_dif = dif
                                min_neg = neg_image
                        neg_name = TRAIN_LOC + repr(i) + '/' + subject + '_neg_' + repr(k) + '.npy'
                        np.save(neg_name, min_neg)
                        print("creating file " + neg_name + " shape " + repr(min_neg.shape))
# Unit test
create_broad_folders(12)


# In[ ]:


def create_test_folder():
    submission_subjects = get_submission_subjects()
    for subject in submission_subjects:
        infile = APS_LOC + subject + '.aps'
        for i in range(NUM_ZONES):
            os.mkdir(TEST_LOC + repr(i))
            targets = zones[i]
            for k in range(len(targets)):
                if targets[k] != None:
                    img = get_single_aps_image(infile, k)
                    resized_image = copy_crop(img, targets[k][1], targets[k][0])
                    target_file_name = TEST_LOC + repr(i) + '/' + subject + '_' + repr(i) + '_' + repr(k) + '.npy'
                    np.save(target_file_name, resized_image)
                    print("creating file " + target_file_name + " shape " + repr(resized_image.shape))
# Unit test
#create_test_folder()


# In[16]:


def conv_net(width, height, lr, my_name):
    network = input_data(shape=[None, width, height, 1], name='features')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum', loss='categorical_crossentropy', 
                         learning_rate=lr, name='labels')

    model = tflearn.DNN(network, checkpoint_path=my_name, tensorboard_dir=TRAIN_PATH, tensorboard_verbose=0, max_checkpoints=1)
    #model = tflearn.DNN(network, checkpoint_path=name, 
    #                    tensorboard_dir=TRAIN_PATH, tensorboard_verbose=0, max_checkpoints=1)
    #tensorboard_verbose=3
    return model


# In[17]:


def subject_drill(subjects, zone, member, train_or_test):
    positive_label = [1.0, 0.0]
    negative_label = [0.0, 1.0]
    my_foci = zones[zone]
    focus = my_foci[member]
    horiz = focus[1]
    vert = focus[0]
    x_start = horiz * IMAGE_DIM
    y_start = vert * IMAGE_DIM
    #print(" in test drill " + repr(len(subjects)))
    vignettes = []
    vignettes = np.asarray(vignettes, dtype=np.float32)
    labels = []
    first = 1
    for i in range(len(subjects)):
        if i == 0:
            labels.append(positive_label)
        else:
            if i % 2 == 0:
                labels.append(positive_label)
            else:
                labels.append(negative_label)
        subject = str(subjects[i])
        if train_or_test == 0:
            file_string = TEST_LOC + subject + '.aps'
        else:
            file_string = TRAIN_LOC + subject + '.npy'
        #print("file string for image " + file_string)
        whole_image = np.load(file_string)
        partial_image = copy_crop(whole_image, x_start, y_start)
        if first == 1:
            vignettes = partial_image
            first = 0
        else:
            vignettes = np.concatenate((vignettes, partial_image), axis=0)
    vignettes = vignettes.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1) 
    labels = np.asarray(labels, np.float32)
    return vignettes, labels


# In[18]:


def subject_zone_drill(subject, zone, train_test):
    my_foci = zones[zone]
    #print(" in test drill " + repr(len(subjects)))
    vignettes = []
    vignettes = np.asarray(vignettes, dtype=np.float32)
    first = 1
    for i in range(NUM_VIEWS_PER_APS):
        if my_foci[i] != None:
            if train_test == 1:
                target_file_name = TRAIN_LOC + repr(zone) + '/' + subject + '_' + repr(i) + '.npy'
            else:
                target_file_name = TEST_LOC + repr(zone) + '/' + subject + '_' + repr(zone) + '_' + repr(i) + '.npy'
            partial_image = np.load(target_file_name)
            if first == 1:
                vignettes = partial_image
                first = 0
            else:
                vignettes = np.concatenate((vignettes, partial_image), axis=0)
    vignettes = vignettes.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1) 
    return vignettes


# In[19]:


def element_fetch(subject, zone, member):
    target_file_name = TRAIN_LOC + repr(zone) + '/' + subject + '_' + repr(member) + '.npy'
    return np.load(target_file_name)


# In[20]:


def neg_element_fetch(subject, zone, member):
    target_file_name = TRAIN_LOC + repr(zone) + '/' + subject + '_neg_' + repr(member) + '.npy'
    return np.load(target_file_name)


# In[ ]:


def team_train(reload_modules, upper):
    #use predict to assign data set members to specialists 
    #NUM_ZONES
    for i in range(upper):
        num_members = len(zones[i])
        positive_subjects = get_positive_subjects(i)
        train_list_size = int(len(positive_subjects) * 0.98)
        holdout_list_size = int(len(positive_subjects) * 0.02)
        holdout_subjects = positive_subjects[train_list_size:]
        positive_subjects = positive_subjects[:-holdout_list_size]
        confidences = []
        training_stacks = []
        label_stacks = []
        for j in range(len(positive_subjects)):
            subject = positive_subjects[j]
            vignettes = subject_zone_drill(subject, i, 1)
            subject_confidences = []
            for k in range(NUM_MODULES):
                tf.reset_default_graph()
                run_name = repr(k) + '/final_' 
                model = conv_net(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE, run_name)
                if reload_modules == 1:
                    model.load(MOD_LOC + run_name + '.tfl', weights_only=True) 
                predictions = model.predict(vignettes)
                print("In module " + repr(k) + " preds " + repr(predictions.shape))
                subject_confidences.append(predictions)
                if j == 0:
                    training_stacks.append([])
            mod_max = -1
            mem_max = -1
            max_val = -1.0
            for k in range(NUM_MODULES):
                m = 0
                for l in range(NUM_VIEWS_PER_APS):
                    print(" shape of subject confidences " + repr(len(subject_confidences)) + " other " + repr(len(subject_confidences[0])))
                    if zones[i][l] != None:
                        if subject_confidences[k][m][0] > max_val:
                            mod_max = k
                            mem_max = l
                            max_val = subject_confidences[k][m][0]
                        m += 1
            entry = []
            entry.append(subject)
            entry.append(mem_max)
            training_stacks[mod_max].append(entry)
        for j in range(NUM_MODULES):            
            member_stack = training_stacks[j]
            features = np.array([])
            labels = []
            empty = 1
            print("length of member stack " + repr(len(member_stack)))
            for k in range(len(member_stack)):
                entry = member_stack[k]
                subject = entry[0]
                member = entry[1]
                partial_features = element_fetch(subject, i, member)
                if empty == 1:
                    features = partial_features
                    labels.append([1.0, 0.0])
                    partial_features = neg_element_fetch(subject, i, member)
                    features = np.concatenate((features, partial_features), axis=0)
                    labels.append([0.0, 1.0])
                    empty = 0
                else:
                    features = np.concatenate((features, partial_features), axis=0)
                    labels.append([1.0, 0.0])
                    features = np.concatenate((features, partial_features), axis=0)
                    labels.append([0.0, 1.0]) 
            features = features.reshape(-1, IMAGE_DIM, IMAGE_DIM, 1) 
            labels = np.asarray(labels, np.float32)
            print("length of features " + repr(len(features)))
            holdouts = features
            holdout_labels = labels
            print("length of labels " + repr(len(labels)))
            print("labels " + repr(len(labels)))
            if(len(labels) > 0):
                tf.reset_default_graph()
                run_name = repr(j) + '/final_'
                model = conv_net(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE, run_name)
                if reload_modules == 1:
                    model.load(MOD_LOC + run_name + '.tfl', weights_only=True) 
                for k in range(NUM_EPOCHS):
                #this needs fixed the holdouts are disabled
                    model.fit({'features': features}, {'labels': labels}, n_epoch=1, 
                              validation_set=({'features': holdouts}, {'labels': holdout_labels}), 
                              shuffle=True, snapshot_step=None, show_metric=True, run_id=run_name)
                model.save(MOD_LOC + run_name + '.tfl')    
# Unit test
#for i in range(CYCLES):
#    team_train(1, 6)


# In[ ]:


def create_submission():
    subjects = get_submission_subjects()
    num_subjects = len(subjects)
    test_results = []
    for i in range(num_subjects):
        zone_results = []
        for j in range(NUM_ZONES):
            zone_results.append(0.0)
        test_results.append(zone_results)
    for i in range(NUM_ZONES):     
        for j in range(len(subjects)):
            subject = subjects[j]
            vignettes = subject_zone_drill(subject, i, 0)
            if j == 0:
                features = vignettes
            else:
                features = np.concatenate((features, vignettes), axis=0)
        for k in range(NUM_MODULES):
            run_name = repr(j) + '/final_'
            tf.reset_default_graph()
            model = conv_net(IMAGE_DIM, IMAGE_DIM, LEARNING_RATE, run_name)
            model.load(MOD_LOC + run_name + '.tfl', weights_only=True)
            temp_test_labels = model.predict(features)
            for k in range(len(temp_test_labels)):
                if temp_test_labels[k][0] > test_results[k][i]:
                    test_results[k][i] = temp_test_labels[k][0]
    np.save('/home/correy/Desktop/pitstop.npy', test_results)
    #test_results = analyse_top(test_results, len(subjects))
    create_output_file(test_results, subjects)


# In[ ]:


def create_output_file(probs, subjects):
    with open('/home/correy/Desktop/stage1_submission.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        entry1 = 'Id'
        entry2 = 'Probability'
        filewriter.writerow([entry1, entry2])
        for i in range(len(subjects)):
            for j in range(len(probs[0])):
                entry1 = subjects[i] + "_Zone" + repr(j+1)
                entry2 = probs[i][j]
                filewriter.writerow([entry1, entry2])

