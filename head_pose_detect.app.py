import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import Average, Input
import fsanet

# from headlib.FSANET_model import *


# Props.
weight_file1 = 'head_pose_fsanet/fsanet_capsule_3_16_2_21_5.h5'
weight_file2 = 'head_pose_fsanet/fsanet_var_capsule_3_16_2_21_5.h5'
weight_file3 = 'head_pose_fsanet/fsanet_noS_capsule_3_16_2_192_5.h5'
face_proto_file = 'head_pose_fsanet/deploy.prototxt'
face_model_file = 'head_pose_fsanet/res10_300x300_ssd_iter_140000.caffemodel'
image_out = True

# Model variables.
model1 = None
model2 = None
model3 = None
model = None
net = None

# load model and weights
img_size = 64
ad = 0.6


def on_set(k, v):
    if k == 'weight_file1':
        global weight_file1
        weight_file1 = v
    elif k == 'weight_file2':
        global weight_file2
        weight_file2 = v
    elif k == 'weight_file3':
        global weight_file3
        weight_file3 = v
    elif k == 'face_proto_file':
        global face_proto_file
        face_proto_file = v
    elif k == 'face_model_file':
        global face_model_file
        face_model_file = v
    elif k == 'image_out':
        global image_out
        image_out = bool(v)


def on_get(k):
    if k == 'weight_file1':
        return weight_file1
    elif k == 'weight_file2':
        return weight_file2
    elif k == 'weight_file3':
        return weight_file3
    elif k == 'face_proto_file':
        return face_proto_file
    elif k == 'face_model_file':
        return face_model_file
    elif k == 'image_out':
        return str(v)


def on_init():
    global model1
    global model2
    global model3
    global model
    global net

    # Gpu setting.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            sys.stdout.write(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs\n")
            sys.stdout.flush()
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            sys.stdout.write(e)
            sys.stdout.flush()

    # Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3, 3, 3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5

    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = fsanet.FSA_net_Capsule(image_size, num_classes,
                             stage_num, lambda_d, S_set)()
    model2 = fsanet.FSA_net_Var_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = fsanet.FSA_net_noS_Capsule(
        image_size, num_classes, stage_num, lambda_d, S_set)()

    sys.stdout.write('Loading models ...')
    sys.stdout.flush()

    model1.load_weights(weight_file1)
    sys.stdout.write('Finished loading model 1.')
    sys.stdout.flush()

    model2.load_weights(weight_file2)
    sys.stdout.write('Finished loading model 2.')
    sys.stdout.flush()

    model3.load_weights(weight_file3)
    sys.stdout.write('Finished loading model 3.')
    sys.stdout.flush()

    inputs = Input(shape=(64, 64, 3))
    x1 = model1(inputs)  # 1x1
    x2 = model2(inputs)  # var
    x3 = model3(inputs)  # w/o
    avg_model = Average()([x1, x2, x3])
    model = Model(inputs=inputs, outputs=avg_model)

    # load our serialized face detector from disk
    sys.stdout.write('[INFO] loading face detector...')
    sys.stdout.flush()
    net = cv2.dnn.readNetFromCaffe(face_proto_file, face_model_file)

    return True


def on_run(image):

    img_h, img_w, _ = np.shape(image)

    heads, draw_image = fsanet.predict_head_pose(
        image, ad, img_size, img_w, img_h, model, net, image_out=image_out)

    return {
        'draw_image': draw_image,
        'heads': np.array(heads)
    }

