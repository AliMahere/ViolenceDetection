import cv2
import numpy as np
import tensorflow as tf
from os import path
import os

sample_rate = 5

protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

def get_ferture_units(video_path,output_path):

    base_file_name = path.basename(video_path)

    cap = cv2.VideoCapture(video_path)

    success, img = cap.read()

    width = img.shape[1]
    height = img.shape[0]

    inHeight = 368
    inWidth = int((inHeight / height) * width)
    inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    points = output[:, 0:19, :, :]
    points = tf.reduce_sum(points, 1)
    connections = output[:, 20:, :, :]
    connections = tf.reduce_sum(connections, 1)
    time_step = np.vstack((points, connections))
    time_step = np.reshape(time_step, [1,-1])
    sample = time_step


    sample_no = 0
    while success :

        inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                                        (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        points = output[:, 0:19, :, :]
        points = tf.reduce_sum(points, 1)
        connections = output[:, 20:, :, :]
        connections = tf.reduce_sum(connections, 1)
        time_step = np.vstack((points,connections))
        time_step = np.reshape(time_step,[1,-1])
        sample = np.vstack((sample, time_step))

        if sample.shape[0] % sample_rate == 0:

            np.save(output_path+'/'+base_file_name+'-'+ str(sample_no)+ ".npy" , np.array([sample]))

            sample_no +=1
            success, img = cap.read()
            inpBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (inWidth, inHeight),
                                            (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inpBlob)
            output = net.forward()

            points = output[:, 0:19, :, :]
            points = tf.reduce_sum(points, 1)
            connections = output[:, 20:, :, :]
            connections = tf.reduce_sum(connections, 1)
            time_step = np.vstack((points, connections))
            time_step = np.reshape(time_step, [1,-1])
            sample = time_step

        # read next frame
        success, img = cap.read()



def merge_dataset(dataset_path, output_path ):
    data_set = None
    for npy_file in os.listdir(dataset_path):
        if npy_file.endswith(".npy")  and data_set != None:
            data_sample = np.load(npy_file)
            data_set = np.vstack(data_set, data_sample)
        elif npy_file.endswith(".npy") and data_set == None:
            data_sample = np.load(npy_file)
            data_set = data_sample
        else:
            continue
    np.save(output_path + '/' "dataset.npy", data_set)


#Loop though vedios
videos_path = "HockeyFights/Non-Violence/"
for video in os.listdir(videos_path):
    if video.endswith(".avi") :
        get_ferture_units(videos_path+video, "output/nv")
        print(video)

    else:
        continue

