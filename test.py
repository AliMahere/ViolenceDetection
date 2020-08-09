import cv2
import numpy as np
import tensorflow as tf
# protoFile = "pose/coco/pose_deploy_linevec.prototxt"
# weightsFile = "pose/coco/pose_iter_440000.caffemodel"
# image = cv2.imread("a.png")
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
# width = image.shape[1]
# height = image.shape[0]
# inHeight = 368
# inWidth = int((inHeight / height) * width)
#
# inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (inWidth, inHeight),
#                                 (0, 0, 0), swapRB=False, crop=False)
# net.setInput(inpBlob)
# output = net.forward()
# print(output.shape)
#
# points = output[:,0:19,:,:]
# points = tf.reduce_sum(points, 1)
# print("points ", points.shape)
# connections = output[:,20:,:,:]
# connections = tf.reduce_sum(connections, 1)
# #ft = tf.reshape([connections,points] , [-1])
# ft = np.vstack((points,connections))
# ft = np.reshape(ft,[-1])
# print("connections ", connections.shape)
# print("ft shape ",ft.shape)
# print("ft type ", type(ft))

npy1 = np.load("output/fi1_xvid.avi-0.npy")
print("npy1.shape",npy1.shape)
npy2 = np.load("output/fi1_xvid.avi-1.npy")

print("npy2.shape",npy2.shape)
merged = npy1
merged = np.vstack((merged,npy2))
print("merged.shape",merged.shape)

