import argparse
import Models, LoadBatches
import glob
import cv2
import numpy as np
import random
import os
import keras

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--epoch_number", type=int, default=5)
parser.add_argument("--test_images", type=str, default="")
parser.add_argument("--output_path", type=str, default="")
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--n_classes", type=int)
parser.add_argument("--data_format", type=str, default="channels_first")

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width = args.input_width
input_height = args.input_height
epoch_number = args.epoch_number
data_format = args.data_format

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}
if model_name:
    modelFN = modelFns[model_name]

    m = modelFN(n_classes, input_height=input_height, input_width=input_width, data_format=data_format)
else:
    print("load model: " + args.save_weights_path + ".model." + str(epoch_number))
    m = keras.models.load_model(args.save_weights_path + ".model." + str(epoch_number))
m.load_weights(args.save_weights_path + "." + str(epoch_number))
m.compile(loss='categorical_crossentropy',
          optimizer='adadelta',
          metrics=['accuracy'])

if model_name:
    output_height = m.outputHeight
    output_width = m.outputWidth
else:
    output_height = input_height
    output_width = input_width

images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
images.sort()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(n_classes)]

for imgName in images:
    outPath = imgName.replace(imgName.split("/")[len(imgName.split("/"))-1], args.output_path)
    X = LoadBatches.getImageArr(path=imgName, width=args.input_width, height=args.input_height, odering=data_format)
    pr = m.predict(np.array([X]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
    seg_img = cv2.resize(seg_img, (input_width, input_height))
    print("write image: " +os.path.join(outPath, os.path.basename(imgName)))
    cv2.imwrite(os.path.join(outPath, os.path.basename(imgName)), seg_img)
