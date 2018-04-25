import argparse
import Models, LoadBatches
import matplotlib.pyplot as plt


def plot_accuracy(f, history, from_list=False, accuraccy='categorical_accuracy'):
    """
    :param f: output file
    :param history: list of accuracy from keras model
    :param from_list: true => history gives as list, false => history given from model.fit_generator
    :return:
    """
    if not from_list:
        print(history.history.keys())
        print("accuracy file: " + f)

        ca = history.history[accuraccy]
        va = history.history['val_' + accuraccy]
    else:
        ca = history[0]
        va = history[1]

    # summarize history for accuracy
    plt.plot(ca)
    plt.plot(va)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f)
    plt.clf()


def plot_loss(f, history, from_list=False):
    """
    :param f: output file
    :param history: list of loss from keras model
    :param from_list: true => history gives as list, false => history given from model.fit_generator
    :return: None
    """
    print("loss file: " + f)
    if not from_list:
        l = history.history['loss']
        vl = history.history['val_loss']
    else:
        l = history[0]
        vl = history[1]
    # summarize history for loss
    plt.plot(l)
    plt.plot(vl)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(f)
    plt.clf()


parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type=str)
parser.add_argument("--n_train_images", type=int)
parser.add_argument("--train_images", type=str)
parser.add_argument("--train_annotations", type=str)
parser.add_argument("--n_classes", type=int)
parser.add_argument("--input_height", type=int, default=224)
parser.add_argument("--input_width", type=int, default=224)

parser.add_argument('--validate', action='store_false')
parser.add_argument("--n_val_images", type=int)
parser.add_argument("--val_images", type=str, default="")
parser.add_argument("--val_annotations", type=str, default="")

parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--val_batch_size", type=int, default=64)
parser.add_argument("--load_weights", type=str, default="")

parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--optimizer_name", type=str, default="adadelta")
parser.add_argument("--data_format", type=str, default="channels_first")
parser.add_argument("--preprocess_annotations", type=int, default=0)

args = parser.parse_args()

n_train_images = args.n_train_images
n_val_images = args.n_val_images
train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights
data_format = args.data_format
preprocess_annotations = bool(args.preprocess_annotations)

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
    val_images_path = args.val_images
    val_segs_path = args.val_annotations
    val_batch_size = args.val_batch_size

modelFns = {'vgg_segnet': Models.VGGSegnet.VGGSegnet, 'vgg_unet': Models.VGGUnet.VGGUnet,
            'vgg_unet2': Models.VGGUnet.VGGUnet2, 'fcn8': Models.FCN8.FCN8, 'fcn32': Models.FCN32.FCN32}
modelFN = modelFns[model_name]

m = modelFN(n_classes, input_height=input_height, input_width=input_width, data_format=data_format)
m.compile(loss='categorical_crossentropy',
          optimizer=optimizer_name,
          metrics=['accuracy'])

if len(load_weights) > 0:
    m.load_weights(load_weights)

print("Model output shape", m.output_shape)

output_height = m.outputHeight
output_width = m.outputWidth

G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes, data_format,
                                           input_height, input_width, output_height, output_width,
                                           preprocess_annotations)

if validate:
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, data_format,
                                                input_height,
                                                input_width, output_height, output_width, preprocess_annotations)

if not validate:
    for ep in range(epochs):
        m.fit_generator(G, 512, epochs=1, verbose=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
else:
    acc = []
    loss = []
    val_acc = []
    val_loss = []
    for ep in range(epochs):
        history = m.fit_generator(G, steps_per_epoch=512, validation_data=G2, validation_steps=200, epochs=1, verbose=1)
        acc.append(history.history['acc'])
        loss.append(history.history['loss'])
        val_acc.append(history.history['val_acc'])
        val_loss.append(history.history['val_loss'])
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
    plot_accuracy(save_weights_path + '.accuracy.png', [acc, val_acc], True, 'acc')
    plot_loss(save_weights_path + '.loss.png', [loss, val_loss], True)

