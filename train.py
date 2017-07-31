"""Training script for the DeepLab-LargeFOV network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC dataset,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time,shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

import math

from deeplab_lfov import DeepLabLFOVModel, ImageReader, decode_labels

BATCH_SIZE = 6
DATA_DIRECTORY = './dataset/class3_def/' #'/home/VOCdevkit'

DATA_TRAIN_LIST_PATH = DATA_DIRECTORY+'/train.txt'
DATA_VAL_LIST_PATH = DATA_DIRECTORY+'/test.txt'

INPUT_SIZE = '505,505'
LEARNING_RATE = 0.0001
MEAN_IMG = tf.Variable(np.array((175,175,175)), trainable=False, dtype=tf.float32)
NUM_STEPS = 20000
RANDOM_SCALE = False  #True
RESTORE_FROM = None   #'./deeplab_lfov.ckpt'
SAVE_DIR = './images/'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots/'
WEIGHTS_PATH   = './util/net_weights.ckpt'
LOG_DIR = './log'

VAL_LOOP = int(math.ceil(29/BATCH_SIZE))

IMG_MEAN = np.array((175,175,175), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data_dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data_train_list", type=str, default=DATA_TRAIN_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data_val_list", type=str, default=DATA_VAL_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input_size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=SAVE_PRED_EVERY,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--snapshot_dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weights_path", type=str, default=WEIGHTS_PATH,
                        help="Path to the file with caffemodel weights. "
                             "If not set, all the variables are initialised randomly.")

    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                       help="where to save log file")

    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')
    
def load(loader, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      loader: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    loader.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the training."""
    args = get_arguments()
  
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_train_inputs"):
        reader_train = ImageReader(
            args.data_dir,
            args.data_train_list,
            input_size,
            RANDOM_SCALE,
            coord)
        image_batch_train, label_batch_train = reader_train.dequeue(args.batch_size)

    with tf.name_scope("create_val_inputs"):
        reader_val = ImageReader(
            args.data_dir,
            args.data_val_list,
            input_size,
            False,
            coord)
        image_batch_val, label_batch_val = reader_val.dequeue(args.batch_size)

    is_training = tf.placeholder(tf.bool,shape = [],name = 'stauts')
    image_batch,label_batch = tf.cond(is_training,lambda: (image_batch_train,label_batch_train),lambda: (image_batch_val,label_batch_val))

    # Create network.
    net = DeepLabLFOVModel(args.weights_path)

    # Define the loss and optimisation parameters.
    loss,recall1,recall2,accuracy = net.loss(image_batch, label_batch)
    optimiser = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    trainable = tf.trainable_variables()
    optim = optimiser.minimize(loss, var_list=trainable)
    
    pred = net.preds(image_batch)

    merged = tf.summary.merge_all() 
    if os.path.exists(args.log_dir):
    	shutil.rmtree(args.log_dir)
    summary_writer = tf.summary.FileWriter(args.log_dir,graph = tf.get_default_graph())

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    # config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=trainable, max_to_keep=40)

    if args.restore_from is not None:
        load(saver, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)

    if os.path.exists(args.snapshot_dir):
        shutil.rmtree(args.snapshot_dir)
   
    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        
        if step % args.save_pred_every == 0:
            loss_value, images, labels, preds, _ = sess.run([loss,image_batch, label_batch, pred, optim],feed_dict={is_training:True})
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in xrange(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels(labels[i, :, :, 0]))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels(preds[i, :, :, 0]))
            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)
            save(saver, sess, args.snapshot_dir, step)

            for i in range(VAL_LOOP):
                images, labels, preds = sess.run([image_batch, label_batch, pred],feed_dict={is_training:False})
                for j in range(BATCH_SIZE):
                    fig, axes = plt.subplots(1, 3, figsize = (16, 12))
                    axes.flat[0].set_title('data')
                    axes.flat[0].imshow((images[j] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                    axes.flat[1].set_title('mask')
                    axes.flat[1].imshow(decode_labels(labels[j, :, :, 0]))

                    axes.flat[2].set_title('pred')
                    axes.flat[2].imshow(decode_labels(preds[j, :, :, 0]))

                plt.savefig(args.save_dir + str(start_time) +'_'+str(i)+"test.png")
                plt.close(fig)

        else:
            loss_value, _ ,summary,recall1_value,recall2_value,acc= sess.run([loss,optim,merged,recall1,recall2,accuracy],feed_dict={is_training:True})
            print(recall1_value,recall2_value,acc)

            summary_writer.add_summary(summary,step)

        duration = time.time() - start_time
        print('step {:<6d} \t loss = {:.8f}, ({:.5f} sec/step)'.format(step,loss_value,duration))


    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
