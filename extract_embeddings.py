
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import sys
import argparse


from algorithms import get_algo
from config import CONFIG
from datasets import create_one_epoch_dataset
from utils import get_embeddings_dataset
from utils import get_lr_opt_global_step
from utils import restore_ckpt
from utils import setup_eval_dir
from utils import load_config
from utils import prepare_gpu
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gfile = tf.io.gfile
layers = tf.keras.layers




def parse_args():
    parser = argparse.ArgumentParser(
        description='Runs a video through TCC and outputs the embeddings')
    parser.add_argument('--logdir',
                        type=str,
                        help='Path to logs.',
                        default='/mnt/nas/workspace/fatemeht/projects/SyncNet/trained_model/all_view/',
                        required=False)
    parser.add_argument('--save_path',
                        type=str,
                        help='Path to folder.',
                        default='/mnt/nas/workspace/fatemeht/projects/SyncNet/Stanford_embeddings/embeddings_stanford5_test.npy',
                        required=False)
    parser.add_argument('--config',
                        type=str,
                        help='training configuration in Json format',
                        default='/mnt/nas/workspace/fatemeht/projects/SyncNet/trained_model/all_view/config.json',
                        required=False)

    parser.add_argument('--path_to_tfrecords',
                        type=str,
                        help='path to tfrecords',
                        default='/bigdata2/dataset/stanford_tfrecords/',
                        required=False)
    parser.add_argument('--dataset',
                        type=str,
                        default='PLAX',
                        required=False)
    parser.add_argument('--split',
                        type=str,
                        default='val',
                        required=False)

    parser.add_argument('--max_embs',
                        type=int,
                        help='Max number of videos to embed. 0 or less means embed all videos in dataset',
                        default=0,
                        required=False)

    parser.add_argument('--visualize',
                        type=bool,
                        help='Visualize images.',
                        default=False,
                        required=False)

    parser.add_argument('--keep_data',
                        type=bool,
                        help='Keep frames of video with embeddings.',
                        default=False,
                        required=False)

    parser.add_argument('--optical_flow',
                        type=bool,
                        help='Seclect true if the input is opticalflow',
                        default=False,
                        required=False)

    parser.add_argument('--keep_labels',
                        type=bool,
                        help='Keep per-frame labels with embeddings',
                        default=True,
                        required=False)

    parser.add_argument('--sample_all_stride',
                        type=int,
                        help='Stride between frames that will be embedded.',
                        default=1,
                        required=False)
    parser.add_argument('--frames_per_batch',
                        type=int,
                        help='frames_per_batchs',
                        default=1,
                        required=False)
    parser.add_argument('--gpu',
                        type=str,
                        help='(optional) index of the gpu to use',
                        required=False,
                        default="-2")
    parser.add_argument('--defun',
                        type=bool,
                        help='Defun functions in algo for faster training',
                        default=False,
                        required=False)

    return parser.parse_args()


evaluated_last_ckpt = False






def evaluate(args):
  """Extract embeddings."""

  logdir = args.logdir
  setup_eval_dir(logdir)
  # Can ignore frame labels if dataset doesn't have per-frame labels.
  CONFIG.DATA.FRAME_LABELS = args.keep_labels
  # Subsample frames in case videos are long or fps is high to save memory.
  CONFIG.DATA.SAMPLE_ALL_STRIDE = args.sample_all_stride

  algo = get_algo(CONFIG.TRAINING_ALGO)
  _, optimizer, _ = get_lr_opt_global_step()
  restore_ckpt(logdir=logdir, **algo.model)

  if args.defun:
    algo.call = tf.function(algo.call)
    algo.compute_loss = tf.function(algo.compute_loss)

  iterator, _ = create_one_epoch_dataset(args.dataset, args.split, mode='eval',
                                      path_to_tfrecords=args.path_to_tfrecords)

  max_embs = None if args.max_embs <= 0 else args.max_embs
  embeddings = get_embeddings_dataset(
      algo.model,
      iterator,
      frames_per_batch=args.frames_per_batch,
      keep_data=args.keep_data,
      optical_flow=args.optical_flow,
      keep_labels=args.keep_labels,
      max_embs=max_embs)
  np.save(gfile.GFile(args.save_path, 'w'), embeddings)
  return  embeddings


def main(_):
  tf.keras.backend.set_learning_phase(0)
  args = parse_args()
  config = load_config(args.config)
  CONFIG.update(config)

  prepare_gpu(args.gpu)
  embeddings = evaluate(args)


if __name__ == '__main__':
  main(sys.argv[1:])