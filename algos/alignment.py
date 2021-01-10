
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from algos.algorithm import Algorithm
from config import CONFIG
from tcc.alignment import compute_alignment_loss

FLAGS = flags.FLAGS


class Alignment(Algorithm):
  """Uses cycle-consistency loss to perform unsupervised training."""

  def compute_loss(self, embs, steps, seq_lens, global_step, training,
                   frame_labels, seq_labels):
    if training:
      batch_size = CONFIG.TRAIN.BATCH_SIZE
      num_steps = CONFIG.TRAIN.NUM_FRAMES
    else:
      batch_size = CONFIG.EVAL.BATCH_SIZE
      num_steps = CONFIG.EVAL.NUM_FRAMES

    loss = compute_alignment_loss(
        embs,
        batch_size,
        steps=steps,
        seq_lens=seq_lens,
        stochastic_matching=CONFIG.ALIGNMENT.STOCHASTIC_MATCHING,
        normalize_embeddings=False,
        loss_type=CONFIG.ALIGNMENT.LOSS_TYPE,
        similarity_type=CONFIG.ALIGNMENT.SIMILARITY_TYPE,
        num_cycles=int(batch_size * num_steps * CONFIG.ALIGNMENT.FRACTION),
        cycle_length=CONFIG.ALIGNMENT.CYCLE_LENGTH,
        temperature=CONFIG.ALIGNMENT.SOFTMAX_TEMPERATURE,
        label_smoothing=CONFIG.ALIGNMENT.LABEL_SMOOTHING,
        variance_lambda=CONFIG.ALIGNMENT.VARIANCE_LAMBDA,
        huber_delta=CONFIG.ALIGNMENT.HUBER_DELTA,
        normalize_indices=CONFIG.ALIGNMENT.NORMALIZE_INDICES)

    return loss
