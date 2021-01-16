
"""List all available algorithms."""

# from alignment import Alignment
from algos.alignment_sal_tcn import AlignmentSaLTCN
# from method.algos.classification import Classification
# from method.algos.sal import SaL
# from method.algos.tcn import TCN

ALGO_NAME_TO_ALGO_CLASS = {
    'alignment_sal_tcn': AlignmentSaLTCN,
}


def get_algo(algo_name):
  """Returns training algo."""
  if algo_name not in ALGO_NAME_TO_ALGO_CLASS.keys():
    raise ValueError('%s not supported yet.' % algo_name)
  algo = ALGO_NAME_TO_ALGO_CLASS[algo_name]
  return algo()
