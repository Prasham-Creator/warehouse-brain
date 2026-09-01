# module_d_connectionist/__init__.py
# Exposes hopfield_recall and rnn_predict_anomaly per §6.6.

from module_d_connectionist.hopfield import (corrupt, energy,
                                             generate_pattern_1,
                                             generate_pattern_2,
                                             generate_pattern_3,
                                             generate_pattern_4,
                                             generate_pattern_5,
                                             generate_pattern_6)
from module_d_connectionist.hopfield import recall as hopfield_recall
from module_d_connectionist.hopfield import train
from module_d_connectionist.rnn_model import (generate_synthetic_sequences,
                                              rnn_predict_anomaly, train_rnn)

__all__ = [
    "hopfield_recall",
    "train",
    "generate_pattern_1",
    "generate_pattern_2",
    "generate_pattern_3",
    "generate_pattern_4",
    "generate_pattern_5",
    "generate_pattern_6",
    "energy",
    "corrupt",
    "rnn_predict_anomaly",
    "train_rnn",
    "generate_synthetic_sequences"
]
