"""SinWave: 正弦の干渉と回折だけで作る学習器（重み行列なし）。"""
from .api import PRESETS, MergedClassifier, SinWaveClassifier, merge
from .model import Config, Parallel, SinLayer, SinWave, encode

__all__ = ["SinWaveClassifier", "MergedClassifier", "merge", "PRESETS",
           "Config", "SinWave", "SinLayer", "Parallel", "encode"]
