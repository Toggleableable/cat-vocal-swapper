import numpy as np
from librosa.display import waveshow
from matplotlib import pyplot as plt
from numpy.typing import NDArray

# pyright: reportUnknownMemberType=false, reportUnusedCallResult=false


def visualise(samples: NDArray[np.float32], sample_rate: float) -> None:
    plt.figure(figsize=(15, 4))
    waveshow(samples, sr=sample_rate)
    plt.show()
