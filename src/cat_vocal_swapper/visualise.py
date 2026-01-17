import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

# pyright: reportUnknownMemberType=false, reportUnusedCallResult=false


def visualise(samples: NDArray[np.float32]) -> None:
    plt.figure(figsize=(15, 4))
    plt.plot(samples, color="slateblue")
    # plt.xlim(0, 200000)
    # plt.ylim(-25000, 25000)
    plt.title("Original Audio Waveform")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
