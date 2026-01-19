import librosa
import numpy as np
from numpy.typing import NDArray


def extract_music_features(
    audio: NDArray[np.float32], sample_rate: float
) -> NDArray[np.float32]:
    n_fft = 256
    hop_length = 128

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=20, n_fft=n_fft, hop_length=hop_length
    )
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)

    mfcc_mean: NDArray[np.float32] = np.mean(mfcc, axis=1)
    chroma_mean: NDArray[np.float32] = np.mean(chroma, axis=1)

    return np.hstack([mfcc_mean, chroma_mean])
