import librosa
import numpy as np
from numpy.typing import NDArray

from .visualise import visualise


def test() -> None:
    path: str = "test.mp3"

    segments = split_audio_by_onsets(path)
    print(len(segments))
    for i in segments:
        visualise(i)


def split_audio_by_onsets(
    audio_path: str,
    backtrack: bool = True,
) -> list[NDArray[np.float32]]:
    audio: NDArray[np.float32]
    sample_rate: float
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    onset_samples = librosa.onset.onset_detect(
        y=audio, sr=sample_rate, backtrack=backtrack, units="samples"
    )

    segments: list[NDArray[np.float32]] = []
    for index in range(len(onset_samples) - 1):
        segments.append(audio[onset_samples[index] : onset_samples[index + 1]])

    return segments
