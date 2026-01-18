import librosa
import numpy as np
from numpy.typing import NDArray

from .audio_functions import export, load_audio, play
from .visualise import visualise


def test() -> None:
    path: str = "test.mp3"
    audio, sample_rate = load_audio(path)
    segments = split_audio_by_onsets(audio, sample_rate)

    export(segments[0], sample_rate)

    for i in segments:
        play(i, sample_rate)
        visualise(i, sample_rate)


def split_audio_by_onsets(
    audio: NDArray[np.float32],
    sample_rate: float,
    backtrack: bool = True,
) -> list[NDArray[np.float32]]:
    onset_samples = librosa.onset.onset_detect(
        y=audio, sr=sample_rate, backtrack=backtrack, units="samples"
    )

    segments: list[NDArray[np.float32]] = []
    for index in range(len(onset_samples) - 1):
        segments.append(audio[onset_samples[index] : onset_samples[index + 1]])

    return segments
