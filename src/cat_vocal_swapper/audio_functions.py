import librosa
import numpy as np
import sounddevice as sd  # pyright: ignore[reportMissingTypeStubs]
import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray


def load_audio(path: str) -> tuple[NDArray[np.float32], float]:
    return librosa.load(path, sr=None, mono=True)


def play(
    audio: NDArray[np.float32], sample_rate: float, blocking: bool = False
) -> None:
    sd.play(audio, samplerate=sample_rate, blocking=blocking)  # pyright: ignore[reportUnknownMemberType]


def export(audio: NDArray[np.float32], sample_rate: float) -> None:
    sf.write("output.mp3", audio, sample_rate)  # pyright: ignore[reportUnknownMemberType]
