import os

import librosa
import numpy as np
import sounddevice as sd  # pyright: ignore[reportMissingTypeStubs]
import soundfile as sf  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray


def load_audio(path: str) -> tuple[NDArray[np.float32], float]:
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    audio, _ = librosa.effects.trim(audio)  # pyright: ignore[reportUnknownMemberType]
    return audio, sample_rate


def play(
    audio: NDArray[np.float32], sample_rate: float, blocking: bool = False
) -> None:
    sd.play(audio, samplerate=sample_rate, blocking=blocking)  # pyright: ignore[reportUnknownMemberType]


def export(audio: NDArray[np.float32], sample_rate: float) -> None:
    sf.write("output.mp3", audio, sample_rate)  # pyright: ignore[reportUnknownMemberType]


def export_segments(
    audio_list: list[NDArray[np.float32]], sample_rate: float, output_folder: str
) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    segment_number = 1
    for audio in audio_list:
        filename = f"outputs/out{segment_number:02d}.mp3"
        sf.write(filename, audio, sample_rate)  # pyright: ignore[reportUnknownMemberType]
        segment_number += 1


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
