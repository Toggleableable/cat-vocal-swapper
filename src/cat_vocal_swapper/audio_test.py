import os

from scipy.spatial.distance import cosine

from .audio_functions import (
    export,
    export_segments,
    load_audio,
    play,
    split_audio_by_onsets,
)
from .comparison_test import extract_music_features
from .visualise import visualise


def test() -> None:
    path: str = "outputs/out11.mp3"
    audio, sample_rate = load_audio(path)
    # audio = split_audio_by_onsets(audio, sample_rate)[2]
    play(audio, sample_rate)

    ref_vec = extract_music_features(audio, sample_rate)

    output_folder = "outputs"
    clips = [f"{output_folder}/{i}" for i in os.listdir("outputs")]
    print(clips)
    scores = {}

    for clip in clips:
        clip_audio, clip_sr = load_audio(clip)
        # visualise(clip_audio, clip_sr)
        vec = extract_music_features(clip_audio, clip_sr)
        scores[clip] = 1 - cosine(ref_vec, vec)

    print(sorted(scores.items(), key=lambda x: x[1], reverse=True))
