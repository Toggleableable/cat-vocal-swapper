from .audio_functions import export, load_audio, play, split_audio_by_onsets
from .visualise import visualise


def test() -> None:
    path: str = "test.mp3"
    audio, sample_rate = load_audio(path)
    segments = split_audio_by_onsets(audio, sample_rate)

    # export(segments[0], sample_rate)

    for i in segments:
        play(i, sample_rate)
        visualise(i, sample_rate)
