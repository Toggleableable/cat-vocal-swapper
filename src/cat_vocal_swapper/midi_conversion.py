# from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportMissingTypeStubs=false


def convert_to_midi(path: str):
    _, midi_data, _ = predict(path)
    midi_data.write("outputs/output.mid")
