import os
import traceback
from io import BytesIO

import av
import librosa
import numpy as np


def wav2(i, o, format):
    inp = av.open(i, "r")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    inp = av.open(i, "r")
    out = av.open(o, "w", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format)
    ostream.sample_rate = sr
    try:
        ostream.channels = 1
    except AttributeError:
        # Newer PyAV: channels is read-only, set layout instead
        ostream.layout = "mono"

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file, sr):
    if not os.path.exists(file):
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    try:
        with open(file, "rb") as f:
            with BytesIO() as out:
                audio2(f, out, "f32le", sr)
                return np.frombuffer(out.getvalue(), np.float32).flatten()

    except Exception:
        # Fallback: use librosa if PyAV fails (e.g. newer av API)
        try:
            audio, _ = librosa.load(file, sr=sr, mono=True)
            return audio
        except Exception:
            raise RuntimeError(traceback.format_exc())
