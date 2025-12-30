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

    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)

    ostream = out.add_stream(format)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        frame.pts = None
        resampled = resampler.resample(frame)
        print(f"[DEBUG] resampler.resample() returned type: {type(resampled)}, len: {len(resampled) if isinstance(resampled, list) else 'N/A'}")
        for resampled_frame in resampled:
            for p in ostream.encode(resampled_frame):
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
                audio = np.frombuffer(out.getvalue(), np.float32)
                print(f"[DEBUG] audio.shape: {audio.shape}, duration: {len(audio)/sr:.2f}s")
                return audio

    except AttributeError:
        audio = file[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file[0], target_sr=16000)

    except Exception:
        raise RuntimeError(traceback.format_exc())
