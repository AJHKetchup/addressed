"""Streaming local English ASR. Audio exists only in bounded memory queues."""
from pathlib import Path
import json
import queue
import threading
import time
import wave


class Transcriber:
    def __init__(self, directory):
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        self.model = Model(str(Path(directory) / 'speech' / 'vosk-model-small-en-us-0.15'))
        self.reset()

    def reset(self, rate=16000):
        from vosk import KaldiRecognizer
        self.rate = rate
        self.recognizer = KaldiRecognizer(self.model, rate)

    def accept(self, pcm, rate=16000, final=False):
        if rate != self.rate:
            self.reset(rate)
        if final:
            text = json.loads(self.recognizer.FinalResult()).get('text', '')
            self.reset()
            return text or None
        if self.recognizer.AcceptWaveform(pcm):
            text = json.loads(self.recognizer.Result()).get('text', '')
            return text or None
        return None

    def wav(self, filename):
        with wave.open(str(filename), 'rb') as audio:
            if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
                raise ValueError('Use a mono PCM16 WAV file')
            while pcm := audio.readframes(int(audio.getframerate() / 10)):
                text = self.accept(pcm, audio.getframerate())
                if text:
                    yield text
            text = self.accept(b'', audio.getframerate(), final=True)
            if text:
                yield text


def microphone(transcriber, device=None, *, capture=None):
    """Decode continuously even while the decision model is busy."""
    audio = queue.Queue(maxsize=30)
    utterances = queue.Queue(maxsize=4)
    stopped = threading.Event()
    overflow = threading.Event()

    def capture_audio(indata, frames, timing, status):
        if status:
            overflow.set()
        try:
            audio.put_nowait(bytes(indata))
        except queue.Full:
            overflow.set()

    def decode():
        try:
            while not stopped.is_set():
                try:
                    pcm = audio.get(timeout=0.2)
                except queue.Empty:
                    continue
                if overflow.is_set():
                    # A discontinuous recording must not become a pieced-together command.
                    while not audio.empty():
                        audio.get_nowait()
                    transcriber.reset()
                    overflow.clear()
                    continue
                text = transcriber.accept(pcm)
                if text:
                    try:
                        utterances.put_nowait((time.monotonic(), text))
                    except queue.Full:
                        pass  # Drop rather than execute a growing backlog of stale requests.
        except Exception as error:
            utterances.put((time.monotonic(), error))

    input_callback = capture_audio
    if capture is None:
        import sounddevice as sd
        source = sd.RawInputStream(device=device, samplerate=16000, channels=1, dtype='int16',
                                   blocksize=1600, callback=input_callback)
    else:
        source = capture(input_callback)
    with source:
        worker = threading.Thread(target=decode, daemon=True)
        worker.start()
        try:
            while True:
                try:
                    when, text = utterances.get(timeout=0.25)
                except queue.Empty:
                    continue
                if isinstance(text, Exception):
                    raise text
                if text and time.monotonic() - when < 5:
                    yield text
        finally:
            stopped.set()
            worker.join(timeout=2)
