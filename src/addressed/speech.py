"""Streaming local English ASR. Audio exists only in bounded memory queues."""
from pathlib import Path
import queue
import threading
import time
import wave


class Transcriber:
    def __init__(self, directory):
        import sherpa_onnx
        folder = Path(directory) / 'speech'
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(folder / 'tokens.txt'),
            encoder=str(folder / 'encoder-epoch-99-avg-1.int8.onnx'),
            decoder=str(folder / 'decoder-epoch-99-avg-1.onnx'),
            joiner=str(folder / 'joiner-epoch-99-avg-1.int8.onnx'),
            num_threads=1, sample_rate=16000, feature_dim=80,
            enable_endpoint_detection=True, decoding_method='greedy_search',
            rule1_min_trailing_silence=2.4, rule2_min_trailing_silence=0.8,
            rule3_min_utterance_length=15)
        self.stream = self.recognizer.create_stream()

    def accept(self, pcm, rate=16000, final=False):
        import numpy as np
        self.stream.accept_waveform(rate, np.frombuffer(pcm, dtype='<i2').astype('float32') / 32768)
        if final:
            self.stream.accept_waveform(rate, np.zeros(rate, dtype='float32'))
            self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        if final or self.recognizer.is_endpoint(self.stream):
            text = self.recognizer.get_result(self.stream)
            if final:
                self.stream = self.recognizer.create_stream()
            else:
                # Keep buffered acoustic context at a sentence boundary.
                self.recognizer.reset(self.stream)
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


def microphone(transcriber, device=None):
    """Decode continuously even while the decision model is busy."""
    import sounddevice as sd
    audio = queue.Queue(maxsize=30)
    utterances = queue.Queue(maxsize=4)
    stopped = threading.Event()
    overflow = threading.Event()

    def capture(indata, frames, timing, status):
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
                    transcriber.stream = transcriber.recognizer.create_stream()
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

    with sd.RawInputStream(device=device, samplerate=16000, channels=1, dtype='int16',
                           blocksize=1600, callback=capture):
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
