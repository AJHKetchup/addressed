"""Local Addressed listening -> bounded Realtime voice + native Hermes -> local listening."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from addressed import Gate, LocalDecisionModel
from addressed.models import default_directory
from addressed.realtime import VoiceSession
from addressed.speech import Transcriber, microphone


class Audio:
    def __init__(self, device=None):
        import sounddevice as sd
        self.mic = sd.RawInputStream(device=device, samplerate=24000, channels=1,
                                     dtype='int16', blocksize=960)
        self.speaker = sd.RawOutputStream(samplerate=24000, channels=1, dtype='int16')
        self.lock = asyncio.Lock()

    async def input(self):
        self.mic.start()
        while True:
            pcm, overflow = await asyncio.to_thread(self.mic.read, 960)
            if overflow:
                raise RuntimeError('Voice microphone overflowed')
            yield bytes(pcm)

    async def play(self, pcm):
        async with self.lock:
            if not self.speaker.active:
                self.speaker.start()
            if await asyncio.to_thread(self.speaker.write, pcm):
                raise RuntimeError('Voice speaker underflowed')

    async def drain(self):
        async with self.lock:
            await asyncio.to_thread(self.speaker.stop)

    async def interrupt(self):
        async with self.lock:
            await asyncio.to_thread(self.speaker.abort)

    async def close(self):
        await asyncio.to_thread(self.mic.close)
        await asyncio.to_thread(self.speaker.close)


async def converse(event, args):
    worker = None
    audio = None
    # The voice API credential must not override Hermes' configured reasoning provider.
    environment = dict(os.environ)
    environment.pop('OPENAI_API_KEY', None)
    try:
        worker = await asyncio.create_subprocess_exec(args.hermes_python, '-u',
            str(Path(__file__).with_name('hermes_voice_worker.py')),
            env=environment, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE)
        ready = await asyncio.wait_for(worker.stdout.readline(), 45)
        if not ready or json.loads(ready).get('ready') is not True:
            raise RuntimeError('Hermes is not ready; check its configured provider')

        async def ask(text, history, context):
            worker.stdin.write((json.dumps({'text': text, 'history': history, 'context': context}) + '\n').encode())
            await worker.stdin.drain()
            line = await worker.stdout.readline()
            if not line:
                raise RuntimeError('Hermes worker disconnected')
            return json.loads(line)

        audio = Audio(args.device)
        session = VoiceSession(event, app_context=args.context,
                               idle_seconds=args.idle_seconds, max_seconds=args.max_seconds)
        reason = await session.connect(os.environ['OPENAI_API_KEY'], ask, audio)
        print('Voice closed: ' + reason, file=sys.stderr)
    finally:
        try:
            if audio:
                await asyncio.wait_for(audio.close(), 2)
        finally:
            if worker and worker.returncode is None:
                worker.terminate()
                try:
                    await asyncio.wait_for(worker.wait(), 2)
                except TimeoutError:
                    worker.kill()
                    await worker.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hermes-python', required=True, help='Python executable in your existing Hermes environment')
    p.add_argument('--models', type=Path, default=default_directory())
    p.add_argument('--device', type=int)
    p.add_argument('--purpose', default='Answer questions and discuss media.')
    p.add_argument('--context', default='', help='Trusted application facts shared with Hermes and Realtime')
    p.add_argument('--threshold', type=float, default=0.90)
    p.add_argument('--idle-seconds', type=float, default=30)
    p.add_argument('--max-seconds', type=float, default=300)
    args = p.parse_args()
    if not os.getenv('OPENAI_API_KEY'):
        p.error('Set OPENAI_API_KEY in your environment; do not put it in this script')
    gate = Gate(LocalDecisionModel(), purpose=args.purpose, threshold=args.threshold,
                followup_threshold=min(args.threshold, 0.80))
    transcriber = Transcriber(args.models)
    try:
        while True:
            print('Listening locally.', file=sys.stderr)
            source = microphone(transcriber, args.device)
            event = None
            try:
                for text in source:
                    event = gate.decide(text)['event']
                    if event:
                        break
            finally:
                # No local TTL, stale queued triggers or second mic loop during voice.
                source.close()
            try:
                asyncio.run(converse(event, args))
            except Exception as error:
                print('Voice stopped: ' + type(error).__name__, file=sys.stderr)
            finally:
                gate.reset()
                transcriber.stream = transcriber.recognizer.create_stream()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
