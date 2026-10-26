"""Optional voice transport. Hermes owns replies; this module owns connection lifetime."""
import asyncio
import base64
import json
import math
import time
from urllib.parse import urlencode


def handoff(event, app_context=''):
    """Copy only accepted dialogue, never the gate's private ambient context."""
    if not isinstance(event, dict) or event.get('type') not in ('activate', 'continue') or not isinstance(event.get('session_id'), str) or not 0 < len(event['session_id']) <= 128:
        raise ValueError('A voice session requires an accepted gate event')
    history = event.get('context', [])
    if not isinstance(history, list) or len(history) > 8:
        raise ValueError('Use at most eight accepted context turns')
    messages = [*history, {'role': 'user', 'text': event.get('text')}]
    for turn in messages:
        if not isinstance(turn, dict) or turn.get('role') not in ('user', 'assistant') or not isinstance(turn.get('text'), str) or not turn['text'].strip() or len(turn['text']) > 2000:
            raise ValueError('Invalid accepted dialogue')
    purpose = event.get('purpose', '')
    if not isinstance(purpose, str) or len(purpose) > 1000 or not isinstance(app_context, str) or len(app_context) > 8000:
        raise ValueError('Invalid application context')
    return {'session_id': event['session_id'], 'purpose': purpose,
            'app_context': app_context, 'messages': [dict(turn) for turn in messages]}


class VoiceSession:
    """One accepted wake -> one bounded connection -> local listening again.

    ask(text, history, context) returns {'reply': str, 'finished': bool}.
    audio supplies input(), play(pcm), drain(), interrupt() and close() coroutines.
    """
    def __init__(self, event, *, app_context='', idle_seconds=30, max_seconds=300):
        if any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n <= 0
               for n in (idle_seconds, max_seconds)):
            raise ValueError('Session time limits must be positive finite seconds')
        self.context = handoff(event, app_context)
        self.idle_seconds = idle_seconds
        self.max_seconds = max_seconds

    async def connect(self, key, ask, audio, *, model='gpt-realtime-2.1'):
        from websockets.asyncio.client import connect
        if not key:
            raise ValueError('OPENAI_API_KEY is required for voice, not for local listening')
        url = 'wss://api.openai.com/v1/realtime?' + urlencode({'model': model})
        try:
            # Includes connection setup; a stuck handshake cannot run indefinitely.
            async with asyncio.timeout(self.max_seconds):
                async with connect(url, additional_headers={'Authorization': 'Bearer ' + key},
                                   open_timeout=15, close_timeout=2, max_size=4_000_000) as ws:
                    return await self.run(ws, ask, audio, model=model)
        except TimeoutError:
            return 'time_limit'
        finally:
            await asyncio.wait_for(audio.close(), 2)

    async def run(self, ws, ask, audio, *, model='gpt-realtime-2.1'):
        """Run on an open Realtime socket, and actually close it on every exit."""
        tasks = []
        incoming = asyncio.Queue(maxsize=8)
        outgoing = asyncio.Queue(maxsize=2000)
        transcripts = {}
        ready = asyncio.Event()
        response_done = asyncio.Event()
        activity = time.monotonic()
        busy = False
        speaking = False
        response_id = None
        cancelled = set()
        generation = 0
        history = self.context['messages'][:-1]

        async def send(event):
            await ws.send(json.dumps(event))

        async def item(turn):
            await send({'type': 'conversation.item.create', 'item': {
                'type': 'message', 'role': turn['role'], 'content': [{
                    'type': 'input_text' if turn['role'] == 'user' else 'output_text',
                    'text': turn['text']}]}})

        async def capture():
            async for pcm in audio.input():
                await send({'type': 'input_audio_buffer.append', 'audio': base64.b64encode(pcm).decode()})
            raise RuntimeError('Microphone stopped')

        async def playback():
            while True:
                pcm, version = await outgoing.get()
                try:
                    if version == generation:
                        await audio.play(pcm)
                finally:
                    outgoing.task_done()

        async def receive():
            nonlocal activity, speaking, response_id, generation
            async for raw in ws:
                event = json.loads(raw)
                kind = event['type']
                if kind == 'session.updated':
                    ready.set()
                elif kind == 'input_audio_buffer.committed':
                    future = transcripts.setdefault(event['item_id'], asyncio.get_running_loop().create_future())
                    incoming.put_nowait((event['item_id'], future))
                elif kind == 'conversation.item.input_audio_transcription.completed':
                    future = transcripts.setdefault(event['item_id'], asyncio.get_running_loop().create_future())
                    if not future.done():
                        future.set_result(event['transcript'])
                elif kind == 'input_audio_buffer.speech_started':
                    speaking = True
                    activity = time.monotonic()
                    if response_id:
                        cancelled.add(response_id)
                        await send({'type': 'response.cancel', 'response_id': response_id})
                    generation += 1
                    await audio.interrupt()
                elif kind == 'input_audio_buffer.speech_stopped':
                    speaking = False
                    activity = time.monotonic()
                elif kind == 'response.created':
                    response_id = event['response']['id']
                elif kind == 'response.output_audio.delta':
                    if event['response_id'] not in cancelled:
                        pcm = base64.b64decode(event['delta'], validate=True)
                        for offset in range(0, len(pcm), 1920):
                            outgoing.put_nowait((pcm[offset:offset + 1920], generation))
                elif kind == 'response.done':
                    if event['response']['status'] not in ('completed', 'cancelled'):
                        raise RuntimeError('Voice rendering did not complete')
                    response_id = None
                    response_done.set()
                elif kind in ('error', 'conversation.item.input_audio_transcription.failed'):
                    raise RuntimeError('Realtime failed; check credentials, credit and model access')
            raise RuntimeError('Realtime disconnected')

        async def turns():
            nonlocal activity, busy
            while True:
                item_id, future = await incoming.get()
                busy = True
                try:
                    text = await asyncio.wait_for(future, 20)
                    transcripts.pop(item_id, None)
                    if not text.strip():
                        continue
                    result = await asyncio.wait_for(ask(text, list(history), self.context), 90)
                    reply, finished = result['reply'], result.get('finished', False)
                    if not isinstance(reply, str) or len(reply) > 8000 or not isinstance(finished, bool):
                        raise ValueError('Invalid Hermes reply')
                    history.append({'role': 'user', 'text': text})
                    version = generation
                    if reply.strip():
                        response_done.clear()
                        await send({'type': 'response.create', 'response': {
                            'conversation': 'none', 'output_modalities': ['audio'],
                            'instructions': 'Read the sole input aloud verbatim. Do not answer or add anything.',
                            'input': [{'type': 'message', 'role': 'user', 'content': [
                                {'type': 'input_text', 'text': reply}]}],
                            'tools': [], 'tool_choice': 'none', 'max_output_tokens': 1200}})
                        await asyncio.wait_for(response_done.wait(), 60)
                        await outgoing.join()
                        # response.done means generation ended, not that the speaker drained.
                        await audio.drain()
                        heard = reply if version == generation else '[Reply interrupted; the user may not have heard it.]'
                        await item({'role': 'assistant', 'text': heard})
                        history.append({'role': 'assistant', 'text': heard})
                    activity = time.monotonic()
                    if finished and version == generation and not speaking and incoming.empty():
                        return 'agent_finished'
                finally:
                    busy = False
                    incoming.task_done()

        async def idle():
            while True:
                await asyncio.sleep(min(0.25, self.idle_seconds))
                if not busy and not speaking and incoming.empty() and time.monotonic() - activity >= self.idle_seconds:
                    return 'idle'

        try:
            async with asyncio.timeout(self.max_seconds):
                receiver = asyncio.create_task(receive())
                tasks.append(receiver)
                await send({'type': 'session.update', 'session': {
                    'type': 'realtime', 'model': model, 'output_modalities': ['audio'],
                    'instructions': 'Only transcribe input and speak supplied Hermes replies. Do not answer independently.\n'
                                    + json.dumps({k: self.context[k] for k in ('purpose', 'app_context')}),
                    'tools': [], 'tool_choice': 'none',
                    'audio': {'input': {'format': {'type': 'audio/pcm', 'rate': 24000},
                        'transcription': {'model': 'gpt-4o-mini-transcribe', 'language': 'en'},
                        'turn_detection': {'type': 'server_vad', 'create_response': False, 'interrupt_response': False}},
                        'output': {'format': {'type': 'audio/pcm', 'rate': 24000}, 'voice': 'marin'}}}})
                await asyncio.wait_for(ready.wait(), 15)
                for turn in self.context['messages']:
                    await item(turn)
                first = asyncio.get_running_loop().create_future()
                first.set_result(self.context['messages'][-1]['text'])
                incoming.put_nowait((None, first))
                tasks.extend(asyncio.create_task(fn()) for fn in (turns, capture, playback, idle))
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                results = [task.result() for task in done]
                return next((result for result in results if result), 'closed')
        except TimeoutError:
            return 'time_limit'
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(ws.close(), 2)
            finally:
                if getattr(ws, 'transport', None):
                    ws.transport.abort()
                await asyncio.wait_for(audio.interrupt(), 2)
