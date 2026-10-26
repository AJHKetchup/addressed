"""Socket contracts, not a claim about an AI model's clarification quality."""
import asyncio
import base64
import json
import unittest

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from addressed.realtime import VoiceSession, handoff

EVENT = {'type': 'activate', 'session_id': 'accepted', 'purpose': 'Control the TV.',
         'context': [], 'text': 'Add Samurai Champloo to our library and play it every day at 1 PM.'}


class Audio:
    def __init__(self):
        self.played = b''
        self.drained = False
        self.interrupted = False

    async def input(self):
        await asyncio.Future()
        yield b''

    async def play(self, pcm):
        self.drained = False
        self.played += pcm

    async def drain(self):
        await asyncio.sleep(0.02)
        self.drained = True

    async def interrupt(self):
        self.interrupted = True


class VoiceContract(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, mode='conversation'):
        received, delegated = [], []
        closed = asyncio.Event()
        audio = Audio()
        count = 0

        def call(name, arguments):
            return {'type': 'function_call', 'name': name, 'call_id': 'call-' + str(count),
                    'arguments': json.dumps(arguments)}

        async def user(ws, text, ident):
            await asyncio.sleep(0.04)
            await ws.send(json.dumps({'type': 'input_audio_buffer.committed', 'item_id': ident}))
            await ws.send(json.dumps({'type': 'conversation.item.input_audio_transcription.completed',
                                      'item_id': ident, 'transcript': text}))

        async def peer(ws):
            nonlocal count
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    received.append(event)
                    if event['type'] == 'session.update':
                        await ws.send(json.dumps({'type': 'session.updated'}))
                    elif event['type'] == 'response.create':
                        count += 1
                        output = []
                        said = ''
                        if mode == 'conversation':
                            if count == 1:
                                said = 'Sub or dub, and one episode daily?'
                            elif count == 2:
                                output = [call('hermes_task', {'task': 'Add Samurai Champloo, English dub, and schedule one episode daily at 13:00 America/Chicago.'})]
                            elif count == 3:
                                said = 'The download and daily slot are requested, pending media.'
                            elif count == 4:
                                output = [call('end_conversation', {})]
                            else:
                                said = 'Goodbye.'
                        elif mode in ('stall', 'failure') and count == 1:
                            output = [call('hermes_task', {'task': 'Read TV status.'})]
                        else:
                            said = 'I am here.'
                        rid = 'r' + str(count)
                        await ws.send(json.dumps({'type': 'response.created', 'response': {'id': rid}}))
                        if said:
                            output = [{'type': 'message', 'content': [{'type': 'output_audio', 'transcript': said}]}]
                            await ws.send(json.dumps({'type': 'response.output_audio.delta', 'response_id': rid,
                                'item_id': 'audio-' + rid, 'delta': base64.b64encode(bytes(1920)).decode()}))
                        await ws.send(json.dumps({'type': 'response.done', 'response': {'id': rid, 'status': 'completed', 'output': output}}))
                        if mode == 'conversation' and count == 1:
                            await user(ws, 'English dub, one episode a day.', 'language')
                        elif mode == 'conversation' and count == 3:
                            await user(ws, 'That is all, goodbye.', 'goodbye')
            finally:
                closed.set()

        async def execute(task, context):
            delegated.append((task, context))
            if mode == 'stall':
                await asyncio.Future()
            if mode == 'failure':
                raise RuntimeError('Action unavailable')
            return {'acquisition': 'requested', 'schedule': 'pending_media'}

        async with serve(peer, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}', close_timeout=0.2) as ws:
                session = VoiceSession(EVENT, app_context='Local timezone America/Chicago.',
                                       idle_seconds=0.5, max_seconds=3)
                reason = await session.run(ws, execute, audio)
                await asyncio.wait_for(closed.wait(), 1)
        self.assertTrue(audio.interrupted)
        self.assertEqual(received[0]['session']['tool_choice'], 'auto')
        self.assertEqual({t['name'] for t in received[0]['session']['tools']}, {'hermes_task', 'end_conversation'})
        items = [e['item'] for e in received if e['type'] == 'conversation.item.create']
        originals = [i for i in items if i['type'] == 'message' and i['content'][0].get('text') == EVENT['text']]
        self.assertEqual(len(originals), 1)
        return reason, audio, delegated, items

    async def test_clarify_delegate_both_intents_and_end_after_audio(self):
        reason, audio, delegated, items = await self.exercise()
        self.assertEqual(reason, 'conversation_finished', json.dumps(items))
        self.assertTrue(audio.drained)
        self.assertEqual(len(delegated), 1)
        task, context = delegated[0]
        self.assertIn('13:00', task)
        self.assertEqual(context['messages'][0]['text'], EVENT['text'])
        self.assertIn({'role': 'user', 'text': 'English dub, one episode a day.'}, context['messages'])
        self.assertEqual(len([i for i in items if i['type'] == 'function_call_output']), 2)

    async def test_silence_closes_socket_without_delegating_smalltalk(self):
        reason, _, delegated, _ = await self.exercise('idle')
        self.assertEqual(reason, 'idle')
        self.assertFalse(delegated)

    async def test_time_limit_closes_while_hermes_stalls(self):
        reason, _, _, _ = await self.exercise('stall')
        self.assertEqual(reason, 'time_limit')

    async def test_tool_failure_returns_error_to_realtime(self):
        reason, _, _, items = await self.exercise('failure')
        self.assertEqual(reason, 'idle')
        output = next(i for i in items if i['type'] == 'function_call_output')
        self.assertEqual(json.loads(output['output'])['status'], 'failed')

    def test_handoff_excludes_private_classifier_state(self):
        context = handoff({**EVENT, 'recent_conversation': [{'text': 'Private room conversation'}]})
        self.assertNotIn('Private room conversation', json.dumps(context))


if __name__ == '__main__':
    unittest.main()
