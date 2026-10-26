"""Exercise the actual socket boundary with a local protocol peer, not a model imitation."""
import asyncio
import base64
import json
import unittest

from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from addressed.realtime import VoiceSession, handoff


EVENT = {'type': 'activate', 'session_id': 'accepted', 'purpose': 'Discuss the TV.',
         'context': [{'role': 'user', 'text': 'We are watching Cowboy Bebop.'},
                     {'role': 'assistant', 'text': 'Do you want the English dub?'}],
         'text': 'Yes, please.'}


class Audio:
    def __init__(self):
        self.played = b''
        self.drained = False
        self.interrupted = False

    async def input(self):
        await asyncio.Future()
        yield b''

    async def play(self, pcm):
        self.played += pcm

    async def drain(self):
        await asyncio.sleep(0.03)
        self.drained = True

    async def interrupt(self):
        self.interrupted = True


class VoiceContract(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, *, finish=False, stall=False, fail=False, followup=False):
        received, questions = [], []
        closed = asyncio.Event()
        audio = Audio()

        async def peer(ws):
            try:
                async for raw in ws:
                    event = json.loads(raw)
                    received.append(event)
                    if event['type'] == 'session.update':
                        await ws.send(json.dumps({'type': 'session.updated'}))
                    elif event['type'] == 'response.create':
                        await ws.send(json.dumps({'type': 'response.created', 'response': {'id': 'r1'}}))
                        await ws.send(json.dumps({'type': 'response.output_audio.delta', 'response_id': 'r1',
                            'delta': base64.b64encode(b'\x00\x00' * 960).decode()}))
                        await ws.send(json.dumps({'type': 'response.done', 'response': {'id': 'r1', 'status': 'completed'}}))
                    elif followup and event['type'] == 'conversation.item.create' and event['item']['content'][0]['text'] == 'English dub it is.':
                        await ws.send(json.dumps({'type': 'input_audio_buffer.committed', 'item_id': 'followup'}))
                        await ws.send(json.dumps({'type': 'conversation.item.input_audio_transcription.completed',
                                                  'item_id': 'followup', 'transcript': 'Thanks, goodbye.'}))
            finally:
                closed.set()

        async def ask(text, history, context):
            questions.append((text, history, context))
            if fail:
                raise RuntimeError('Hermes offline')
            if stall:
                await asyncio.Future()
            return {'reply': 'Goodbye.' if len(questions) == 2 else 'English dub it is.',
                    'finished': finish or len(questions) == 2}

        async with serve(peer, '127.0.0.1', 0) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f'ws://127.0.0.1:{port}', close_timeout=0.2) as ws:
                session = VoiceSession(EVENT, app_context='Episode one is ready.',
                                       idle_seconds=0.06, max_seconds=0.8)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'Hermes offline'):
                        await session.run(ws, ask, audio)
                    reason = 'failed'
                else:
                    reason = await session.run(ws, ask, audio)
                await asyncio.wait_for(closed.wait(), 1)
        self.assertTrue(audio.interrupted)
        self.assertEqual(len(questions), 2 if followup else 1)
        text, history, context = questions[0]
        self.assertEqual(text, EVENT['text'])
        self.assertEqual(history, EVENT['context'])
        self.assertEqual(context['app_context'], 'Episode one is ready.')
        items = [e['item'] for e in received if e['type'] == 'conversation.item.create']
        self.assertEqual([i['content'][0]['text'] for i in items].count(EVENT['text']), 1)
        self.assertEqual([i['role'] for i in items[:3]], ['user', 'assistant', 'user'])
        self.assertEqual(items[1]['content'][0]['type'], 'output_text')
        self.assertIn('Episode one is ready.', received[0]['session']['instructions'])
        if followup:
            self.assertEqual(questions[1][0], 'Thanks, goodbye.')
            self.assertEqual(questions[1][1][-2:], [{'role': 'user', 'text': EVENT['text']},
                                                  {'role': 'assistant', 'text': 'English dub it is.'}])
        return reason, audio

    async def test_goodbye_finishes_after_speaker_drains(self):
        reason, audio = await self.exercise(finish=True)
        self.assertEqual(reason, 'agent_finished')
        self.assertTrue(audio.drained)
        self.assertEqual(len(audio.played), 1920)

    async def test_silence_really_closes_socket(self):
        reason, audio = await self.exercise()
        self.assertEqual(reason, 'idle')
        self.assertTrue(audio.drained)

    async def test_time_limit_closes_while_agent_is_stuck(self):
        reason, _ = await self.exercise(stall=True)
        self.assertEqual(reason, 'time_limit')

    async def test_failure_still_closes_socket(self):
        await self.exercise(fail=True)

    async def test_followup_keeps_context_and_can_end_session(self):
        reason, _ = await self.exercise(followup=True)
        self.assertEqual(reason, 'agent_finished')

    def test_handoff_ignores_private_classifier_state(self):
        data = handoff({**EVENT, 'recent_conversation': [{'text': 'Private room conversation'}]})
        self.assertNotIn('Private room conversation', json.dumps(data))
        data['messages'][0]['text'] = 'changed'
        self.assertEqual(EVENT['context'][0]['text'], 'We are watching Cowboy Bebop.')


if __name__ == '__main__':
    unittest.main()
