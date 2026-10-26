"""Privacy and session contracts, not an evaluation of model accuracy."""
import unittest

from addressed import Gate, LocalDecisionModel


class Backend:
    def score(self, state):
        # Stand in for model IO; these checks exercise handoff state, not detection.
        return {'ambient private detail': 0.1, 'Play the next episode.': 0.99, 'Yes.': 0.85}[state['current_utterance']]


class Contract(unittest.TestCase):
    def test_only_accepted_history_crosses_handoff(self):
        gate = Gate(Backend())
        gate.decide('ambient private detail')
        first = gate.decide('Play the next episode.')['event']
        self.assertEqual(first['text'], 'Play the next episode.')
        self.assertEqual(first['context'], [])
        gate.assistant_said('Do you mean episode six?')
        following = gate.decide('Yes.')['event']
        self.assertEqual(following['session_id'], first['session_id'])
        self.assertEqual(following['context'], [
            {'role': 'user', 'text': 'Play the next episode.'},
            {'role': 'assistant', 'text': 'Do you mean episode six?'}])

    def test_expiry_and_reset_clear_handoff_context(self):
        now = [0]
        gate = Gate(Backend(), clock=lambda: now[0])
        first = gate.decide('Play the next episode.')['event']
        now[0] = 31
        next_event = gate.decide('Play the next episode.')['event']
        self.assertNotEqual(first['session_id'], next_event['session_id'])
        self.assertEqual(next_event['context'], [])
        gate.reset()
        self.assertFalse(gate.accepted)
        self.assertFalse(gate.recent)

    def test_backend_failure_and_known_playback_do_not_activate(self):
        class Broken:
            def score(self, state):
                raise OSError('offline')
        gate = Gate(Broken())
        self.assertIsNone(gate.decide('Play the next episode.')['event'])
        self.assertEqual(gate.decide('Play the next episode.', source='playback')['reason'], 'playback_or_assistant')

    def test_cloud_endpoint_is_rejected(self):
        for url in ['https://example.com', 'http://192.168.1.5:8017', 'http://127.0.0.1@remote.test']:
            with self.assertRaises(ValueError):
                LocalDecisionModel(url)


if __name__ == '__main__':
    unittest.main()
