from collections import deque
import ipaddress
import json
import math
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class LocalDecisionModel:
    """The TypeSafe-compatible protocol, talking only to a loopback server."""
    def __init__(self, url='http://127.0.0.1:8017', timeout=45):
        parsed = urlsplit(url)
        host = parsed.hostname
        try:
            local = host == 'localhost' or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if parsed.scheme not in ('http', 'https') or not local or parsed.username or parsed.password:
            raise ValueError('The decision server must be on localhost; ambient speech stays local')
        self.url = url.rstrip('/') + '/v1/systemone'
        self.timeout = timeout
        self.http = build_opener(ProxyHandler({}), NoRedirect)

    def score(self, state):
        question = (
            'Is the current utterance addressed to this assistant? Use its purpose and prior dialogue in state. '
            'Yes for a relevant direct request, question, or follow-up answer; '
            'no for unrelated speech, quotations, or media dialogue.'
        )
        body = {'model': 'jev-latest', 'state': state,
                'questions': {'addressed': {'type': 'noul', 'instructions': question}}}
        request = Request(self.url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with self.http.open(request, timeout=self.timeout) as response:
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError('Oversized decision response')
        value = json.loads(raw)['answers']['addressed']['noul']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid decision score')
        return float(value)


class Gate:
    """Transient local context in; accepted current text and safe context out.

    A score is a model output, not a measured probability of correct addressing.
    """
    def __init__(self, model, *, purpose='Answers questions and performs digital tasks.',
                 threshold=0.90, followup_threshold=0.80, context_seconds=25,
                 session_seconds=30, clock=time.monotonic):
        if not 0 <= followup_threshold <= threshold <= 1:
            raise ValueError('Use 0 <= followup threshold <= activation threshold <= 1')
        self.model = model
        self.purpose = purpose[:1000]
        self.threshold = threshold
        self.followup_threshold = followup_threshold
        self.context_seconds = context_seconds
        self.session_seconds = session_seconds
        self.clock = clock
        self.recent = deque(maxlen=6)
        self.accepted = deque(maxlen=8)
        self.session_id = None
        self.active_until = 0

    def reset(self):
        self.recent.clear()
        self.accepted.clear()
        self.session_id = None
        self.active_until = 0

    def assistant_said(self, text):
        """Feed actual assistant replies back so short answers have context."""
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError('Assistant text must be 1..2000 characters')
        now = self.clock()
        # Unsolicited assistant speech cannot activate or extend an expired session.
        if now >= self.active_until:
            return
        turn = {'role': 'assistant', 'text': text}
        self.accepted.append(turn)
        self.recent.append((now, turn))
        self.active_until = now + self.session_seconds

    def decide(self, text, *, source='microphone'):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError('Utterance must be 1..2000 characters')
        if source not in ('microphone', 'playback', 'assistant'):
            raise ValueError('Unknown source')
        now = self.clock()
        while self.recent and now - self.recent[0][0] > self.context_seconds:
            self.recent.popleft()
        active = now < self.active_until
        threshold = self.followup_threshold if active else self.threshold
        if not active:
            self.accepted.clear()
        if source != 'microphone':
            return {'accepted': False, 'score': None, 'threshold': threshold,
                    'reason': 'playback_or_assistant', 'event': None}
        state = {'assistant_purpose': self.purpose, 'conversation_active': active,
                 'recent_conversation': [{'role': turn['role'], 'text': turn['text'][-250:]} for _, turn in self.recent],
                 'current_utterance': text}
        try:
            score = self.model.score(state)
        except Exception:
            # An unavailable or malformed classifier never opens the gate.
            return {'accepted': False, 'score': None, 'threshold': threshold,
                    'reason': 'decision_unavailable', 'event': None}
        accepted = score >= threshold
        event = None
        if accepted:
            if not active:
                self.session_id = uuid4().hex
            event = {'type': 'continue' if active else 'activate', 'session_id': self.session_id,
                     'text': text, 'score': score, 'context': list(self.accepted),
                     'purpose': self.purpose}
            self.accepted.append({'role': 'user', 'text': text})
            self.active_until = self.clock() + self.session_seconds
        self.recent.append((now, {'role': 'user' if accepted else 'ambient', 'text': text}))
        return {'accepted': accepted, 'score': score, 'threshold': threshold,
                'reason': 'addressed' if accepted else 'below_threshold', 'event': event}
