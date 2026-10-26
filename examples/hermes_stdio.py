"""Run with Hermes' existing Python environment; accepted JSONL arrives on stdin.

Example: addressed text | /path/to/hermes/venv/bin/python examples/hermes_stdio.py
This example answers text only; choose your own toolset when adding real actions.
"""
from contextlib import redirect_stdout
import json
import sys

from hermes_cli.config import load_config
from hermes_cli.runtime_provider import resolve_runtime_provider
from run_agent import AIAgent


agent = None
session_id = None
for line in sys.stdin:
    event = json.loads(line)
    if event.get('type') not in ('activate', 'continue'):
        continue
    with redirect_stdout(sys.stderr):
        if session_id != event['session_id']:
            config = load_config()
            model = config['model']['default']
            runtime = resolve_runtime_provider(target_model=model)
            agent = AIAgent(model=model, provider=runtime['provider'],
                            api_mode=runtime.get('api_mode'), base_url=runtime.get('base_url'),
                            api_key=runtime.get('api_key'), enabled_toolsets=[],
                            quiet_mode=True, skip_memory=True, skip_context_files=True,
                            save_trajectories=False)
            if agent.valid_tool_names:
                raise RuntimeError('This text-only example must not expose action tools')
            session_id = event['session_id']
        history = [{'role': turn['role'], 'content': turn['text']} for turn in event.get('context', [])]
        result = agent.run_conversation(event['text'], conversation_history=history)
    print(json.dumps({'role': 'assistant', 'text': result['final_response']}), flush=True)
