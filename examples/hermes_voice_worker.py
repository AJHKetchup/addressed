"""Run in Hermes' own Python environment. Only conversation-ending tools are enabled."""
from contextlib import redirect_stdout
import json
import logging
import sys


def main():
    logging.disable(logging.CRITICAL)
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from tools.registry import registry
    from run_agent import AIAgent

    finished = False

    def finish(args, **kwargs):
        nonlocal finished
        finished = True
        return json.dumps({'ending': True})

    registry.register(name='finish_conversation', toolset='addressed_session', handler=finish,
        schema={'name': 'finish_conversation',
                'description': 'End when the user says goodbye, is done, or asks to stop listening. Not after every command.',
                'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}})
    with redirect_stdout(sys.stderr):
        config = load_config()
        model = config['model']['default']
        runtime = resolve_runtime_provider(target_model=model)
        agent = AIAgent(model=model, provider=runtime['provider'], api_mode=runtime.get('api_mode'),
            base_url=runtime.get('base_url'), api_key=runtime.get('api_key'),
            enabled_toolsets=['addressed_session'], quiet_mode=True, max_iterations=6,
            skip_memory=True, skip_context_files=True, load_soul_identity=False,
            save_trajectories=False, max_tokens=600)
        if agent.valid_tool_names != {'finish_conversation'}:
            raise RuntimeError('This example may expose only the finish_conversation tool')
    print(json.dumps({'ready': True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        finished = False
        instructions = ('You are a conversational voice assistant running in Hermes. Speak briefly in plain speech. '
            'Ask natural clarifying questions and use the supplied conversation context. '
            'When the user is done or asks you to stop listening, call finish_conversation and give a short farewell. '
            'Do not finish merely because you answered one request. '
            'This example has no action tools; never claim to control devices or download media.\n'
            'Assistant purpose: ' + request['context']['purpose'] + '\nApplication context: '
            + request['context']['app_context'])
        # Gate uses text; Hermes' native history expects content.
        history = [{'role': turn['role'], 'content': turn['text']} for turn in request['history']]
        with redirect_stdout(sys.stderr):
            result = agent.run_conversation(request['text'], system_message=instructions,
                                             conversation_history=history)
        if not result.get('completed'):
            raise RuntimeError('Hermes did not complete this turn')
        print(json.dumps({'reply': result['final_response'], 'finished': finished}), flush=True)


if __name__ == '__main__':
    main()
