import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .gate import Gate, LocalDecisionModel
from .models import default_directory, setup, setup_runtime


def parser():
    p = argparse.ArgumentParser(description='A local, context-aware front door for voice agents.')
    commands = p.add_subparsers(dest='command', required=True)
    install = commands.add_parser('setup', help='Explicitly download pinned local models and CPU runtime')
    install.add_argument('--speech', action='store_true', help='Also download the 40 MB English streaming speech model')
    install.add_argument('--models', type=Path, default=default_directory())
    serve = commands.add_parser('serve', help='Run the local decision model; localhost only')
    serve.add_argument('--models', type=Path, default=default_directory())
    serve.add_argument('--threads', type=int, default=2)
    for command, help_text in [('text', 'Read JSONL or plain transcript lines from stdin'),
                               ('listen', 'Transcribe a microphone or WAV locally and gate each utterance')]:
        c = commands.add_parser(command, help=help_text)
        c.add_argument('--decision-url', default='http://127.0.0.1:8017')
        c.add_argument('--purpose', default='Answer questions and perform digital tasks.')
        c.add_argument('--threshold', type=float, default=0.90)
        c.add_argument('--followup-threshold', type=float, default=0.80)
        c.add_argument('--decisions', action='store_true', help='Print all transcript decisions to stderr; may reveal ambient speech locally')
        if command == 'text':
            c.add_argument('--jsonl', action='store_true', help='Accept structured events from a trusted integration; otherwise every line is literal speech')
        if command == 'listen':
            c.add_argument('--models', type=Path, default=default_directory())
            c.add_argument('--device', help='PortAudio input device name or index')
            c.add_argument('--wav', type=Path, help='Replay a mono PCM16 WAV instead of a microphone')
    commands.add_parser('devices', help='List available microphone devices')
    return p


def process_event(gate, event, explain=False):
    if isinstance(event, str):
        event = {'text': event}
    if not isinstance(event, dict):
        raise ValueError('Use a transcript string or JSON object')
    role = event.get('role', 'user')
    if role == 'assistant':
        gate.assistant_said(event['text'])
        return
    if role == 'control' and event.get('action') == 'reset':
        gate.reset()
        return
    if role != 'user':
        raise ValueError('Unknown event role')
    started = time.monotonic()
    result = gate.decide(event['text'], source=event.get('source', 'microphone'))
    if explain:
        print(json.dumps({'text': event['text'], **result, 'decision_ms': round((time.monotonic() - started) * 1000)}), file=sys.stderr, flush=True)
    if result['reason'] == 'decision_unavailable':
        print('Decision unavailable; ignored utterance. Start addressed serve and check its model.', file=sys.stderr)
    if result['event']:
        print(json.dumps(result['event']), flush=True)


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.command == 'setup':
            setup_runtime(args.models)
            setup(args.models, speech=args.speech)
            print('Models verified. Run addressed serve in one terminal, then addressed listen in another.')
            return
        if args.command == 'serve':
            if not 1 <= args.threads <= 64:
                raise ValueError('Use 1..64 CPU threads')
            names = ('llama-server.exe', 'llama-server')
            candidates = [p for name in names for p in (args.models / 'runtime').rglob(name)]
            if not candidates:
                raise ValueError('Run addressed setup first')
            model = args.models / 'decision' / 'decision.gguf'
            if not model.is_file():
                raise ValueError('Run addressed setup first')
            command = [str(candidates[0]), '-m', str(model), '--host', '127.0.0.1', '--port', '8017',
                       '-t', str(args.threads), '-c', '2048', '-b', '2048', '-ub', '2048',
                       '--no-warmup', '--alias', 'jev-latest']
            child = subprocess.Popen(command)
            try:
                raise SystemExit(child.wait())
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
        if args.command == 'devices':
            import sounddevice as sd
            print(sd.query_devices())
            return
        gate = Gate(LocalDecisionModel(args.decision_url), purpose=args.purpose,
                    threshold=args.threshold, followup_threshold=args.followup_threshold)
        if args.command == 'text':
            for line in sys.stdin:
                line = line.rstrip('\r\n')
                if not line.strip():
                    continue
                try:
                    event = json.loads(line) if args.jsonl else line
                    process_event(gate, event, args.decisions)
                except (ValueError, KeyError, TypeError):
                    print('Ignored malformed input event.', file=sys.stderr)
        elif args.command == 'listen':
            from .speech import Transcriber, microphone
            transcriber = Transcriber(args.models)
            device = int(args.device) if args.device and args.device.isdigit() else args.device
            source = transcriber.wav(args.wav) if args.wav else microphone(transcriber, device)
            print('Listening locally. Ctrl+C stops capture. Only accepted requests go to stdout.', file=sys.stderr)
            for text in source:
                process_event(gate, text, args.decisions)
    except KeyboardInterrupt:
        pass
    except (ValueError, FileNotFoundError, OSError, ImportError) as error:
        p.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
