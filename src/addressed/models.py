"""Pinned, checksum-verified weights; downloads happen only on explicit setup."""
import hashlib
import os
from pathlib import Path
import sys
from urllib.request import urlopen

DECISION = ('LiquidAI/d1-omni-600M-GGUF',
            '8eb2d2a1c5404bba8a34a88bd8e159c242197d51', {
    'decision.gguf': ('d1-omni-600M-Q8_0.gguf', 'cd94463f4cac9c700ec6750df8df005bb6846de410e159d1f5e985e1171693d3'),
})
RUNTIMES = {
    ('Linux', 'x86_64'): ('ubuntu-x64.tar.gz', 'cf07c1c112d709bd67e522017471de57ed0ec8bea031d732c5422351c07f65ac'),
    ('Windows', 'AMD64'): ('win-cpu-x64.zip', '37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b'),
    ('Darwin', 'arm64'): ('macos-arm64.tar.gz', 'dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd'),
    ('Darwin', 'x86_64'): ('macos-x64.tar.gz', 'cab28238d78158109b48deb27b05d50e0dcef56275d2e76da9ca8281526c7532'),
}
SPEECH = ('csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17',
          'd42f2d9f7ca24806fb667456a18a9f1b60f70d16', {
    'tokens.txt': ('tokens.txt', '49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb'),
    'encoder-epoch-99-avg-1.int8.onnx': ('encoder-epoch-99-avg-1.int8.onnx', '3810755ce7c3ab26b42a8bcf39d191308fa27fb0f53358823ba46141d03b7eb3'),
    'decoder-epoch-99-avg-1.onnx': ('decoder-epoch-99-avg-1.onnx', '45a7f940ecfb53d89fa270ad11b88b961e53a317203eb24b1c8e95ed208b0f30'),
    'joiner-epoch-99-avg-1.int8.onnx': ('joiner-epoch-99-avg-1.int8.onnx', 'e085d73b593cf9b0707f370dbd656d58327d3fe36d80d849202ef81df02cb01e'),
})


def default_directory():
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    return cache / 'addressed'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def setup(directory, speech=False):
    for kind, manifest in [('decision', DECISION)] + ([('speech', SPEECH)] if speech else []):
        repository, revision, files = manifest
        folder = Path(directory) / kind
        folder.mkdir(parents=True, exist_ok=True)
        for filename, (remote, expected) in files.items():
            target = folder / filename
            if target.exists() and digest(target) == expected:
                continue
            print('Downloading ' + kind + '/' + filename, file=sys.stderr)
            temporary = target.with_suffix(target.suffix + '.download')
            try:
                with urlopen(f'https://huggingface.co/{repository}/resolve/{revision}/{remote}', timeout=60) as response, temporary.open('wb') as output:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        output.write(block)
                if digest(temporary) != expected:
                    raise ValueError('Checksum mismatch: ' + filename)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def setup_runtime(directory):
    import platform
    import tarfile
    import zipfile
    folder = Path(directory) / 'runtime'
    if list(folder.rglob('llama-server')) or list(folder.rglob('llama-server.exe')):
        return
    key = (platform.system(), platform.machine())
    if key not in RUNTIMES:
        raise ValueError('Install llama.cpp for your platform and use --decision-url')
    asset, expected = RUNTIMES[key]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / asset
    print('Downloading llama.cpp b11669 CPU runtime', file=sys.stderr)
    try:
        with urlopen('https://github.com/ggml-org/llama.cpp/releases/download/b11669/llama-b11669-bin-' + asset, timeout=60) as source, archive.open('wb') as out:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                out.write(block)
        if digest(archive) != expected:
            raise ValueError('Runtime checksum mismatch')
        if asset.endswith('.zip'):
            with zipfile.ZipFile(archive) as z:
                for member in z.namelist():
                    if not (folder / member).resolve().is_relative_to(folder.resolve()):
                        raise ValueError('Invalid runtime archive path')
                z.extractall(folder)
        else:
            with tarfile.open(archive) as t:
                t.extractall(folder, filter='data')
    finally:
        archive.unlink(missing_ok=True)
