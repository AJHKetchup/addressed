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
SPEECH_NAME = 'vosk-model-small-en-us-0.15'
SPEECH_ARCHIVE_HASH = '30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498'
SPEECH_FILES = {
    'README': 'c0cf286e4f7783306c5f6469b37db69228fb16803b03cae661edb2d7bba64ebb',
    'graph/disambig_tid.int': '9ad87cc166d0998f08f758f47a6223a120dfffcbee805c4849c3aa5e6bb3c0fc',
    'graph/Gr.fst': '023c8b7e30704a9e37765c635c252e608a02f361235bf94abdcf2a5225d85b20',
    'graph/HCLr.fst': '5caafba3081e1646545ac6bff0dd7a318e53dcbdc86f237909ce1d2ac1293d34',
    'am/final.mdl': '75370a0137f9daf8f469dedd7daa4513ae7a621f03240c6e512e2b50b656a7b6',
    'ivector/splice.conf': '9f0c5f7c82d18eaf25d8bce470efa9f7741f88411fe428774bc0a9bb69a24756',
    'ivector/global_cmvn.stats': '33be09afcc80059847a275c3d043b51f1ab954c7c2438ddbbf4745e8ba144ff9',
    'ivector/final.ie': '3f37faf90c375b9e4740b569398b5829ed9cc07d19be6d441f72c3b71d7efcc6',
    'ivector/final.dubm': '8c5d7dd69d2122313baaf19f61f35dd3fa18b70c62ac0687e311e1c46e6daca7',
    'ivector/online_cmvn.conf': 'a2f3571754b64297cb7efb2e7ca3df61995c5a45fcbb97188f90613552bb2dfe',
    'ivector/final.mat': 'ddd83586dc5f928cda8738b922c85ffe38fc789cb5f9151a712ca12f37265382',
    'conf/model.conf': '8f14cb1eeb07c762c371db648c6be688d347236155ca0f64fb13b6567a8ce81f',
    'conf/mfcc.conf': '1e2228006d01d805ad1c267fee9f79709ca87ac51bd82b0e3f5c69ba543f0fc4',
    'graph/phones/word_boundary.int': 'da199d9c991e0e84681ddbb34627b915b26302d50a8fdaa23c51e2bc3a50b5c3',
}


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
    for kind, manifest in [('decision', DECISION)]:
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
    if speech:
        setup_speech(directory)


def setup_speech(directory):
    import zipfile
    folder = Path(directory) / 'speech'
    model = folder / SPEECH_NAME
    if all((model / name).is_file() and digest(model / name) == expected for name, expected in SPEECH_FILES.items()):
        return
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / (SPEECH_NAME + '.download.zip')
    print('Downloading Vosk small English speech model', file=sys.stderr)
    try:
        with urlopen('https://alphacephei.com/vosk/models/' + SPEECH_NAME + '.zip', timeout=60) as source, archive.open('wb') as out:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                out.write(block)
        if digest(archive) != SPEECH_ARCHIVE_HASH:
            raise ValueError('Speech model archive checksum mismatch')
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if not (folder / member).resolve().is_relative_to(model.resolve()):
                    raise ValueError('Invalid speech archive path')
            z.extractall(folder)
        if not all(digest(model / name) == expected for name, expected in SPEECH_FILES.items()):
            raise ValueError('Extracted speech model checksum mismatch')
    finally:
        archive.unlink(missing_ok=True)


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
