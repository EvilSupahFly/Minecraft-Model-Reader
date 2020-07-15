import os
import shutil
import zipfile
import json
from urllib.request import urlopen
import io
from typing import Generator

from . import launcher_manifest
import minecraft_model_reader
from minecraft_model_reader import log
from .java_rp_handler import JavaRP


def generator_unpacker(gen: Generator):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def get_latest() -> JavaRP:
    return generator_unpacker(
        get_latest_iter()
    )


def get_latest_iter() -> Generator[float, None, JavaRP]:
    vanilla_rp_path = os.path.join(minecraft_model_reader.path, 'resource_packs', 'java_vanilla')
    new_version = launcher_manifest['latest']['release']
    if new_version is None:
        if os.path.isdir(vanilla_rp_path):
            log.error('Could not download the launcher manifest. The resource pack seems to be present so using that.')
        else:
            log.error('Could not download the launcher manifest. The resource pack is not present, blocks may not be rendered correctly.')
    else:
        if os.path.isdir(vanilla_rp_path):
            if os.path.isfile(os.path.join(vanilla_rp_path, 'version')):
                with open(os.path.join(vanilla_rp_path, 'version')) as f:
                    old_version = f.read()
                if old_version != new_version:
                    yield from _remove_and_download_iter(vanilla_rp_path, new_version)
            else:
                yield from _remove_and_download_iter(vanilla_rp_path, new_version)
        else:
            yield from _remove_and_download_iter(vanilla_rp_path, new_version)
    return JavaRP(vanilla_rp_path)


_java_vanilla_fix = None
_java_vanilla_latest = None


def get_java_vanilla_fix():
    global _java_vanilla_fix
    if _java_vanilla_fix is None:
        _java_vanilla_fix = JavaRP(os.path.join(minecraft_model_reader.path, 'resource_packs', 'java_vanilla_fix'))
    return _java_vanilla_fix


def get_java_vanilla_latest():
    global _java_vanilla_latest
    if _java_vanilla_latest is None:
        _java_vanilla_latest = get_latest()
    return _java_vanilla_latest


def get_java_vanilla_latest_iter() -> Generator[float, None, JavaRP]:
    global _java_vanilla_latest
    if _java_vanilla_latest is None:
        _java_vanilla_latest = yield from get_latest_iter()
    return _java_vanilla_latest


def _remove_and_download(path, version):
    for _ in _remove_and_download_iter(path, version):
        pass


def _remove_and_download_iter(path, version) -> Generator[float, None, None]:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    exists = yield from download_resources_iter(path, version)
    if exists:
        with open(os.path.join(path, 'version'), 'w') as f:
            f.write(version)


def download_resources(path, version) -> bool:
    return generator_unpacker(
        download_resources_iter(path, version)
    )


def download_resources_iter(path, version, chunk_size=4096) -> Generator[float, None, bool]:
    log.info(f'Downloading Java resource pack for version {version}')
    version_url = next((v["url"] for v in launcher_manifest['versions'] if v['id'] == version), None)
    if version_url is None:
        log.error(f'Could not find Java resource pack for version {version}.')
        return False

    try:
        version_manifest = json.load(urlopen(version_url))
        version_client_url = version_manifest["downloads"]["client"]["url"]

        response = urlopen(version_client_url)
        data = []
        data_size = int(response.headers["content-length"].strip())
        index = 0
        chunk = b"hello"
        while chunk:
            chunk = response.read(chunk_size)
            data.append(chunk)
            index += 1
            yield min(
                1.0,
                (index * chunk_size) / data_size
            )

        client = zipfile.ZipFile(io.BytesIO(
            b"".join(data)
        ))
        for fpath in client.namelist():
            if fpath.startswith('assets/'):
                client.extract(fpath, path)
        client.extract('pack.mcmeta', path)
        client.extract('pack.png', path)

    except:
        log.error(
            f'Failed to download and extract the Java resource pack for version {version}. Make sure you have a connection to the internet.',
            exc_info=True
        )
        return False
    log.info(f'Finished downloading Java resource pack for version {version}')
    return True
