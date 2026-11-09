"""ClawsGO Science Agent：核验公开实例文件、冻结清单及压缩包完整性。"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main():
    manifest = json.loads((ROOT / 'instances' / 'file_checksums.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = ROOT / record['path']
        if not path.is_file() or path.stat().st_size != record['bytes']:
            raise ValueError('实例文件不存在或大小不同：' + record['path'])
        if file_hash(path) != record['sha256']:
            raise ValueError('实例文件哈希不一致：' + record['path'])
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8-sig'))
        elif path.name.endswith('.json.gz'):
            with gzip.open(path, 'rt', encoding='utf-8-sig') as stream:
                graph = json.load(stream)
            if not isinstance(graph, dict):
                raise ValueError('完整图导出格式不同：' + record['path'])

    counts = {}
    for hours, directory, expected in [(3, 'dataset/instances', 100), (24, 'dataset/instances/h24', 30)]:
        folder = ROOT / directory
        records = json.loads((folder / 'manifest.json').read_text(encoding='utf-8-sig'))
        paths = list(folder.glob('tasks_*.npz'))
        if len(paths) != expected or len(records) != expected:
            raise ValueError(f'{hours}小时冻结实例数量不正确')
        for row in records.values():
            path = folder / f"tasks_{row['tasks']:04d}_seed_{row['seed']:02d}.npz"
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError('冻结实例压缩内容损坏：' + path.name)
        counts[hours] = len(paths)

    for directory in ['instances/revision_audit', 'instances/revision_audit_m200']:
        folder = ROOT / directory
        records = json.loads((folder / 'manifest.json').read_text(encoding='utf-8-sig'))
        for row in records:
            if file_hash(folder / row['file']) != row['sha256']:
                raise ValueError('历史完整图清单不一致：' + row['file'])

    print(f"实例校验通过：{len(manifest['files'])}个输入文件，100个三小时和30个24小时冻结图。")
    print('已核验全部文件SHA-256、JSON格式、冻结图压缩完整性及历史图导出清单。')
    print('本检查未运行优化实验。')


if __name__ == '__main__':
    main()
