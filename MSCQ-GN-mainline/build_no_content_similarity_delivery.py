"""Package completed, independently audited local experiment outputs."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
RESULT = ROOT/'Result/no_content_similarity_seed0_20261011'
FIGURES = ROOT/'Visualization/no_content_similarity_seed0_20261011'
ARCHIVE = ROOT/'Visualization/no_content_similarity_seed0_20261011.zip'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    complete = read(RESULT/'COMPLETE.json')
    assert complete['audit']['passed'] and complete['audit']['verified_checkpoints'] == 20
    summary = read(RESULT/'summary.json')
    assert summary['all_complete'] and summary['completed_new_runs'] == 10
    assert read(RESULT/'independent_audit.json')['passed']
    plan = read(RESULT/'PLAN.json')
    for name, expected in plan['code_sha256'].items():
        assert digest(ROOT/name) == expected, name
        assert digest(RESULT/'source_snapshot'/name) == expected, name
    for name, expected in plan['reference_manifest'].items():
        assert digest(ROOT/name) == expected, name
    assert len(list(FIGURES.glob('*.png'))) == 3
    assert (FIGURES/'figures_metadata.json').is_file()
    files = [path for folder in (RESULT, FIGURES) for path in folder.rglob('*')
             if path.is_file() and path.name not in ('delivery_manifest.json', 'RUNNING.lock')
             and path.suffix != '.tmp' and '__pycache__' not in path.parts]
    # Include unchanged C predictions/curves/protocols for portable pair audits.
    # Original C weight files stay in their existing workspace reference.
    files.extend(ROOT/name for name in plan['reference_manifest']
                 if Path(name).suffix != '.pt')
    for name in ('Model/no_content_similarity_control.py',
                 'no_content_similarity_seed0_experiment.py',
                 'analyze_no_content_similarity_control.py',
                 'plot_no_content_similarity_control.py',
                 'tests/test_no_content_similarity_control.py',
                 'audit_no_content_similarity_control.py',
                 'build_no_content_similarity_delivery.py'):
        files.append(ROOT/name)
    files = sorted(set(files), key=lambda path: str(path.relative_to(ROOT)))
    manifest = {path.relative_to(ROOT).as_posix():
                {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in files}
    target = FIGURES/'delivery_manifest.json'
    target.write_text(json.dumps(dict(files=manifest, all_complete=True,
        new_runs=10, reused_runs=10, includes_new_checkpoints=True),
        ensure_ascii=False, indent=2), encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for path in files:
            output.write(path, path.relative_to(ROOT).as_posix())
        output.write(target, target.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == len(files) + 1
        for name, expected in manifest.items():
            data = package.read(name)
            assert len(data) == expected['bytes']
            assert hashlib.sha256(data).hexdigest() == expected['sha256']
    print(json.dumps(dict(archive=str(ARCHIVE), entries=len(files)+1,
        bytes=ARCHIVE.stat().st_size, checked=True)))


if __name__ == '__main__':
    main()
