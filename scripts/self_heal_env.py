#!/usr/bin/env python3
"""Self-heal environment script.

Usage:
  python scripts/self_heal_env.py            # audit & fix
  python scripts/self_heal_env.py --dry-run  # only report

What it does:
  1. Reads requirements.txt.
  2. Checks importability of key packages.
  3. Pins missing/mismatched core versions (pydantic / pydantic-core pairing).
  4. Optionally reinstalls missing packages.
  5. Writes a summary JSON report.

Safe: It only installs if not present or clearly broken.
Set NO_INSTALL=1 to force read-only audit.
"""
from __future__ import annotations
import argparse, importlib, json, os, re, subprocess, sys, pathlib
from dataclasses import dataclass
from typing import List, Dict, Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
REQ_FILE = ROOT / 'requirements.txt'
REPORT_FILE = ROOT / 'env_audit_report.json'

CRITICAL_PAIRS = {
    'pydantic': 'pydantic_core',
}

@dataclass
class PackageStatus:
    name: str
    required_version: str | None
    installed_version: str | None
    ok: bool
    action: str | None = None
    notes: str | None = None


def parse_requirements(path: pathlib.Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        m = re.match(r'([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)', line)
        if m:
            pkg, ver = m.groups()
            mapping[pkg.lower()] = ver
    return mapping


def get_installed_version(pkg: str) -> str | None:
    try:
        module = importlib.import_module(pkg)
        return getattr(module, '__version__', None)
    except Exception:
        return None


def check_packages(reqs: Dict[str,str]) -> List[PackageStatus]:
    statuses: List[PackageStatus] = []
    for pkg, ver in sorted(reqs.items()):
        installed = get_installed_version(pkg)
        ok = installed == ver
        action = None
        notes = None
        if installed is None:
            ok = False
            action = f'install {pkg}=={ver}'
        elif installed != ver:
            # For non-critical packages we allow drift; mark upgrade path
            action = f'consider reinstall {pkg}=={ver}'
            notes = f'version drift: {installed} (wanted {ver})'
        statuses.append(PackageStatus(pkg, ver, installed, ok, action, notes))
    # Pair sanity (pydantic/pydantic_core): require coherent minor
    if 'pydantic' in reqs and 'pydantic_core' in reqs:
        pyd_v = get_installed_version('pydantic')
        core_v = get_installed_version('pydantic_core')
        if pyd_v and core_v:
            # Rough minor alignment check
            pv_minor = '.'.join(pyd_v.split('.')[:2])
            cv_minor = '.'.join(core_v.split('.')[:2])
            if pv_minor != cv_minor:
                statuses.append(PackageStatus('pydantic_pair', None, f'{pyd_v}/{core_v}', False, 'reinstall pydantic & pydantic-core', 'Minor versions mismatch'))
    return statuses


def perform_actions(statuses: List[PackageStatus], dry_run: bool, no_install: bool) -> List[str]:
    executed: List[str] = []
    for st in statuses:
        if not st.action: continue
        if no_install: continue
        if dry_run: continue
        cmd = [sys.executable, '-m', 'pip'] + st.action.split()
        try:
            subprocess.run(cmd, check=True)
            executed.append(' '.join(cmd))
        except subprocess.CalledProcessError as e:
            executed.append(f'FAILED:{" ".join(cmd)} code={e.returncode}')
    return executed


def main():
    parser = argparse.ArgumentParser(description='Audit & self-heal Python env for RAG system')
    parser.add_argument('--dry-run', action='store_true', help='Only report, do not install')
    args = parser.parse_args()
    no_install = os.getenv('NO_INSTALL') == '1'

    reqs = parse_requirements(REQ_FILE)
    statuses = check_packages(reqs)
    executed = perform_actions(statuses, args.dry_run, no_install)

    report = {
        'root': str(ROOT),
        'dry_run': args.dry_run,
        'no_install_env': no_install,
        'package_count': len(statuses),
        'actions_executed': executed,
        'packages': [st.__dict__ for st in statuses],
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f'Report written to {REPORT_FILE}')

if __name__ == '__main__':
    main()
