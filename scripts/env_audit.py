"""
Environment audit helper
Prints python/conda/pip info, installed packages, and searches for duplicate package entries
Usage: python scripts/env_audit.py > ~/rereg_logs/env_audit.log
"""
import sys
import platform
import subprocess
import json
from importlib import metadata

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR:\n{e.output}"

def python_info():
    return {
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'implementation': platform.python_implementation()
    }

def pip_info():
    return {
        'pip_version': run_cmd('python -m pip --version'),
        'pip_cache_dir': run_cmd('python -m pip cache dir')
    }

def conda_info():
    conda = run_cmd('conda --version')
    if conda.startswith('ERROR'):
        return {'conda': None}
    return {
        'conda_version': conda,
        'conda_list': run_cmd('conda list --json')
    }

def installed_packages():
    pkgs = []
    try:
        for dist in metadata.distributions():
            pkgs.append({'name': dist.metadata['Name'], 'version': dist.version, 'location': str(dist.locate_file(''))})
    except Exception:
        # fallback to pip freeze
        pip_freeze = run_cmd('python -m pip freeze')
        for line in pip_freeze.splitlines():
            if '==' in line:
                name, ver = line.split('==', 1)
                pkgs.append({'name': name, 'version': ver, 'location': ''})
    return pkgs

def find_duplicates(pkgs):
    by_name = {}
    for p in pkgs:
        n = p['name'].lower()
        by_name.setdefault(n, []).append(p)
    duplicates = {n: v for n, v in by_name.items() if len(v) > 1}
    return duplicates

if __name__ == '__main__':
    out = {}
    out['python'] = python_info()
    out['pip'] = pip_info()
    out['conda'] = conda_info()
    pkgs = installed_packages()
    out['installed_count'] = len(pkgs)
    out['installed_packages_sample'] = pkgs[:80]
    out['duplicates'] = find_duplicates(pkgs)

    print(json.dumps(out, indent=2, ensure_ascii=False))
