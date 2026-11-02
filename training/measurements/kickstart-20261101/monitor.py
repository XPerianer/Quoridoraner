"""Ten-minute read-only monitor of one kickstart arm; backs up snapshots and resumable state locally.

python monitor.py <resnet|transformer>   (artifacts/<arm>/, status in monitor-<arm>.json)
"""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

ARM = sys.argv[1]
SESSION = f'qkick-{ARM}'
HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts' / ARM
SSH = ['ssh', '-o', f'ProxyCommand=colab ssh --proxy-mode -s {SESSION}', '-o', 'StrictHostKeyChecking=no',
       '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4', '-o', 'ConnectTimeout=60',
       '-o', f'UserKnownHostsFile=/tmp/{SESSION}-hosts2', 'root@colab-runtime']
QUERY = f'''import json,pathlib
r=pathlib.Path('/content/kick-{ARM}')
s=(r/'STATUS').read_text().strip() if (r/'STATUS').exists() else 'starting'
m=r/'rl'/'metrics.json'
rows=json.loads(m.read_text()) if m.exists() else []
last=rows[-1] if rows else {{}}
anchors=[(x['iteration'],x['anchor_eval']['score']) for x in rows if 'anchor_eval' in x]
models=sorted(p.name for p in (r/'snapshots').glob('model-*.pt'))
evals={{p.name:json.loads(p.read_text()).get('summary',{{}}).get('score') for p in (r/'eval').glob('*.json')}}
print(json.dumps(dict(status=s,iteration=last.get('iteration'),hours=sum(x['seconds'] for x in rows)/3600,
 teacher_weight=last.get('teacher_weight'),teacher_kl=last.get('teacher_kl'),policy=last.get('policy'),
 value=last.get('value'),seconds=last.get('seconds'),anchors=anchors,models=models,evals=evals)))'''
BUNDLE = f'''import pathlib,subprocess
r=pathlib.Path('/content/kick-{ARM}')
paths=[]
for p in r.rglob('*'):
 if not p.is_file() or '.tmp' in p.name: continue
 rel=p.relative_to(r)
 if rel.parts[0]=='rl' and p.name=='state.pt': continue
 if rel.parts[0]=='snapshots' and p.name.startswith('state-h'): continue
 paths.append(str(rel))
subprocess.run(['tar','-cf','-','-C',str(r),'--']+paths,check=True)'''


def remote(script, **kwargs):
    return subprocess.run(SSH + ['python -c ' + shlex.quote(script)], check=True, timeout=1200, **kwargs)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    backed, last_backup = None, 0
    while True:
        try:
            sessions = subprocess.run(['colab', 'sessions'], capture_output=True, text=True, check=True, timeout=60).stdout
            if f'[{SESSION}]' not in sessions:
                raise RuntimeError('Runtime missing; not reconnecting (SSH could create a replacement session)')
            status = json.loads(remote(QUERY, capture_output=True, text=True).stdout)
            status['checked_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
            (HERE / f'monitor-{ARM}.json').write_text(json.dumps(status, indent=2) + '\n')
            print(status['checked_utc'], status['status'], 'it', status['iteration'], f"{status['hours']:.2f}h",
                  'lambda', status['teacher_weight'], 'anchors', status['anchors'][-3:], flush=True)
            key = (tuple(status['models']), tuple(sorted(status['evals'])))
            if key != backed or status['status'] in ('DONE', 'FAILED') or time.monotonic() - last_backup >= 1200:
                temporary = ART.parent / f'{ARM}-backup.tar.tmp'
                with temporary.open('wb') as stream:
                    remote(BUNDLE, stdout=stream)
                # Extract beside the backup, then replace file by file: an interrupted transfer fails in
                # extractall and never truncates the last good state-latest.
                incoming = ART.parent / f'{ARM}-incoming'
                shutil.rmtree(incoming, ignore_errors=True)
                with tarfile.open(temporary) as archive:
                    archive.extractall(incoming, filter='data')
                for path in sorted(p for p in incoming.rglob('*') if p.is_file()):
                    target = ART / path.relative_to(incoming)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.name == 'rl.log' and target.exists():
                        # A resumed VM starts a new log; keep earlier VMs' logs.
                        old = target.with_name(f'rl-before-{time.strftime("%Y%m%dT%H%M%S")}.log')
                        if target.stat().st_size > path.stat().st_size:
                            os.replace(target, old)
                    os.replace(path, target)
                shutil.rmtree(incoming)
                temporary.unlink()
                backed, last_backup = key, time.monotonic()
                print('backup extracted', flush=True)
            if status['status'] in ('DONE', 'FAILED'):
                return
        except Exception as error:
            print(time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()), 'error:', error, flush=True)
        time.sleep(600)


if __name__ == '__main__':
    main()
