"""Ten-minute read-only monitor; back up hourly checkpoints and resumable state locally."""
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

HERE = Path(__file__).resolve().parent
ART = HERE / 'artifacts'
SSH = ['ssh', '-o', 'ProxyCommand=colab ssh --proxy-mode -s qtransformer-night',
       '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/tmp/qtransformer-night-hosts',
       'root@colab-runtime']
QUERY = '''import json,pathlib,re,subprocess
r=pathlib.Path('/content/transformer-night')
s=(r/'STATUS').read_text().strip()
logs={p.name:p.read_text()[-3000:] for p in [r/'gen.log',r/'pretrain.log',r/'rl.log',r/'pipeline.log'] if p.exists()}
models=sorted(p.name for p in (r/'snapshots').glob('*.pt'))
evals={}
for p in (r/'eval').glob('*.json'):
 try:
  d=json.loads(p.read_text());evals[p.name]=d
 except ValueError: pass
processes=[]
for line in subprocess.check_output(['ps','-eo','pid=,stat=,etime=,args='],text=True).splitlines():
 fields=line.split(None,3)
 if len(fields)!=4: continue
 args=fields[3]
 if re.match(r'^python[0-9.]* -u training/(az.py rl|evaluate_transformer.py)',args) or args.startswith('bash /content/transformer-night/run.sh'):
  processes.append(dict(pid=int(fields[0]),state=fields[1],elapsed=fields[2],command=args))
print(json.dumps(dict(status=s,models=models,logs=logs,evaluations=evals,processes=processes)))'''
BUNDLE = '''import pathlib,subprocess,sys
r=pathlib.Path('/content/transformer-night')
paths=[]
for p in r.rglob('*'):
 if not p.is_file(): continue
 rel=p.relative_to(r)
 if p.name.startswith('gen') and p.name!='gen.log': continue
 if rel.parts[0] in ('resnet',): continue
 if p.name.endswith(('.npz','.tmp')) or '.tmp' in p.name: continue
 if rel.parts[0]=='snapshots' and p.name.startswith('state-h'): continue
 if rel.parts[0]=='rl' and p.name=='state.pt': continue
 if p.suffix in ('.pt','.gz','.json','.log','.txt','.sh') or p.name in ('STATUS','DONE','INPUTS_READY'):
  paths.append(str(rel))
subprocess.run(['tar','-czf','-','-C',str(r),'--']+paths,check=True)'''

def remote(script, **kwargs):
    return subprocess.run(SSH + ['python -c ' + shlex.quote(script)], check=True, timeout=600, **kwargs)

def main():
    ART.mkdir(exist_ok=True)
    backed_models = None
    last_backup = 0
    while True:
        try:
            sessions = subprocess.run(['colab','sessions'], capture_output=True, text=True, check=True, timeout=30).stdout
            if '[qtransformer-night]' not in sessions:
                raise RuntimeError('Runtime missing; refusing SSH that could create a replacement session')
            status = json.loads(remote(QUERY, capture_output=True, text=True).stdout)
            status['checked_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
            (HERE/'monitor-status.json').write_text(json.dumps(status,indent=2)+'\n')
            print(status['checked_utc'],status['status'],'models',len(status['models']),'evals',len(status['evaluations']),flush=True)
            if (status['models'] != backed_models or status['status'] in ('DONE','FAILED')
                    or (status['models'] and time.monotonic() - last_backup >= 1200)):
                temporary = ART/'backup-latest.tar.gz.tmp'
                with temporary.open('wb') as stream:
                    remote(BUNDLE, stdout=stream)
                with tarfile.open(temporary) as archive:
                    archive.extractall(ART/'backup', filter='data')
                temporary.replace(ART/'backup-latest.tar.gz')
                backed_models = status['models']
                last_backup = time.monotonic()
                print('Local backup verified and extracted',flush=True)
            if status['status'] in ('DONE','FAILED'):
                return
        except Exception as error:
            print(time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),str(error),flush=True)
        time.sleep(600)

if __name__ == '__main__':
    main()
