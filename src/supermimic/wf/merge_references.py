"""Merge immutable physical reference libraries without changing clip splits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import os


def merge(sources,output,hardlink=False):
    output=Path(output)
    if (output/'manifest.json').exists():raise FileExistsError('select a new library directory')
    output.mkdir(parents=True,exist_ok=True);result=None;seen=set()
    for source in map(Path,sources):
        raw=(source/'manifest.json').read_bytes();m=json.loads(raw)
        if not m.get('complete') or m.get('format')!='wf_physical_references_v1':raise ValueError('incomplete reference source')
        if result is None:
            result=copy.deepcopy(m);result.update(clips=[],rejected=[],sources=[],level='mixed')
        if result['provenance']['physics_hash']!=m['provenance']['physics_hash'] or result['dt']!=m['dt']:
            raise ValueError('source physics/control periods differ')
        result['sources'].append({'directory':str(source),'sha256':hashlib.sha256(raw).hexdigest(),'level':m['level']})
        for r in m['clips']:
            level=r.get('level',m['level'])
            if not isinstance(level,int):raise ValueError('each merged clip needs its numeric curriculum level')
            # Independent generators may reuse an RNG seed for different
            # physical executions. Reject the repeated episode, not unrelated
            # state/action bytes that happen to share a seed.
            identity=(r['seed'],level,r['sha256'])
            if identity in seen:raise ValueError('duplicate source episode')
            seen.add(identity);i=len(result['clips']);record=copy.deepcopy(r);name=f'clip_{i:04d}'
            for field,suffix in (('file','.npz'),('keyframes','_keyframes.json')):
                dest=name+suffix
                if hardlink:
                    try:os.link(source/r[field],output/dest)
                    except OSError:shutil.copy2(source/r[field],output/dest)
                else:shutil.copy2(source/r[field],output/dest)
                record[field]=dest
            record.update(source_directory=str(source),level=level)
            result['clips'].append(record)
        result['rejected'].extend(m['rejected'])
    if result is None:raise ValueError('at least one library required')
    result['requested_clips']=len(result['clips']);result['complete']=True
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sources',nargs='+',required=True);p.add_argument('--output',required=True)
    p.add_argument('--hardlink',action='store_true',help='share immutable physical clip bytes within one filesystem')
    a=p.parse_args();r=merge(a.sources,a.output,a.hardlink)
    print(json.dumps({'clips':len(r['clips']),'train':sum(x['split']=='train' for x in r['clips']),'output':a.output}))


if __name__=='__main__':main()
