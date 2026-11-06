"""Fetch and verify exact official sources. No third-party code is republished."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

SOURCES = {
 'wan': ('https://github.com/Wan-Video/Wan2.1.git','9737cba9c1c3c4d04b33fcad41c111989865d315'),
 'teacache': ('https://github.com/ali-vilab/TeaCache.git','7c10efc4702c6b619f47805f7abe4a7a08085aa0'),
 'magcache': ('https://github.com/Zehong-Ma/MagCache.git','df81cb181776c2c61477c08e1d21f87fda1cd938'),
 'seacache': ('https://github.com/jiwoogit/SeaCache.git','8dcf49097fcd37e39774fe7409cb3b9e0fdb4fe2'),
 'sencache': ('https://github.com/vita-epfl/SenCache.git','3fb96a41511ede1e38f3555c863b1cb7cddfe567'),
}
FORWARDS = {
 'teacache': ('TeaCache4Wan2.1/teacache_generate.py','teacache_forward'),
 'magcache': ('MagCache4Wan2.1/magcache_generate.py','magcache_forward'),
 'seacache': ('Wan2.1/seacache_generate.py','seacache_forward'),
 'sencache': ('Wan2.1/sencache.py','sencache_forward'),
}
def git(path, *args):
 return subprocess.check_output(['git','-C',str(path),*args],text=True).strip()
def verify(root: Path, names=None):
 result={}
 for name in (names or SOURCES):
  path=root/name; expected=SOURCES[name][1]
  got=git(path,'rev-parse','HEAD')
  if got!=expected: raise ValueError(f'{name}: revision {got} != pinned {expected}')
  if git(path,'status','--porcelain','--untracked-files=no'):
   raise ValueError(f'{name}: tracked files modified; restore or use a clean pinned checkout')
  result[name]={'url':SOURCES[name][0],'commit':got}
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--directory',type=Path,default=Path('vendor'))
 p.add_argument('--verify-only',action='store_true');a=p.parse_args()
 a.directory.mkdir(parents=True,exist_ok=True)
 if not a.verify_only:
  for name,(url,rev) in SOURCES.items():
   dst=a.directory/name
   if not dst.exists():
    subprocess.run(['git','clone','--no-checkout',url,str(dst)],check=True)
    subprocess.run(['git','-C',str(dst),'checkout','--detach',rev],check=True)
 print(json.dumps(verify(a.directory),indent=2))
if __name__=='__main__': main()
