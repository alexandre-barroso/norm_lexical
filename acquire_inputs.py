"""Download only the four pinned public train/dev files; verify every byte hash."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();a.directory.mkdir(parents=True,exist_ok=True)
for item in json.loads((Path(__file__).parent/'inputs.json').read_text()).values():
 path=a.directory/item['file']
 data=path.read_bytes() if path.exists() else urllib.request.urlopen(item['url'],timeout=60).read()
 if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Pinned input changed: '+item['file'])
 if not path.exists():path.write_bytes(data)
 print('Verified',item['file'])
