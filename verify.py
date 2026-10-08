"""Run synthetic checks and verify stored or newly reproduced numeric results."""
import argparse,collections,hashlib,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path);a=p.parse_args()
 r=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(ROOT),'-p','test_*.py'],cwd=ROOT)
 if r.returncode:raise SystemExit(r.returncode)
 expected=json.loads((ROOT/'expected_results.json').read_text())
 if a.results:
  result=json.loads((a.results/'results.json').read_text())
  assert result['paired_summaries']==expected['paired_summaries'],'Paired results differ'
  drop={'training_seconds','model_file','model_sha256','model_bytes'}
  models=[{k:v for k,v in m.items() if k not in drop} for m in result['models']]
  assert models==expected['models'],'Raw accuracy, model settings or counts differ'
  rows=[json.loads(x) for x in (a.results/'events.jsonl').read_text().splitlines()]
  assert len(rows)==2352
  assert hashlib.sha256((a.results/'events.jsonl').read_bytes()).hexdigest()==expected['original_events_sha256'],'Predictions differ from original 2352 pairs'
  for x in rows:
   g=x['raw']!=x['gold'] and x['edited']==x['gold'];h=x['raw']==x['gold'] and x['edited']!=x['gold']
   assert x['outcome']==('gain' if g else 'harm' if h else 'both_correct' if x['raw']==x['gold'] else 'both_wrong')
   cg=sum(c['raw']!=c['gold'] and c['edited']==c['gold'] for c in x['context_changes'])
   ch=sum(c['raw']==c['gold'] and c['edited']!=c['gold'] for c in x['context_changes'])
   assert cg-ch==x['context']['edited_correct']-x['context']['raw_correct']
  print('PASS: 2352 paired predictions, all 16 summary cells and eight model/regime records exactly match the original run.')
 else:
  print('PASS: synthetic checks. Corpus reproduction NOT RUN; supply --results after the documented run.')
if __name__=='__main__':main()
