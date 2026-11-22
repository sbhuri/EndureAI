"""Rebuild corpus from the included source PDF. Requires optional pypdf."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from pypdf import PdfReader
root=Path(__file__).resolve().parents[1];pdf=root/'fema_fnss.pdf'
existing=json.loads((root/'data/corpus.json').read_text())
if hashlib.sha256(pdf.read_bytes()).hexdigest()!=existing['pdf_sha256']:
 raise SystemExit('PDF differs from recorded source. Review provenance before replacing corpus.')
existing['pages']=[{'id':f'fnss-p{i+1}','page':i+1,'text':t} for i,p in enumerate(PdfReader(pdf).pages) if (t:=p.extract_text()).strip()]
existing['rebuilt_at']=datetime.now(timezone.utc).isoformat()
(root/'data/corpus.json').write_text(json.dumps(existing,indent=2))
print('Rebuilt',len(existing['pages']),'pages. Rerun tests and evaluation.')
