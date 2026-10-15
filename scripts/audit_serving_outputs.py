"""Revalidate saved complete model outputs without regenerating or altering them."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.serving.output_validation import OutputValidationError, validate_output

parser = argparse.ArgumentParser()
parser.add_argument('--http-results', required=True)
parser.add_argument('--manifest', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
records = {int(record['clip_id']): record for record in (
    json.loads(line) for line in Path(args.manifest).read_text(encoding='utf-8').splitlines()
    if line.strip())}
checks = json.loads(Path(args.http_results).read_text(encoding='utf-8'))['checks']
audit = {'scope': 'saved complete outputs revalidated, no new model inference', 'outputs': []}
for check in checks:
    body = check['body']
    if body.get('result') is None or 'text' not in body:
        continue
    record = records[body['clip_id']]
    item = {'source_check': check['name'], 'clip_id': body['clip_id']}
    try:
        validate_output(body['text'], record, len(record['selected_frames']))
        item.update(valid=True, validation_errors=[])
    except OutputValidationError as error:
        item.update(valid=False, validation_errors=error.issues)
    audit['outputs'].append(item)
audit['total'] = len(audit['outputs'])
audit['passed'] = sum(item['valid'] for item in audit['outputs'])
with Path(args.output).open('x', encoding='utf-8') as handle:
    json.dump(audit, handle, indent=2)
print(json.dumps(audit, indent=2), flush=True)
