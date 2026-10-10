"""Print a relocated historical path; immutable experiment JSON is never rewritten."""
import json
from pathlib import Path
import sys

mapping=json.loads(Path(__file__).with_name('legacy_path_map_20261007.json').read_text())
for value in sys.argv[1:]:
    for old,new in sorted(mapping.items(),key=lambda item:-len(item[0])):
        if value == old or value.startswith(old+'/'):
            value=new+value[len(old):]
            break
    print(value)
