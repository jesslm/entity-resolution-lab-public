#!/usr/bin/env python3
"""
Git filter script to strip outputs from Jupyter notebooks.
This can be used as a git clean filter to automatically remove outputs.
"""
import json
import sys

def strip_notebook_outputs(content):
    """Remove outputs, execution_count, and execution metadata from notebook"""
    nb = json.loads(content)
    
    for cell in nb['cells']:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = None
        # Remove execution-related metadata
        if 'metadata' in cell:
            for key in list(cell['metadata'].keys()):
                if 'execution' in key.lower() or 'collapsed' in key.lower():
                    del cell['metadata'][key]
    
    # Remove notebook-level execution metadata
    if 'metadata' in nb:
        if 'execution' in nb['metadata']:
            del nb['metadata']['execution']
    
    return json.dumps(nb, indent=1, ensure_ascii=False)

if __name__ == '__main__':
    input_content = sys.stdin.read()
    output_content = strip_notebook_outputs(input_content)
    sys.stdout.write(output_content)
