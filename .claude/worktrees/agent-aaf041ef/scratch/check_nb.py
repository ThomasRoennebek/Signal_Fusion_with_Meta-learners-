import json
with open('Notebooks/13_prepare_gold_labels.ipynb') as f:
    nb = json.load(f)
    
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and cell['source']:
        if any('candidate sampling strategy' in line.lower() for line in cell['source']):
            print('\n--- STOP REACHED (12. candidate sampling strategy) ---')
            break
    if cell['cell_type'] == 'code':
        outs = cell.get('outputs', [])
        for o in outs:
            if o.get('output_type') == 'stream':
                print("".join(o.get('text', '')))
            elif o.get('output_type') == 'execute_result':
                print("".join(o.get('data', {}).get('text/plain', [])))
