import json
from pathlib import Path

notebook_path = Path("/Users/Thomas/Desktop/Master Thesis/notebooks/16_stage1_baselines.ipynb")

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- 1. Remove add_view_availability_flags definition (id: 30285d07) ---
# We'll just empty the function body or remove the cell. Removing the cell might mess up numbering if anything else uses it.
# Actually, it's safer to just remove it.
nb["cells"] = [c for c in nb["cells"] if c.get("id") != "30285d07"]

# --- 2. Remove the call to add_view_availability_flags (id: 044cc23e) ---
for cell in nb["cells"]:
    if cell.get("id") == "044cc23e":
        source = cell["source"]
        # Remove the line calling it
        new_source = [line for line in source if "add_view_availability_flags" not in line]
        cell["source"] = new_source
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Successfully updated Notebook 16.")
