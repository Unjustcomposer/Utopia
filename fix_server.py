with open("server.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    line_num = i + 1
    # Skip dashboard endpoint
    if 83 <= line_num <= 86:
        continue
    # Skip experiment, search, sensitivity, calibrate, grad-search
    if 144 <= line_num <= 546:
        continue
    # Skip broken imports
    if 34 <= line_num <= 37:
        continue
        
    if line_num == 12:
        new_lines.append("import jax\nimport ray\n")
        continue
        
    if line_num == 21:
        new_lines.append("from typing import Dict, Any, Optional, Tuple, List\n")
        continue
        
    new_lines.append(line)

with open("server.py", "w") as f:
    f.writelines(new_lines)
