
from pathlib import Path
import ast

engine = Path("era_engine/core.py")
backup = Path("era_engine/core_before_rebound_fix.py.bak")
candidate = Path("era_engine/core_rebound_candidate.py")

# Safety: preserve the original working engine.
if not backup.exists():
    raise SystemExit("Backup missing. Stopping.")

source = engine.read_text()

if source != backup.read_text():
    raise SystemExit("Engine changed since backup. Stopping.")

start = "        ballhandler = rng.choices(offense,"
end = "    for pid, stat in stats.items():"

if source.count(start) != 1 or source.count(end) != 1:
    raise SystemExit("Unexpected engine structure.")

before, remainder = source.split(start, 1)
body, after = remainder.split(end, 1)
body = start + body

# Turnovers and shooting fouls end the possession.
old_exit = "            continue\n"

if body.count(old_exit) != 2:
    raise SystemExit("Unexpected turnover/foul logic.")

body = body.replace(
    old_exit,
    "            break\n"
)

# Made baskets end the possession.
old_miss = (
    "        else:\n"
    "            # A miss creates exactly one credited rebound"
)

new_miss = (
    "            break\n"
    "        else:\n"
    "            # A miss creates exactly one credited rebound"
)

if body.count(old_miss) != 1:
    raise SystemExit("Made-shot branch not found.")

body = body.replace(old_miss, new_miss)

# Continue play only after an offensive rebound.
old_rebound = (
    "            # Offensive rebound is credited but "
    "follow-up possession sequence is not modeled yet."
)

new_rebound = (
    "            if not offensive_reb:\n"
    "                break\n"
)

if body.count(old_rebound) != 1:
    raise SystemExit("Rebound logic not found.")

body = body.replace(old_rebound, new_rebound)

# Prototype: allow up to four shot sequences per possession.
# Additional offensive rebounds after the fourth attempt
# are recorded, but further play is not yet simulated.
body = (
    "        for second_chance in range(4):\n"
    + "".join(
        "    " + line if line.strip() else line
        for line in body.splitlines(keepends=True)
    )
)

updated = before + body + end + after

# Validate generated Python syntax.
ast.parse(updated)

# Write a separate candidate, not the working engine.
candidate.write_text(updated)

print("Candidate engine generated successfully.")
print("Original core.py remains unchanged.")
