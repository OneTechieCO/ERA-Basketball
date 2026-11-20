
from pathlib import Path
import ast

candidate = Path("era_engine/core_passing_candidate.py")
working = Path("era_engine/core.py")
backup = Path("era_engine/core_before_passing_fix.py.bak")

if working.read_bytes() != backup.read_bytes():
    raise SystemExit("Working engine differs from backup.")

source = candidate.read_text()

old = '''                passer = shooter
                shooter = recipient
                pass_count += 1

            # Only the final passer may receive an assist.
'''

new = '''                # Defense can disrupt the attempted pass.
                passer_defender = defense[offense.index(shooter)]
                recipient_defender = defense[offense.index(recipient)]

                pass_tov_chance = clamp(
                    .045
                    + (70 - shooter.passing) * .001
                    + (70 - shooter.iq) * .0005
                    + (passer_defender.steals - 70) * .0006
                    + (recipient_defender.defense - 70) * .0004,
                    .01,
                    .15,
                )

                if rng.random() < pass_tov_chance:
                    stats[shooter.player_id]['tov'] += 1

                    if rng.random() < .65:
                        stats[recipient_defender.player_id]['stl'] += 1

                    if track_events:
                        events.append({
                            'possession': poss_index,
                            'kind': 'pass_turnover',
                            'offense': shooter.player_id,
                            'defender': recipient_defender.player_id,
                        })

                    pass_failed = True
                    break

                passer = shooter
                shooter = recipient
                pass_count += 1

            # A failed pass ends the possession.
            if pass_failed:
                break

            # Only the final passer may receive an assist.
'''

# Initialize the flag before the passing loop.
old_loop = '''            pass_count = 0
            max_passes = 6

            for _ in range(max_passes):
'''

new_loop = '''            pass_count = 0
            max_passes = 6
            pass_failed = False

            for _ in range(max_passes):
'''

if source.count(old) != 1:
    raise SystemExit("Expected passing block not found.")

if source.count(old_loop) != 1:
    raise SystemExit("Expected passing loop not found.")

updated = source.replace(old_loop, new_loop)
updated = updated.replace(old, new)

ast.parse(updated)

# Modify only the experimental candidate.
candidate.write_text(updated)

print("Passing turnover logic added successfully.")
print("Working core.py remains unchanged.")
