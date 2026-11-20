"""FICTIONAL TEST FIXTURES: Not official NBA-derived profiles or calibrated eras."""
from .core import Player, Era, Team
from .storage import write_player, write_era, write_team

ERAS = [
    Era('1985-86', 102.0, .035, .84, False, 1.0, 24, rule_note='illustrative 1980s proxy'),
    Era('1995-96', 91.5, .15, .78, False, .90, 24, rule_note='illustrative; shortened historical three-point line, geometry proxy'),
    Era('2005-06', 90.5, .23, .48, True, 1.0, 24, rule_note='illustrative 2000s proxy'),
    Era('2015-16', 96.0, .33, .32, True, 1.0, 24, rule_note='illustrative 2010s proxy'),
    Era('2023-24', 99.2, .39, .28, True, 1.0, 24, rule_note='illustrative 2020s proxy'),
]

# All names, attributes, and source-free research IDs below are fictional.
NAMES = ['Sterling Cross', 'Marcus Jett', 'Samson Oakes', 'Adrian Vance', 'Leon Bridges',
         'Kieran Blake', 'Harold Crest', 'Victor Stonewell', 'Damon Rush', 'Darius Keene']


def demo_players() -> list[Player]:
    out = []
    for i, name in enumerate(NAMES):
        shooter = i in (0, 5)
        big = i in (2, 7)
        defender = i in (6, 7, 9)
        skills = {'rim': 84 if big else 77 + i % 7, 'paint': 82 if big else 72,
                  'mid': 86 if i in (1,5) else 73, 'three': 98 if shooter else (44 if big else 74)}
        tendencies = {'rim': .42 if big else .25, 'paint': .35 if big else .17,
                      'mid': .18 if big else .31, 'three': .05 if big else (.53 if shooter else .27)}
        out.append(Player(f'ERA_{i+1:04d}', name, '2010s' if shooter else '1990s',
                          94 if shooter else 78, 95 if defender else (82 if big else 73),
                          86 if shooter else 72, 96 if big else 70, 92 if shooter else 78,
                          84, .33 if shooter else .19, skills, tendencies,
                          steals=86 if defender else 70, blocks=93 if big else 52))
    return out


def initialize_demo(con) -> None:
    for p in demo_players():
        write_player(con, p)
    for era in ERAS:
        write_era(con, era, verified=False)
    write_team(con, Team('Bay City Legends', tuple(f'ERA_{i:04d}' for i in range(1,6)), 'perimeter'))
    write_team(con, Team('Chicago Stampede', tuple(f'ERA_{i:04d}' for i in range(6,11)), 'balanced'))
