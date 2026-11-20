"""SQLite database for fictional player profiles, historical parameters and simulations."""
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from .core import Player, Era, Team

SCHEMA = '''
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS players (
 player_id TEXT PRIMARY KEY, fictional_name TEXT NOT NULL UNIQUE,
 origin_era TEXT NOT NULL, profile_json TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS nba_research_links (
 player_id TEXT PRIMARY KEY REFERENCES players(player_id),
 nba_player_id INTEGER NOT NULL UNIQUE, research_note TEXT DEFAULT 'research only');
CREATE TABLE IF NOT EXISTS eras (
 era_key TEXT PRIMARY KEY, settings_json TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS teams (
 team_name TEXT PRIMARY KEY, roster_json TEXT NOT NULL, coach_style TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS simulations (
 simulation_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
 model_version TEXT NOT NULL, result_json TEXT NOT NULL);
'''


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute('PRAGMA foreign_keys=ON')
    con.executescript(SCHEMA)
    return con


def write_player(con: sqlite3.Connection, p: Player, nba_id: int | None = None) -> None:
    con.execute('INSERT OR REPLACE INTO players(player_id,fictional_name,origin_era,profile_json,version) VALUES (?,?,?,?,COALESCE((SELECT version FROM players WHERE player_id=?),1))',
                (p.player_id, p.name, p.origin_era, json.dumps(asdict(p)), p.player_id))
    if nba_id is not None:
        con.execute('INSERT OR REPLACE INTO nba_research_links(player_id,nba_player_id) VALUES (?,?)', (p.player_id, nba_id))
    con.commit()


def write_era(con: sqlite3.Connection, era: Era, verified: bool = False) -> None:
    con.execute('INSERT OR REPLACE INTO eras(era_key,settings_json,verified) VALUES (?,?,?)', (era.key, json.dumps(asdict(era)), int(verified)))
    con.commit()


def write_team(con: sqlite3.Connection, t: Team) -> None:
    con.execute('INSERT OR REPLACE INTO teams(team_name,roster_json,coach_style) VALUES (?,?,?)', (t.name, json.dumps(t.player_ids), t.coach_style))
    con.commit()


def load(con: sqlite3.Connection) -> tuple[dict[str, Player], dict[str, Era], dict[str, Team]]:
    players = {p.player_id:p for (s,) in con.execute('SELECT profile_json FROM players') if (p := Player(**json.loads(s)))}
    eras = {e.key:e for (s,) in con.execute('SELECT settings_json FROM eras') if (e := Era(**json.loads(s)))}
    teams = {t.name:t for name, roster, style in con.execute('SELECT team_name, roster_json, coach_style FROM teams') if (t := Team(name, tuple(json.loads(roster)), style))}
    return players, eras, teams


def save_simulation(con: sqlite3.Connection, result: dict) -> int:
    cursor = con.execute('INSERT INTO simulations(model_version,result_json) VALUES (?,?)',
                         (result['model_version'], json.dumps(result)))
    con.commit()
    return int(cursor.lastrowid)


def create_research_schema(con: sqlite3.Connection) -> None:
    con.execute('''CREATE TABLE IF NOT EXISTS nba_season_research (
      nba_player_id INTEGER NOT NULL, season_id TEXT NOT NULL,
      source TEXT NOT NULL, fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
      stats_json TEXT NOT NULL,
      PRIMARY KEY(nba_player_id, season_id, source))''')
    con.executescript('''
    CREATE TABLE IF NOT EXISTS nba_source_response (
      source_name TEXT NOT NULL,
      endpoint TEXT NOT NULL,
      dataset_name TEXT NOT NULL,
      season_id TEXT NOT NULL,
      entity_kind TEXT NOT NULL,
      entity_id INTEGER NOT NULL,
      entity_name TEXT NOT NULL,
      per_mode TEXT NOT NULL,
      retrieved_on TEXT NOT NULL,
      field_names_json TEXT NOT NULL,
      raw_json TEXT NOT NULL,
      PRIMARY KEY (source_name, endpoint, dataset_name, season_id, entity_kind, entity_id, per_mode)
    );
    CREATE TABLE IF NOT EXISTS nba_player_season_research (
      nba_player_id INTEGER NOT NULL,
      player_name TEXT NOT NULL,
      season_id TEXT NOT NULL,
      team_id INTEGER NOT NULL,
      team_abbreviation TEXT,
      source_name TEXT NOT NULL,
      endpoint TEXT NOT NULL,
      dataset_name TEXT NOT NULL,
      counting_measure TEXT,
      games REAL,
      minutes REAL,
      points REAL,
      fgm REAL,
      fga REAL,
      fg3m REAL,
      fg3a REAL,
      ftm REAL,
      fta REAL,
      assists REAL,
      offensive_rebounds REAL,
      defensive_rebounds REAL,
      steals REAL,
      blocks REAL,
      turnovers REAL,
      usage_rate REAL,
      usage_rate_measure TEXT,
      PRIMARY KEY (nba_player_id, season_id, team_id, source_name, endpoint, dataset_name)
    );
    CREATE TABLE IF NOT EXISTS nba_team_season_research (
      team_id INTEGER NOT NULL,
      team_name TEXT NOT NULL,
      season_id TEXT NOT NULL,
      source_name TEXT NOT NULL,
      endpoint TEXT NOT NULL,
      dataset_name TEXT NOT NULL,
      counting_measure TEXT,
      games REAL,
      minutes REAL,
      points REAL,
      fgm REAL,
      fga REAL,
      fg3m REAL,
      fg3a REAL,
      ftm REAL,
      fta REAL,
      assists REAL,
      offensive_rebounds REAL,
      defensive_rebounds REAL,
      steals REAL,
      blocks REAL,
      turnovers REAL,
      pace REAL,
      pace_measure TEXT,
      possessions REAL,
      possessions_measure TEXT,
      PRIMARY KEY (team_id, season_id, source_name, endpoint, dataset_name)
    );
    ''')
    con.commit()
