"""Read-only historical rates and calibration notes.

Derived rates are calculated when a stored season total can be divided by a
stored denominator. They are not written back over NBA source columns, and
they are not player ratings. This module does not run the simulator.
"""
import json
import sqlite3
from decimal import Decimal, InvalidOperation
from pathlib import Path

SOURCE_PLAYER_ENDPOINT = "playercareerstats"
SOURCE_PLAYER_DATASET = "SeasonTotalsRegularSeason"
SOURCE_TEAM_COUNTING_ENDPOINT = "teamyearbyyearstats"
SOURCE_TEAM_ADVANCED_ENDPOINT = "leaguedashteamstats"

# Carried forward from the audited NBA.com season totals. These are reported,
# not used to edit either side of the gap.
KNOWN_TURNOVER_GAPS = {
    (1610612741, "1995-96"): 41,
    (1610612744, "2015-16"): 26,
}

COMPARISON_SEASONS = (
    {"season_id": "1995-96", "team_id": 1610612741, "team_name": "Chicago Bulls", "team_abbreviation": "CHI"},
    {"season_id": "2015-16", "team_id": 1610612744, "team_name": "Golden State Warriors", "team_abbreviation": "GSW"},
)

COUNTING_FIELDS = (
    ("points", "points"),
    ("field_goals_made", "fgm"),
    ("field_goals_attempted", "fga"),
    ("three_pointers_made", "fg3m"),
    ("three_pointers_attempted", "fg3a"),
    ("free_throws_made", "ftm"),
    ("free_throws_attempted", "fta"),
    ("assists", "assists"),
    ("offensive_rebounds", "offensive_rebounds"),
    ("defensive_rebounds", "defensive_rebounds"),
    ("steals", "steals"),
    ("blocks", "blocks"),
    ("turnovers", "turnovers"),
)

FORMULAS = {
    "per_game": "counting_stat / games, when games is present and not zero",
    "minutes_per_game": "minutes / games, when both are present and games is not zero",
    "field_goal_percentage": "field_goals_made / field_goals_attempted, when attempts are present and not zero",
    "three_point_percentage": "three_pointers_made / three_pointers_attempted, when attempts are present and not zero",
    "free_throw_percentage": "free_throws_made / free_throws_attempted, when attempts are present and not zero",
    "three_point_attempt_rate": "three_pointers_attempted / field_goals_attempted, when field-goal attempts are present and not zero",
    "effective_field_goal_percentage": "(field_goals_made + 0.5 * three_pointers_made) / field_goals_attempted, when all three are present and attempts are not zero",
    "rebounds": "offensive_rebounds + defensive_rebounds, only when both are present",
    "per_36_minutes": "counting_stat / minutes * 36, when minutes are present and not zero",
    "team_per_100_possessions": "counting_stat / possessions * 100, only when the counting stat and possessions are on the same source row and possessions_measure is season_total",
    "usage_fraction": "Basketball-Reference displayed USG% / 100. This is an on-floor share, not a per-game rate, and it is not written into Player.usage or the NBA usage_rate column.",
    "player_per_100_on_floor_possessions": "not calculated; the stored sources do not include possession exposure while the player was on the floor",
}

SIMULATOR_COMPARISON = {
    "simulation_was_run": False,
    "comparable_with_current_per_game_outputs": [
        "points per game",
        "assists, rebounds, steals, blocks, and turnovers per game",
        "field-goal, three-point, and free-throw percentages",
        "field-goal attempts and three-point attempts per game",
        "three-point attempt rate, which can be derived from simulated attempts",
        "team points per game from simulated scores",
    ],
    "comparable_limit": "Those shapes exist on simulate_series averages. The fictional demo rosters have five players and no substitutes, so each of those players still plays the whole game. Historical rates remain benchmarks rather than a Jordan-versus-Curry result.",
    "requires_minutes_or_rotation_model": [
        "per-36-minute counting rates for a player who did not play the whole game",
        "a comparison that needs a historical starting lineup",
    ],
    "minutes_reason": "simulate_game records the seconds each active player spends on the court. A five-player roster plays all 48 minutes. A larger roster uses the regulation scheduler, and a player who is not selected records zero minutes. Dividing a full-game line by 48 and multiplying by 36 only rescales that line.",
    "unavailable": [
        "NBA.com usage rate, which is NULL in the stored player rows",
        "Chicago 1995-96 NBA.com pace and possessions, which are NULL",
        "team counting stats per 100 possessions, because possessions and the counting totals are not on the same source row and are not combined",
        "player per-100-on-floor-possession rates, because possession exposure is absent",
        "shot-zone makes and attempts",
    ],
    "pace_methodologies_not_interchangeable": [
        "NBA.com LeagueDashTeamStats PACE stays on that advanced row. Chicago's value is NULL. Golden State's stored value is a rate from that endpoint.",
        "Basketball-Reference pace is a separate pace factor: an estimate of possessions per 48 minutes, using 48 * ((Tm Poss + Opp Poss) / (2 * (Tm MP / 5))).",
        "The seeded era paces, 91.5 for 1995-96 and 96.0 for 2015-16, are illustrative engine inputs and are not historical imports.",
    ],
}


def _number(value):
    if value is None or value == "" or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def quotient(numerator, denominator):
    """Divide only when both numbers are present and the denominator is not zero."""
    numerator = _number(numerator)
    denominator = _number(denominator)
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def scale(numerator, denominator, factor: float):
    """Return numerator / denominator * factor, or None when the quotient is undefined."""
    base = quotient(numerator, denominator)
    if base is None:
        return None
    return base * factor


def rebound_total(record: dict):
    offensive = _number(record.get("offensive_rebounds"))
    defensive = _number(record.get("defensive_rebounds"))
    if offensive is None or defensive is None:
        return None
    return offensive + defensive


def effective_field_goal_percentage(record: dict):
    made = _number(record.get("fgm"))
    threes = _number(record.get("fg3m"))
    attempts = _number(record.get("fga"))
    if made is None or threes is None or attempts is None or attempts == 0:
        return None
    return (made + 0.5 * threes) / attempts


def _per_unit(record: dict, denominator, factor: float | None = None, include_minutes: bool = False) -> dict:
    rates = {}
    for label, column in COUNTING_FIELDS:
        value = record.get(column)
        rates[label] = scale(value, denominator, factor) if factor is not None else quotient(value, denominator)
    if include_minutes:
        rates["minutes"] = quotient(record.get("minutes"), denominator)
    total_rebounds = rebound_total(record)
    rates["rebounds"] = scale(total_rebounds, denominator, factor) if factor is not None else quotient(total_rebounds, denominator)
    return rates


def percentage_block(record: dict) -> dict:
    return {
        "field_goal": quotient(record.get("fgm"), record.get("fga")),
        "three_point": quotient(record.get("fg3m"), record.get("fg3a")),
        "free_throw": quotient(record.get("ftm"), record.get("fta")),
        "three_point_attempt_rate": quotient(record.get("fg3a"), record.get("fga")),
        "effective_field_goal": effective_field_goal_percentage(record),
    }


def per_36_minutes(record: dict) -> dict | None:
    if _number(record.get("minutes")) in (None, 0):
        return None
    return _per_unit(record, record.get("minutes"), 36)


def per_100_possessions(record: dict) -> dict | None:
    """Use possessions only from this same row. Never borrow them from another endpoint."""
    if record.get("possessions_measure") != "season_total":
        return None
    if _number(record.get("possessions")) in (None, 0):
        return None
    return _per_unit(record, record.get("possessions"), 100)


def _query(con, sql: str, params=()) -> list[dict]:
    cursor = con.execute(sql, params)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _player_rates(record: dict, external_usage: dict | None) -> dict:
    return {
        "nba_player_id": record["nba_player_id"],
        "player_name": record["player_name"],
        "season_id": record["season_id"],
        "team_id": record["team_id"],
        "team_abbreviation": record.get("team_abbreviation"),
        "source_name": record["source_name"],
        "endpoint": record["endpoint"],
        "dataset_name": record["dataset_name"],
        "counting_measure": record.get("counting_measure"),
        "per_game": _per_unit(record, record.get("games"), include_minutes=True),
        "percentages": percentage_block(record),
        "per_36_minutes": per_36_minutes(record),
        "per_100_on_floor_possessions": None,
        "per_100_on_floor_possessions_reason": FORMULAS["player_per_100_on_floor_possessions"],
        "nba_usage_rate": record.get("usage_rate"),
        "external_on_floor_usage_fraction": None if external_usage is None else external_usage.get("normalized_fraction"),
        "external_on_floor_usage_source": None if external_usage is None else _external_provenance(external_usage),
    }


def _team_source_rates(record: dict) -> dict:
    possessions_reason = None
    per_100 = per_100_possessions(record)
    if per_100 is None:
        possessions_reason = (
            "Per-100-possession rates are omitted unless this same source row has a season-total possessions value."
        )
    return {
        "source_name": record["source_name"],
        "endpoint": record["endpoint"],
        "dataset_name": record["dataset_name"],
        "counting_measure": record.get("counting_measure"),
        "pace": record.get("pace"),
        "pace_measure": record.get("pace_measure"),
        "possessions": record.get("possessions"),
        "possessions_measure": record.get("possessions_measure"),
        "per_game": _per_unit(record, record.get("games"), include_minutes=True),
        "percentages": percentage_block(record),
        "per_100_possessions": per_100,
        "per_100_possessions_reason": possessions_reason,
    }


def _turnover_gap(team_record: dict | None, player_records: list[dict], season: dict) -> dict:
    team_turnovers = None if team_record is None else _number(team_record.get("turnovers"))
    player_turnovers = None
    if player_records and all(_number(record.get("turnovers")) is not None for record in player_records):
        player_turnovers = sum(_number(record["turnovers"]) for record in player_records)
    computed = None if team_turnovers is None or player_turnovers is None else team_turnovers - player_turnovers
    known = KNOWN_TURNOVER_GAPS.get((season["team_id"], season["season_id"]))
    games = None if team_record is None else _number(team_record.get("games"))
    return {
        "team_turnovers": team_turnovers,
        "player_turnovers": player_turnovers,
        "computed_team_minus_players": computed,
        "carried_forward_gap": known,
        "matches_carried_forward_gap": computed == known,
        "unreconciled_gap_per_team_game": quotient(computed, games),
        "reconciled": False,
    }


def ensure_external_schema(con) -> None:
    con.execute('''CREATE TABLE IF NOT EXISTS research_external_observation (
      source_name TEXT NOT NULL,
      url TEXT NOT NULL,
      retrieved_on TEXT NOT NULL,
      season_id TEXT NOT NULL,
      entity_kind TEXT NOT NULL,
      entity_id INTEGER NOT NULL,
      entity_name TEXT NOT NULL,
      field_name TEXT NOT NULL,
      field_label TEXT NOT NULL,
      displayed_value REAL,
      normalized_fraction REAL,
      measurement_definition TEXT NOT NULL,
      methodology TEXT NOT NULL,
      import_status TEXT NOT NULL,
      status_reason TEXT,
      raw_json TEXT NOT NULL,
      PRIMARY KEY (source_name, url, season_id, entity_id, field_name)
    )''')


def _percentage_fraction(displayed_text: str) -> float:
    return float(Decimal(displayed_text) / Decimal(100))


def classify_external_observation(payload: dict) -> dict:
    """Accept a labeled source cell, or keep the payload as an unimported lead.

    The displayed cell text is the imported value. A sort key on the same cell
    is preserved in the raw payload and is not substituted for that text.
    """
    cells = payload.get("labeled_cells") if isinstance(payload.get("labeled_cells"), dict) else {}
    field_name = str(payload.get("field_name") or "")
    displayed_text = cells.get(field_name)
    reason = None
    displayed_value = None
    normalized_fraction = None
    status = "unimported"
    if displayed_text in (None, ""):
        reason = "labeled source field is absent"
    elif payload.get("page_season_id") != payload.get("season_id"):
        reason = "page season does not match the observation season"
    elif field_name == "pace":
        if payload.get("table_id") != "team_misc" or cells.get("player") != "Team":
            reason = "pace cell is not the team row of team_misc"
        else:
            status = "verified"
    elif field_name == "usg_pct":
        if payload.get("table_id") != "advanced":
            reason = "usage cell is not on the regular-season advanced table"
        elif cells.get("year_id") != payload.get("season_id"):
            reason = "usage row season does not match the observation season"
        elif not cells.get("team_name_abbr"):
            reason = "usage row has no team abbreviation"
        else:
            status = "verified"
    else:
        reason = "field is not one of the verified external fields"
    if status == "verified":
        try:
            displayed_value = float(Decimal(str(displayed_text)))
        except (InvalidOperation, ValueError):
            status = "unimported"
            reason = "labeled source field is not a number"
            displayed_value = None
        else:
            if field_name == "usg_pct":
                normalized_fraction = _percentage_fraction(str(displayed_text))
    definition = payload.get("column_definition") or payload.get("glossary_definition") or ""
    return {
        "source_name": payload.get("source_name") or "",
        "url": payload.get("url") or "",
        "retrieved_on": payload.get("retrieved_on") or "",
        "season_id": payload.get("season_id") or "",
        "entity_kind": payload.get("entity_kind") or "",
        "entity_id": payload.get("entity_id"),
        "entity_name": payload.get("entity_name") or "",
        "field_name": field_name,
        "field_label": payload.get("field_label") or field_name,
        "displayed_value": displayed_value,
        "normalized_fraction": normalized_fraction,
        "measurement_definition": definition,
        "methodology": payload.get("methodology") or "",
        "import_status": status,
        "status_reason": reason,
        "raw_json": json.dumps(payload, allow_nan=False),
    }


def store_external_observation(con, observation: dict) -> None:
    ensure_external_schema(con)
    con.execute(
        '''INSERT OR REPLACE INTO research_external_observation(
             source_name, url, retrieved_on, season_id, entity_kind, entity_id, entity_name,
             field_name, field_label, displayed_value, normalized_fraction, measurement_definition,
             methodology, import_status, status_reason, raw_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (observation["source_name"], observation["url"], observation["retrieved_on"], observation["season_id"],
         observation["entity_kind"], observation["entity_id"], observation["entity_name"], observation["field_name"],
         observation["field_label"], observation["displayed_value"], observation["normalized_fraction"],
         observation["measurement_definition"], observation["methodology"], observation["import_status"],
         observation["status_reason"], observation["raw_json"]),
    )


def import_external_directory(con, directory: str) -> dict:
    """Load saved labeled excerpts. A missing field stays an unimported lead."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"External research directory is missing: {root}")
    counts = {"verified": 0, "unimported": 0}
    for path in sorted(root.glob("*.json")):
        observation = classify_external_observation(json.loads(path.read_text()))
        store_external_observation(con, observation)
        counts[observation["import_status"]] += 1
    con.commit()
    return counts


def _external_provenance(row: dict) -> dict:
    raw = json.loads(row["raw_json"])
    return {
        "source_name": row["source_name"],
        "url": row["url"],
        "displayed_value": row["displayed_value"],
        "normalized_fraction": row["normalized_fraction"],
        "measurement_definition": row["measurement_definition"],
        "glossary_url": raw.get("glossary_url"),
        "glossary_definition": raw.get("glossary_definition"),
        "methodology": row["methodology"],
    }


def _external_rows(con) -> list[dict]:
    if con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_external_observation'"
    ).fetchone() is None:
        return []
    return _query(con, "SELECT * FROM research_external_observation ORDER BY season_id, entity_name, field_name")


def build_comparison_report(con) -> dict:
    """Assemble Chicago 1995-96 and Golden State 2015-16 benchmarks without writing source rows."""
    external = _external_rows(con)
    verified = [row for row in external if row["import_status"] == "verified"]
    usage_by_player = {
        (row["entity_id"], row["season_id"]): row
        for row in verified if row["field_name"] == "usg_pct" and row["entity_kind"] == "player"
    }
    pace_by_team = {
        (row["entity_id"], row["season_id"]): row
        for row in verified if row["field_name"] == "pace" and row["entity_kind"] == "team"
    }
    seasons = []
    for season in COMPARISON_SEASONS:
        players = _query(
            con,
            '''SELECT * FROM nba_player_season_research
               WHERE season_id=? AND team_id=? AND endpoint=? AND dataset_name=?
                 AND UPPER(COALESCE(team_abbreviation, '')) != 'TOT'
               ORDER BY player_name''',
            (season["season_id"], season["team_id"], SOURCE_PLAYER_ENDPOINT, SOURCE_PLAYER_DATASET),
        )
        counting = _query(
            con,
            '''SELECT * FROM nba_team_season_research
               WHERE season_id=? AND team_id=? AND endpoint=?''',
            (season["season_id"], season["team_id"], SOURCE_TEAM_COUNTING_ENDPOINT),
        )
        advanced_rows = _query(
            con,
            '''SELECT * FROM nba_team_season_research
               WHERE season_id=? AND team_id=? AND endpoint=?''',
            (season["season_id"], season["team_id"], SOURCE_TEAM_ADVANCED_ENDPOINT),
        )
        counting_row = counting[0] if counting else None
        advanced_row = advanced_rows[0] if advanced_rows else None
        external_pace = pace_by_team.get((season["team_id"], season["season_id"]))
        seasons.append({
            "season_id": season["season_id"],
            "team_id": season["team_id"],
            "team_name": season["team_name"],
            "player_count": len(players),
            "players": [
                _player_rates(record, usage_by_player.get((record["nba_player_id"], record["season_id"])))
                for record in players
            ],
            "team_counting_source": None if counting_row is None else _team_source_rates(counting_row),
            "team_advanced_source": None if advanced_row is None else _team_source_rates(advanced_row),
            "external_pace": None if external_pace is None else {
                **_external_provenance(external_pace),
                "merged_into_nba_pace": False,
            },
            "turnover_gap": _turnover_gap(counting_row, players, season),
            "other_endpoint_player_rows": _query(
                con,
                '''SELECT endpoint, COUNT(*) AS rows
                   FROM nba_player_season_research
                   WHERE season_id=? AND team_id=? AND endpoint!=?
                   GROUP BY endpoint ORDER BY endpoint''',
                (season["season_id"], season["team_id"], SOURCE_PLAYER_ENDPOINT),
            ),
        })
    return {
        "formulas": FORMULAS,
        "simulator_comparison": SIMULATOR_COMPARISON,
        "seasons": seasons,
        "external_observations": [
            {
                **_external_provenance(row),
                "season_id": row["season_id"],
                "entity_name": row["entity_name"],
                "field_name": row["field_name"],
                "import_status": row["import_status"],
                "status_reason": row["status_reason"],
            }
            for row in external
        ],
        "source_rows_rewritten": False,
    }
