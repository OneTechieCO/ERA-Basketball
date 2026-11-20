"""Optional server-side NBA research fetcher; never bundled or invoked by game clients.

Use ONLY with permission to access and reuse NBA data in the intended product.
Requires: pip install nba_api pandas requests
Usage: python -m era_engine.nba_import research.db 201939
       python -m era_engine.nba_import research.db --verified-seasons

Parsed rows are research records only. They are not player ratings and they
are not written into fictional player profiles.
"""
import argparse
import json
from .storage import connect, create_research_schema

SOURCE_NAME = "nba.com via nba_api"

# Season-total counting stats requested for the research tables.
# Rates such as FG_PCT stay in the raw response and are not copied here.
COUNTING_FIELDS = (
    ("GP", "games"),
    ("MIN", "minutes"),
    ("PTS", "points"),
    ("FGM", "fgm"),
    ("FGA", "fga"),
    ("FG3M", "fg3m"),
    ("FG3A", "fg3a"),
    ("FTM", "ftm"),
    ("FTA", "fta"),
    ("AST", "assists"),
    ("OREB", "offensive_rebounds"),
    ("DREB", "defensive_rebounds"),
    ("STL", "steals"),
    ("BLK", "blocks"),
    ("TOV", "turnovers"),
)
USAGE_FIELDS = ("USG_PCT", "E_USG_PCT")

PLAYER_TARGETS = (
    {"player_id": 893, "player_name": "Michael Jordan", "season_id": "1995-96"},
    {"player_id": 201939, "player_name": "Stephen Curry", "season_id": "2015-16"},
)
TEAM_TARGETS = (
    {"team_id": 1610612741, "team_name": "Chicago Bulls", "season_id": "1995-96"},
    {"team_id": 1610612744, "team_name": "Golden State Warriors", "season_id": "2015-16"},
)


def import_player_seasons(db_path: str, nba_player_id: int) -> int:
    try:
        from nba_api.stats.endpoints import playercareerstats
    except ImportError as exc:
        raise RuntimeError('Install optional research package: pip install nba_api pandas requests') from exc
    con = connect(db_path)
    create_research_schema(con)
    # Requests can be blocked or rate-limited; failures are intentional errors, no fabricated fallback.
    data = playercareerstats.PlayerCareerStats(player_id=nba_player_id, timeout=30)
    df = data.season_totals_regular_season.get_data_frame()
    count = 0
    for record in df.to_dict(orient='records'):
        season = str(record['SEASON_ID'])
        con.execute('INSERT OR REPLACE INTO nba_season_research(nba_player_id,season_id,source,stats_json) VALUES(?,?,?,?)',
                    (nba_player_id, season, 'nba.com via nba_api', json.dumps(record, default=str, allow_nan=False)))
        count += 1
    con.commit()
    con.close()
    return count


def dataset_table(raw: dict, dataset_name: str) -> tuple[list[str], list[dict]]:
    """Return headers and row dicts for one named result set. Missing sets are empty."""
    for result_set in raw.get("resultSets") or []:
        if result_set.get("name") != dataset_name:
            continue
        headers = [str(header) for header in result_set.get("headers") or []]
        rows = []
        for row in result_set.get("rowSet") or []:
            rows.append({header: row[index] if index < len(row) else None for index, header in enumerate(headers)})
        return headers, rows
    return [], []


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def counting_columns(record: dict, measure: str | None) -> dict:
    """Copy only counting fields that are present. Absent fields stay NULL."""
    columns = {"counting_measure": measure if any(key in record for key, _ in COUNTING_FIELDS) else None}
    for source_name, column in COUNTING_FIELDS:
        columns[column] = _number(record[source_name]) if source_name in record else None
    return columns


def usage_columns(record: dict) -> dict:
    for source_name in USAGE_FIELDS:
        if source_name in record and _number(record[source_name]) is not None:
            return {"usage_rate": _number(record[source_name]), "usage_rate_measure": "rate"}
    return {"usage_rate": None, "usage_rate_measure": None}


def pace_columns(record: dict, possessions_measure: str | None) -> dict:
    pace = _number(record["PACE"]) if "PACE" in record else None
    possessions = _number(record["POSS"]) if "POSS" in record else None
    return {
        "pace": pace,
        "pace_measure": "rate" if pace is not None else None,
        "possessions": possessions,
        "possessions_measure": possessions_measure if possessions is not None else None,
    }


def store_source_response(con, *, source_name, endpoint, dataset_name, season_id,
                          entity_kind, entity_id, entity_name, per_mode, retrieved_on,
                          field_names, raw) -> None:
    con.execute(
        '''INSERT OR REPLACE INTO nba_source_response(
             source_name, endpoint, dataset_name, season_id, entity_kind, entity_id,
             entity_name, per_mode, retrieved_on, field_names_json, raw_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
        (source_name, endpoint, dataset_name, season_id, entity_kind, entity_id,
         entity_name, per_mode, retrieved_on, json.dumps(field_names),
         json.dumps(raw, default=str, allow_nan=False)),
    )


def store_player_season(con, *, player_id, player_name, season_id, record, source_name,
                        endpoint, dataset_name, counting_measure) -> bool:
    """Store one regular-season player row. Returns False when the season or team id is absent."""
    if str(record.get("SEASON_ID")) != season_id or record.get("TEAM_ID") is None:
        return False
    values = counting_columns(record, counting_measure)
    values.update(usage_columns(record))
    con.execute(
        '''INSERT OR REPLACE INTO nba_player_season_research(
             nba_player_id, player_name, season_id, team_id, team_abbreviation,
             source_name, endpoint, dataset_name, counting_measure,
             games, minutes, points, fgm, fga, fg3m, fg3a, ftm, fta, assists,
             offensive_rebounds, defensive_rebounds, steals, blocks, turnovers,
             usage_rate, usage_rate_measure)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (player_id, player_name, season_id, record["TEAM_ID"], record.get("TEAM_ABBREVIATION"),
         source_name, endpoint, dataset_name, values["counting_measure"],
         values["games"], values["minutes"], values["points"], values["fgm"], values["fga"],
         values["fg3m"], values["fg3a"], values["ftm"], values["fta"], values["assists"],
         values["offensive_rebounds"], values["defensive_rebounds"], values["steals"],
         values["blocks"], values["turnovers"], values["usage_rate"], values["usage_rate_measure"]),
    )
    return True


def store_team_season(con, *, team_id, team_name, season_id, record, source_name,
                      endpoint, dataset_name, counting_measure, possessions_measure) -> None:
    values = counting_columns(record, counting_measure)
    values.update(pace_columns(record, possessions_measure))
    con.execute(
        '''INSERT OR REPLACE INTO nba_team_season_research(
             team_id, team_name, season_id, source_name, endpoint, dataset_name,
             counting_measure, games, minutes, points, fgm, fga, fg3m, fg3a, ftm, fta,
             assists, offensive_rebounds, defensive_rebounds, steals, blocks, turnovers,
             pace, pace_measure, possessions, possessions_measure)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
        (team_id, team_name, season_id, source_name, endpoint, dataset_name,
         values["counting_measure"], values["games"], values["minutes"], values["points"],
         values["fgm"], values["fga"], values["fg3m"], values["fg3a"], values["ftm"],
         values["fta"], values["assists"], values["offensive_rebounds"],
         values["defensive_rebounds"], values["steals"], values["blocks"], values["turnovers"],
         values["pace"], values["pace_measure"], values["possessions"], values["possessions_measure"]),
    )


def import_player_career_response(con, raw: dict, *, player_id: int, player_name: str,
                                  season_id: str, retrieved_on: str,
                                  counting_measure: str = "season_total") -> int:
    """Keep the raw career response and the one regular-season row for season_id."""
    endpoint = str(raw.get("resource") or "playercareerstats")
    per_mode = str((raw.get("parameters") or {}).get("PerMode") or "")
    headers, rows = dataset_table(raw, "SeasonTotalsRegularSeason")
    store_source_response(
        con, source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="SeasonTotalsRegularSeason",
        season_id=season_id, entity_kind="player", entity_id=player_id, entity_name=player_name,
        per_mode=per_mode, retrieved_on=retrieved_on, field_names=headers, raw=raw,
    )
    stored = 0
    for record in rows:
        if int(record.get("PLAYER_ID") or player_id) != int(player_id):
            continue
        if store_player_season(
            con, player_id=player_id, player_name=player_name, season_id=season_id, record=record,
            source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="SeasonTotalsRegularSeason",
            counting_measure=counting_measure if per_mode == "Totals" else None,
        ):
            stored += 1
    return stored


def import_team_year_response(con, raw: dict, *, team_id: int, team_name: str, season_id: str,
                              retrieved_on: str) -> int:
    endpoint = str(raw.get("resource") or "teamyearbyyearstats")
    per_mode = str((raw.get("parameters") or {}).get("PerMode") or "")
    headers, rows = dataset_table(raw, "TeamStats")
    store_source_response(
        con, source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="TeamStats",
        season_id=season_id, entity_kind="team", entity_id=team_id, entity_name=team_name,
        per_mode=per_mode, retrieved_on=retrieved_on, field_names=headers, raw=raw,
    )
    measure = "season_total" if per_mode == "Totals" else None
    stored = 0
    matched = [row for row in rows if str(row.get("YEAR")) == season_id and int(row.get("TEAM_ID") or 0) == int(team_id)]
    for record in matched:
        store_team_season(
            con, team_id=team_id, team_name=team_name, season_id=season_id, record=record,
            source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="TeamStats",
            counting_measure=measure, possessions_measure=measure,
        )
        stored += 1
    if not matched:
        store_team_season(
            con, team_id=team_id, team_name=team_name, season_id=season_id, record={},
            source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="TeamStats",
            counting_measure=None, possessions_measure=None,
        )
    return stored


def import_team_advanced_response(con, raw: dict, *, team_id: int, team_name: str, season_id: str,
                                  retrieved_on: str) -> int:
    """Store pace and possessions only when that advanced row was returned."""
    endpoint = str(raw.get("resource") or "leaguedashteamstats")
    per_mode = str((raw.get("parameters") or {}).get("PerMode") or "")
    headers, rows = dataset_table(raw, "LeagueDashTeamStats")
    store_source_response(
        con, source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="LeagueDashTeamStats",
        season_id=season_id, entity_kind="team", entity_id=team_id, entity_name=team_name,
        per_mode=per_mode, retrieved_on=retrieved_on, field_names=headers, raw=raw,
    )
    matched = [row for row in rows if int(row.get("TEAM_ID") or 0) == int(team_id)]
    possessions_measure = "season_total" if per_mode == "Totals" else None
    if not matched:
        store_team_season(
            con, team_id=team_id, team_name=team_name, season_id=season_id, record={},
            source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="LeagueDashTeamStats",
            counting_measure=None, possessions_measure=None,
        )
        return 0
    for record in matched:
        # Advanced rows are not the season counting line. Keep pace, possessions,
        # games, and minutes from this response; leave the other counting columns
        # NULL even if a future payload adds them under a different meaning.
        limited = {key: record[key] for key in ("GP", "MIN", "PACE", "POSS") if key in record}
        store_team_season(
            con, team_id=team_id, team_name=team_name, season_id=season_id, record=limited,
            source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="LeagueDashTeamStats",
            counting_measure="season_total" if per_mode == "Totals" and any(key in limited for key in ("GP", "MIN")) else None,
            possessions_measure=possessions_measure,
        )
    return len(matched)


def _regular_season_record(raw: dict, record: dict, season_id: str) -> dict | None:
    """Label a row with the request season when the row itself has no SEASON_ID.

    The season is taken from the saved request parameters. A playoff payload and a
    parameter season other than season_id are not stored as this regular season.
    """
    parameters = raw.get("parameters") or {}
    season_type = parameters.get("SeasonType")
    if season_type not in (None, "", "Regular Season"):
        return None
    labeled = dict(record)
    if labeled.get("SEASON_ID") is None:
        if str(parameters.get("Season") or "") != season_id:
            return None
        labeled["SEASON_ID"] = season_id
    elif str(labeled.get("SEASON_ID")) != season_id:
        return None
    return labeled


def _is_requested_team(record: dict, team_id: int) -> bool:
    """True only for this team's own row. Combined TOT rows and other teams are not this team."""
    if record.get("TEAM_ID") is None:
        return False
    if str(record.get("TEAM_ABBREVIATION") or "").upper() == "TOT":
        return False
    return int(record["TEAM_ID"]) == int(team_id)


def import_roster_response(con, raw: dict, *, team_id: int, team_name: str, season_id: str,
                           retrieved_on: str) -> list[tuple[int, str]]:
    """Store the roster payload. Position is not a starting lineup and is not a stat line."""
    endpoint = str(raw.get("resource") or "commonteamroster")
    per_mode = str((raw.get("parameters") or {}).get("PerMode") or "")
    headers, rows = dataset_table(raw, "CommonTeamRoster")
    store_source_response(
        con, source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="CommonTeamRoster",
        season_id=season_id, entity_kind="team", entity_id=team_id, entity_name=team_name,
        per_mode=per_mode, retrieved_on=retrieved_on, field_names=headers, raw=raw,
    )
    players = []
    for record in rows:
        if record.get("PLAYER_ID") is None:
            continue
        if int(record.get("TeamID") or team_id) != int(team_id):
            continue
        # CommonTeamRoster labels the season by its starting year ("1995"),
        # while the stat endpoints use the full season id ("1995-96").
        roster_season = str(record.get("SEASON") or season_id)
        if roster_season not in (season_id, season_id.split("-")[0]):
            continue
        name = record.get("PLAYER") or record.get("PLAYER_NAME") or ""
        players.append((int(record["PLAYER_ID"]), str(name)))
    return players


def import_league_dash_player_response(con, raw: dict, *, team_id: int, team_name: str,
                                       season_id: str, retrieved_on: str) -> tuple[int, int]:
    """Store dash rows only when the row's own team id is the requested team.

    A team-filtered response can still contain another team's row. That row stays
    in the raw payload and is not rewritten onto the requested team.
    """
    endpoint = str(raw.get("resource") or "leaguedashplayerstats")
    per_mode = str((raw.get("parameters") or {}).get("PerMode") or "")
    headers, rows = dataset_table(raw, "LeagueDashPlayerStats")
    store_source_response(
        con, source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="LeagueDashPlayerStats",
        season_id=season_id, entity_kind="team", entity_id=team_id, entity_name=team_name,
        per_mode=per_mode, retrieved_on=retrieved_on, field_names=headers, raw=raw,
    )
    measure = "season_total" if per_mode == "Totals" else None
    stored = rejected = 0
    for record in rows:
        if not _is_requested_team(record, team_id) or record.get("PLAYER_ID") is None:
            rejected += 1
            continue
        labeled = _regular_season_record(raw, record, season_id)
        if labeled is None:
            rejected += 1
            continue
        if store_player_season(
            con, player_id=int(labeled["PLAYER_ID"]), player_name=str(labeled.get("PLAYER_NAME") or ""),
            season_id=season_id, record=labeled, source_name=SOURCE_NAME, endpoint=endpoint,
            dataset_name="LeagueDashPlayerStats", counting_measure=measure,
        ):
            stored += 1
        else:
            rejected += 1
    return stored, rejected


def import_team_dashboard_response(con, raw: dict, *, team_id: int, team_name: str, season_id: str,
                                   retrieved_on: str) -> int:
    """Keep the dashboard payload and do not invent a team id for its player rows.

    PlayersSeasonTotals, as returned for these seasons, has no TEAM_ID. Attaching
    the request's team id would attribute rows the source itself did not label.
    """
    endpoint = str(raw.get("resource") or "teamplayerdashboard")
    per_mode = str((raw.get("parameters") or {}).get("PerMode") or "")
    headers, rows = dataset_table(raw, "PlayersSeasonTotals")
    store_source_response(
        con, source_name=SOURCE_NAME, endpoint=endpoint, dataset_name="PlayersSeasonTotals",
        season_id=season_id, entity_kind="team", entity_id=team_id, entity_name=team_name,
        per_mode=per_mode, retrieved_on=retrieved_on, field_names=headers, raw=raw,
    )
    if "TEAM_ID" not in headers:
        return len(rows)
    measure = "season_total" if per_mode == "Totals" else None
    for record in rows:
        if not _is_requested_team(record, team_id) or record.get("PLAYER_ID") is None:
            continue
        labeled = _regular_season_record(raw, record, season_id)
        if labeled is None:
            continue
        store_player_season(
            con, player_id=int(labeled["PLAYER_ID"]), player_name=str(labeled.get("PLAYER_NAME") or ""),
            season_id=season_id, record=labeled, source_name=SOURCE_NAME, endpoint=endpoint,
            dataset_name="PlayersSeasonTotals", counting_measure=measure,
        )
    return 0


ROSTER_PAYLOADS = (
    {
        "season_id": "1995-96",
        "team_id": 1610612741,
        "team_name": "Chicago Bulls",
        "roster": "bulls_1995_roster.json",
        "dash": "bulls_1995_players_dash.json",
        "dashboard": "bulls_1995_dashboard.json",
    },
    {
        "season_id": "2015-16",
        "team_id": 1610612744,
        "team_name": "Golden State Warriors",
        "roster": "gsw_2015_roster.json",
        "dash": "gsw_2015_players_dash.json",
        "dashboard": "gsw_2015_dashboard.json",
    },
)


def _named_players(raw: dict, dataset_name: str) -> list[tuple[int, str]]:
    _headers, rows = dataset_table(raw, dataset_name)
    found = []
    for record in rows:
        if record.get("PLAYER_ID") is None:
            continue
        name = record.get("PLAYER_NAME") or record.get("PLAYER") or ""
        found.append((int(record["PLAYER_ID"]), str(name)))
    return found


def _load_json(path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieved payload is missing: {path}")
    return json.loads(path.read_text())


def load_team_rosters(con, directory: str, retrieved_on: str) -> dict:
    """Import saved roster, dash, and career payloads. Does not invent a missing player."""
    from pathlib import Path
    root = Path(directory)
    summary = {
        "roster_players": {},
        "dash_rows": {},
        "dash_rejected_other_team": {},
        "dashboard_rows_without_team_id": {},
        "career_rows": {},
        "missing_career_files": [],
    }
    for target in ROSTER_PAYLOADS:
        season_id = target["season_id"]
        team_id = target["team_id"]
        team_name = target["team_name"]
        roster_raw = _load_json(root / target["roster"])
        dash_raw = _load_json(root / target["dash"])
        dashboard_raw = _load_json(root / target["dashboard"])
        roster_players = import_roster_response(
            con, roster_raw, team_id=team_id, team_name=team_name,
            season_id=season_id, retrieved_on=retrieved_on,
        )
        stored, rejected = import_league_dash_player_response(
            con, dash_raw, team_id=team_id, team_name=team_name,
            season_id=season_id, retrieved_on=retrieved_on,
        )
        unparsed = import_team_dashboard_response(
            con, dashboard_raw, team_id=team_id, team_name=team_name,
            season_id=season_id, retrieved_on=retrieved_on,
        )
        names = {player_id: name for player_id, name in roster_players}
        for player_id, name in _named_players(dash_raw, "LeagueDashPlayerStats"):
            names.setdefault(player_id, name)
        for player_id, name in _named_players(dashboard_raw, "PlayersSeasonTotals"):
            names.setdefault(player_id, name)
        career_rows = 0
        for player_id, name in names.items():
            path = root / "careers" / f"{player_id}.json"
            if not path.is_file():
                summary["missing_career_files"].append({
                    "player_id": player_id, "player_name": name, "season_id": season_id, "team_id": team_id,
                })
                continue
            career_rows += import_player_career_response(
                con, json.loads(path.read_text()), player_id=player_id, player_name=name,
                season_id=season_id, retrieved_on=retrieved_on,
            )
        summary["roster_players"][season_id] = len(roster_players)
        summary["dash_rows"][season_id] = stored
        summary["dash_rejected_other_team"][season_id] = rejected
        summary["dashboard_rows_without_team_id"][season_id] = unparsed
        summary["career_rows"][season_id] = career_rows
    con.commit()
    return summary


def audit_team_player_totals(con, *, team_id: int, season_id: str,
                             player_endpoint: str = "playercareerstats",
                             player_dataset: str = "SeasonTotalsRegularSeason",
                             team_endpoint: str = "teamyearbyyearstats",
                             team_dataset: str = "TeamStats") -> dict:
    """Compare one team's player rows with that team's season row. Does not change either side."""
    columns = [column for _source, column in COUNTING_FIELDS]
    player_sums = con.execute(
        f'''SELECT COUNT(*), {", ".join(f"SUM({column})" for column in columns)}
            FROM nba_player_season_research
            WHERE team_id=? AND season_id=? AND endpoint=? AND dataset_name=?
              AND UPPER(COALESCE(team_abbreviation, '')) != 'TOT' ''',
        (team_id, season_id, player_endpoint, player_dataset),
    ).fetchone()
    team_row = con.execute(
        f'''SELECT {", ".join(columns)}
            FROM nba_team_season_research
            WHERE team_id=? AND season_id=? AND endpoint=? AND dataset_name=?''',
        (team_id, season_id, team_endpoint, team_dataset),
    ).fetchone()
    report = {"team_id": team_id, "season_id": season_id, "player_rows": player_sums[0] or 0, "fields": {}}
    for index, column in enumerate(columns):
        player_value = player_sums[index + 1]
        team_value = team_row[index] if team_row is not None else None
        difference = None if player_value is None or team_value is None else team_value - player_value
        report["fields"][column] = {
            "players": player_value, "team": team_value, "team_minus_players": difference,
        }
    return report


def load_verified_research(con, directory: str, retrieved_on: str) -> dict[str, int]:
    """Load previously saved endpoint payloads from directory. Does not invent missing files."""
    from pathlib import Path
    root = Path(directory)
    required = {
        "jordan_career.json": ("player", 893, "Michael Jordan", "1995-96"),
        "curry_career.json": ("player", 201939, "Stephen Curry", "2015-16"),
        "bulls_year.json": ("team_year", 1610612741, "Chicago Bulls", "1995-96"),
        "warriors_year.json": ("team_year", 1610612744, "Golden State Warriors", "2015-16"),
        "league_1995_advanced.json": ("team_advanced", 1610612741, "Chicago Bulls", "1995-96"),
        "gsw_2015_advanced.json": ("team_advanced", 1610612744, "Golden State Warriors", "2015-16"),
    }
    counts = {"player_rows": 0, "team_rows": 0, "advanced_rows": 0}
    for filename, (kind, entity_id, name, season_id) in required.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Retrieved payload is missing: {path}")
        raw = json.loads(path.read_text())
        if kind == "player":
            counts["player_rows"] += import_player_career_response(
                con, raw, player_id=entity_id, player_name=name, season_id=season_id, retrieved_on=retrieved_on,
            )
        elif kind == "team_year":
            counts["team_rows"] += import_team_year_response(
                con, raw, team_id=entity_id, team_name=name, season_id=season_id, retrieved_on=retrieved_on,
            )
        else:
            counts["advanced_rows"] += import_team_advanced_response(
                con, raw, team_id=entity_id, team_name=name, season_id=season_id, retrieved_on=retrieved_on,
            )
    con.commit()
    return counts


def main():
    p = argparse.ArgumentParser(description='Research-only NBA importer; permission required')
    p.add_argument('db')
    p.add_argument('nba_player_id', type=int, nargs='?')
    p.add_argument('--verified-seasons', metavar='PAYLOAD_DIR',
                   help='Load saved PlayerCareerStats, TeamYearByYearStats, and advanced team payloads')
    p.add_argument('--team-rosters', metavar='PAYLOAD_DIR',
                   help='Load saved roster, league dash, and career payloads for the two historical teams')
    p.add_argument('--retrieved-on', default='2026-11-11')
    args = p.parse_args()
    if args.verified_seasons or args.team_rosters:
        con = connect(args.db)
        create_research_schema(con)
        if args.verified_seasons:
            print(load_verified_research(con, args.verified_seasons, args.retrieved_on))
        if args.team_rosters:
            print(load_team_rosters(con, args.team_rosters, args.retrieved_on))
            for target in TEAM_TARGETS:
                print(audit_team_player_totals(
                    con, team_id=target["team_id"], season_id=target["season_id"],
                ))
        con.close()
        return
    if args.nba_player_id is None:
        raise SystemExit('nba_player_id is required unless --verified-seasons is set')
    print('Imported season rows:', import_player_seasons(args.db, args.nba_player_id))

if __name__ == '__main__':
    main()
