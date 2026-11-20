"""Command-line entry point. Run: python -m era_engine.cli init /path/to/era.db"""
import argparse
import json
from pathlib import Path
from .storage import connect, load, save_simulation
from .seed import initialize_demo
from .core import simulate_series, estimate_season


def main():
    parser = argparse.ArgumentParser(description='Era Basketball Python prototype')
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Create SQLite database with fictional TEST data')
    init.add_argument('db')
    sim = sub.add_parser('simulate', help='Run individual games and summary')
    sim.add_argument('db')
    sim.add_argument('--era', default='1995-96')
    sim.add_argument('--games', type=int, default=10)
    sim.add_argument('--seed', type=int, default=7)
    sim.add_argument('--team-a', default='Bay City Legends')
    sim.add_argument('--team-b', default='Chicago Stampede')
    sim.add_argument('--output', default='results.json')
    predict = sub.add_parser('project', help='Monte Carlo 82-game scoring interval (slow)')
    predict.add_argument('db')
    predict.add_argument('--era', default='1995-96')
    predict.add_argument('--runs', type=int, default=20)
    predict.add_argument('--output', default='projection.json')
    args = parser.parse_args()
    con = connect(args.db)
    try:
        if args.command == 'init':
            initialize_demo(con)
            print('Created sample DB:', args.db)
            return
        players, eras, teams = load(con)
        if args.command == 'simulate':
            result = simulate_series(teams[args.team_a], teams[args.team_b], players, eras[args.era], args.games, args.seed)
            result['simulation_id'] = save_simulation(con, result)
        else:
            result = estimate_season(teams['Bay City Legends'], teams['Chicago Stampede'], players, eras[args.era], args.runs)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print('Wrote:', args.output)
        print(json.dumps({'model_version':result['model_version'], 'era': result.get('era', args.era), 'games':result.get('games'),
                          'Sterling Cross PPG': result.get('averages', {}).get('ERA_0001', {}).get('pts'),
                          'note':result.get('note', 'Prototype statistics only')}, indent=2))
    finally:
        con.close()

if __name__ == '__main__':
    main()
