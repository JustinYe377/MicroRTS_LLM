#!/usr/bin/env python3
"""
Update the GitHub Pages site data from tournament results and benchmark leaderboard.

Merges data from:
  - tournament_results/tournament_*.json (tournament runs)
  - benchmark_results/leaderboard.json (existing benchmark data)

Output:
  - docs/data/tournament_results.json (canonical site data file)

Usage:
    python3 tournament/update_site_data.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

TOURNAMENT_RESULTS_DIR = "tournament_results"
BENCHMARK_LEADERBOARD = "benchmark_results/leaderboard.json"
SITE_DATA_FILE = "docs/data/tournament_results.json"

# Anchors for grade calculation
ANCHORS = ["RandomBiasedAI", "HeavyRush", "LightRush", "WorkerRush", "Tiamat", "CoacAI"]


def score_to_grade(score):
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def load_benchmark_entries():
    """Load entries from benchmark_results/leaderboard.json."""
    path = Path(BENCHMARK_LEADERBOARD)
    if not path.exists():
        return []

    with open(path) as f:
        data = json.load(f)

    entries = []
    for entry in data.get("entries", []):
        entries.append({
            "source": "benchmark",
            "team_name": None,
            "display_name": entry.get("model", "Unknown"),
            "agent_class": entry.get("agent_class", ""),
            "model_provider": "ollama",
            "model_name": entry.get("model", "").split(" (")[0],
            "agent_type": entry.get("agent_type", entry.get("format", "unknown")),
            "score": entry.get("score", 0),
            "grade": entry.get("grade", score_to_grade(entry.get("score", 0))),
            "eliminated_at": entry.get("eliminated_at"),
            "date": entry.get("date", ""),
            "map": entry.get("map", ""),
            "games_per_matchup": entry.get("games_per_matchup", 1),
            "opponents": entry.get("opponents", {})
        })

    return entries


def load_tournament_entries():
    """Load entries from tournament result files."""
    results_dir = Path(TOURNAMENT_RESULTS_DIR)
    if not results_dir.exists():
        return []

    entries = []
    for result_file in sorted(results_dir.glob("tournament_*.json")):
        with open(result_file) as f:
            data = json.load(f)

        for team in data.get("results", []):
            # Determine map string for display
            maps = team.get("maps", [])
            if maps:
                map_display = ", ".join(maps)
            else:
                map_display = team.get("map", data.get("config", {}).get("map", ""))

            entries.append({
                "source": "tournament",
                "source_file": result_file.name,
                "team_name": team.get("team_name"),
                "display_name": team.get("display_name", team.get("team_name", "Unknown")),
                "agent_class": team.get("agent_class", ""),
                "model_provider": team.get("model_provider", "unknown"),
                "model_name": team.get("model_name", "unknown"),
                "agent_type": "submission",
                "score": team.get("score", 0),
                "grade": team.get("grade", score_to_grade(team.get("score", 0))),
                "eliminated_at": team.get("eliminated_at"),
                "date": team.get("date", data.get("date", "")),
                "map": map_display,
                "maps": maps,
                "map_scores": team.get("map_scores", {}),
                "games_per_matchup": team.get("games_per_matchup",
                                               data.get("config", {}).get("games_per_matchup", 1)),
                "opponents": team.get("opponents", {})
            })

    return entries


def build_leaderboard(entries):
    """
    Build the leaderboard: best score per unique display_name.
    Also tracks the most recent evaluation date per entry (last_updated).
    Returns sorted list of entries (highest score first).
    """
    best = {}
    latest_date = {}  # track most recent date per display_name
    for entry in entries:
        key = entry["display_name"]
        if key not in best or entry["score"] > best[key]["score"]:
            best[key] = dict(entry)  # copy to avoid mutation
        # Track the most recent date this entry was evaluated
        entry_date = entry.get("date", "")
        if key not in latest_date or entry_date > latest_date[key]:
            latest_date[key] = entry_date

    # Add last_updated and synthesize map_scores for single-map entries
    for key, entry in best.items():
        entry["last_updated"] = latest_date.get(key, entry.get("date", ""))
        if not entry.get("map_scores"):
            raw_map = entry.get("map", "")
            if "8x8" in raw_map:
                entry["map_scores"] = {"8x8": {"score": entry["score"], "grade": entry["grade"]}}
            elif "16x16" in raw_map:
                entry["map_scores"] = {"16x16": {"score": entry["score"], "grade": entry["grade"]}}

    return sorted(best.values(), key=lambda x: x["score"], reverse=True)


def build_history(entries):
    """Build chronological history of all tournament runs."""
    # Group by date
    history = []
    for entry in sorted(entries, key=lambda x: x.get("date", ""), reverse=True):
        h = {
            "display_name": entry["display_name"],
            "score": entry["score"],
            "grade": entry["grade"],
            "date": entry.get("date", ""),
            "source": entry.get("source", "unknown"),
            "map": entry.get("map", ""),
            "agent_class": entry.get("agent_class", ""),
            "opponents": entry.get("opponents", {}),
        }
        if entry.get("map_scores"):
            h["map_scores"] = entry["map_scores"]
        else:
            # Synthesize map_scores for single-map entries (e.g. old benchmarks)
            raw_map = entry.get("map", "")
            if "8x8" in raw_map:
                h["map_scores"] = {"8x8": {"score": entry["score"], "grade": entry["grade"]}}
            elif "16x16" in raw_map:
                h["map_scores"] = {"16x16": {"score": entry["score"], "grade": entry["grade"]}}
        history.append(h)
    return history


def build_h2h_matrix(tournament_entries):
    """Build head-to-head matrix from tournament files."""
    h2h = {}  # {player0: {player1: {wins, losses, draws}}}

    results_dir = Path(TOURNAMENT_RESULTS_DIR)
    if not results_dir.exists():
        return h2h

    for result_file in sorted(results_dir.glob("tournament_*.json")):
        with open(result_file) as f:
            data = json.load(f)

        for game in data.get("head_to_head", []):
            p0 = game.get("player0", "")
            p1 = game.get("player1", "")
            result = game.get("result", "")

            if p0 not in h2h:
                h2h[p0] = {}
            if p1 not in h2h[p0]:
                h2h[p0][p1] = {"wins": 0, "losses": 0, "draws": 0}

            if p1 not in h2h:
                h2h[p1] = {}
            if p0 not in h2h[p1]:
                h2h[p1][p0] = {"wins": 0, "losses": 0, "draws": 0}

            if result == "win":
                h2h[p0][p1]["wins"] += 1
                h2h[p1][p0]["losses"] += 1
            elif result == "loss":
                h2h[p0][p1]["losses"] += 1
                h2h[p1][p0]["wins"] += 1
            elif result == "draw":
                h2h[p0][p1]["draws"] += 1
                h2h[p1][p0]["draws"] += 1

    return h2h


def main():
    print("Updating site data...")

    # Load all entries
    benchmark_entries = load_benchmark_entries()
    tournament_entries = load_tournament_entries()
    all_entries = benchmark_entries + tournament_entries

    print(f"  Benchmark entries: {len(benchmark_entries)}")
    print(f"  Tournament entries: {len(tournament_entries)}")

    # Build outputs
    leaderboard = build_leaderboard(all_entries)
    history = build_history(all_entries)
    h2h = build_h2h_matrix(tournament_entries)

    site_data = {
        "generated": datetime.now().isoformat(),
        "anchors": ANCHORS,
        "maps": [
            {"path": "maps/8x8/basesWorkers8x8.xml", "label": "8x8"},
            {"path": "maps/16x16/basesWorkers16x16.xml", "label": "16x16"},
        ],
        "scoring": {
            "max_score": 100,
            "format": "single-elimination-multimap",
            "description": "Average of per-map scores (each 0-100)",
            "grades": {
                "A+": "90-100",
                "A": "80-89",
                "B": "70-79",
                "C": "60-69",
                "D": "40-59",
                "F": "0-39"
            }
        },
        "leaderboard": leaderboard,
        "history": history,
        "head_to_head": h2h
    }

    # Write output
    output_path = Path(SITE_DATA_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(site_data, f, indent=2)

    print(f"  Leaderboard entries: {len(leaderboard)}")
    print(f"  History entries: {len(history)}")
    print(f"  Written to: {output_path}")


if __name__ == "__main__":
    main()
