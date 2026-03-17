#!/usr/bin/env python3
"""
MicroRTS Benchmark Runner

Runs all submitted agents against the 6 built-in reference AIs.
No head-to-head, no elimination — every agent plays all 6 opponents
and gets a full score breakdown.

Usage:
    python3 tournament/run_benchmark.py
    python3 tournament/run_benchmark.py --games 3
    python3 tournament/run_benchmark.py --agent yebot
    python3 tournament/run_benchmark.py --games 3 --agent yebot
    python3 tournament/run_benchmark.py --submissions-dir submissions
"""

import subprocess
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Reuse validation from the tournament runner
sys.path.insert(0, str(Path(__file__).parent))
from validate_submission import validate_submission, find_all_submissions

# ── Config (mirrors run_tournament.py exactly) ─────────────────────────────────
CONFIG_FILE  = "resources/config.properties"
RESULTS_DIR  = "benchmark_results"
MAX_CYCLES   = 1500
MAP          = "maps/8x8/basesWorkers8x8.xml"
GAME_TIMEOUT = 900  # 15 min per game

# ── Reference AIs (same weights as tournament) ─────────────────────────────────
ANCHORS = {
    "ai.RandomBiasedAI": {
        "name": "RandomBiasedAI",
        "weight": 10,
        "tier": "easy"
    },
    "ai.abstraction.HeavyRush": {
        "name": "HeavyRush",
        "weight": 20,
        "tier": "medium-hard"
    },
    "ai.abstraction.LightRush": {
        "name": "LightRush",
        "weight": 15,
        "tier": "medium"
    },
    "ai.abstraction.WorkerRush": {
        "name": "WorkerRush",
        "weight": 15,
        "tier": "medium"
    },
    "ai.competition.tiamat.Tiamat": {
        "name": "Tiamat",
        "weight": 20,
        "tier": "hard"
    },
    "ai.coac.CoacAI": {
        "name": "CoacAI",
        "weight": 20,
        "tier": "hard"
    },
}

# ─────────────────────────────────────────────────────────────────────────────
#  GAME EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def update_config(ai1, ai2):
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'^AI1=.*$',        f'AI1={ai1}',          content, flags=re.MULTILINE)
    content = re.sub(r'^AI2=.*$',        f'AI2={ai2}',          content, flags=re.MULTILINE)
    content = re.sub(r'^max_cycles=.*$', f'max_cycles={MAX_CYCLES}', content, flags=re.MULTILINE)
    content = re.sub(r'^headless=.*$',   'headless=true',        content, flags=re.MULTILINE)
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


def run_game(ai1, ai2, name1="", name2=""):
    """Run one game. Returns dict with result/ticks."""
    update_config(ai1, ai2)
    display1 = name1 or ai1.split(".")[-1]
    display2 = name2 or ai2.split(".")[-1]
    print(f"    {display1} vs {display2} ... ", end="", flush=True)

    try:
        result = subprocess.run(
            ["java", "-cp", "lib/*:lib/bots/*:bin", "rts.MicroRTS", "-f", CONFIG_FILE],
            capture_output=True, text=True,
            timeout=GAME_TIMEOUT, env=os.environ.copy()
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"result": "timeout", "ticks": MAX_CYCLES}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"result": "error", "ticks": 0}

    winner = None
    ticks  = MAX_CYCLES

    m = re.search(r'WINNER:\s*(-?\d+)', output)
    if m:
        winner = int(m.group(1))

    m = re.search(r'FINAL_TICK:\s*(\d+)', output)
    if m:
        ticks = int(m.group(1))

    if winner is None:
        if "Player 0 wins" in output:
            winner = 0
        elif "Player 1 wins" in output:
            winner = 1

    if winner == 0:
        print(f"WIN  ({ticks} ticks)")
        return {"result": "win",  "ticks": ticks}
    elif winner == 1:
        print(f"LOSS ({ticks} ticks)")
        return {"result": "loss", "ticks": ticks}
    else:
        print(f"DRAW ({ticks} ticks)")
        return {"result": "draw", "ticks": ticks}


# ─────────────────────────────────────────────────────────────────────────────
#  SCORING (identical to run_tournament.py)
# ─────────────────────────────────────────────────────────────────────────────

def game_score(result, ticks):
    if result == "win":
        if ticks < MAX_CYCLES * 0.50: return 1.2
        if ticks < MAX_CYCLES * 0.75: return 1.1
        return 1.0
    if result == "draw": return 0.5
    return 0.0


def benchmark_score(ref_games):
    """0-100 weighted score across all anchors."""
    total = 0.0
    for anchor_class, anchor_info in ANCHORS.items():
        games = ref_games.get(anchor_class, [])
        if not games:
            continue
        avg = sum(game_score(g["result"], g["ticks"]) for g in games) / len(games)
        total += avg * anchor_info["weight"]
    return round(total, 1)


def grade(score):
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


# ─────────────────────────────────────────────────────────────────────────────
#  SUBMISSION INSTALL + COMPILE
# ─────────────────────────────────────────────────────────────────────────────

def install_submission(sub_dir):
    sub_dir = Path(sub_dir)
    meta    = json.loads((sub_dir / "metadata.json").read_text())
    src     = sub_dir / meta["agent_file"]
    java    = src.read_text()

    pkg_m = re.search(r'^\s*package\s+([\w.]+)\s*;', java, re.MULTILINE)
    if not pkg_m:
        raise ValueError(f"No package in {meta['agent_file']}")

    pkg        = pkg_m.group(1)
    target_dir = Path("src") / pkg.replace(".", "/")
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target_dir / meta["agent_file"])

    fqcn         = f"{pkg}.{meta['agent_class']}"
    display_name = meta.get("display_name", meta["team_name"])
    return fqcn, display_name, meta


def compile_project():
    print("Compiling ... ", end="", flush=True)
    try:
        r = subprocess.run(["ant", "build"], capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            print("OK (ant)")
            return True
        print(f"FAILED (ant)\n{r.stderr}")
        return False
    except FileNotFoundError:
        pass  # no ant

    sources = subprocess.run(["find", "src", "-name", "*.java"],
                              capture_output=True, text=True)
    with open("sources.list", "w") as f:
        f.write(sources.stdout)

    r = subprocess.run(
        ["javac", "-cp", "lib/*:bin", "-d", "bin", "@sources.list"],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode == 0:
        print("OK (javac)")
        return True
    print(f"FAILED\n{r.stderr}")
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(games_per_pair=1, filter_agent=None, submissions_dir="submissions"):
    now       = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    print("=" * 60)
    print("MicroRTS Benchmark — Agents vs Built-in AIs")
    print("=" * 60)
    print(f"Date:            {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"Map:             {MAP}")
    print(f"Max cycles:      {MAX_CYCLES}")
    print(f"Games per pair:  {games_per_pair}")
    if filter_agent:
        print(f"Filter:          {filter_agent}")
    print()

    # ── Discover ──────────────────────────────────────────────────────────────
    print("DISCOVERING SUBMISSIONS")
    print("-" * 40)
    all_subs   = find_all_submissions(submissions_dir)
    valid_subs = []

    for sub in all_subs:
        # Optional single-agent filter
        if filter_agent and sub.name.lower() != filter_agent.lower():
            continue
        ok, errors = validate_submission(sub)
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"  {tag} {sub.name}")
        for e in errors:
            print(f"         {e}")
        if ok:
            valid_subs.append(sub)

    if not valid_subs:
        print("No valid submissions found.")
        sys.exit(1)
    print(f"\n{len(valid_subs)} agent(s) to benchmark\n")

    # ── Install ───────────────────────────────────────────────────────────────
    print("INSTALLING")
    print("-" * 40)
    contestants = {}  # fqcn → {display_name, meta}
    for sub in valid_subs:
        try:
            fqcn, display, meta = install_submission(sub)
            contestants[fqcn] = {"display_name": display, "meta": meta}
            print(f"  Installed: {display} ({fqcn})")
        except Exception as e:
            print(f"  FAILED:    {sub.name} — {e}")
    print()

    # ── Compile ───────────────────────────────────────────────────────────────
    if not compile_project():
        print("Build failed. Aborting.")
        sys.exit(1)
    print()

    # ── Run games ─────────────────────────────────────────────────────────────
    print("BENCHMARK GAMES")
    print("-" * 40)
    print(f"Each agent plays all {len(ANCHORS)} built-in AIs × {games_per_pair} game(s)\n")

    all_data = {}  # display_name → {ref_games}

    for fqcn, info in contestants.items():
        name = info["display_name"]
        all_data[name] = {"ref_games": {}}

        print(f"── {name} ──")
        for anchor_class, anchor_info in ANCHORS.items():
            results = []
            for g in range(games_per_pair):
                if games_per_pair > 1:
                    print(f"  [Game {g+1}/{games_per_pair}]")
                r = run_game(fqcn, anchor_class, name, anchor_info["name"])
                results.append(r)
            all_data[name]["ref_games"][anchor_class] = results
        print()

    # ── Results table ─────────────────────────────────────────────────────────
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    scores = {
        name: benchmark_score(data["ref_games"])
        for name, data in all_data.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Summary table
    col_w = 28
    print(f"{'Rank':<5} {'Agent':<{col_w}} {'Score':>6}  {'Grade':<4}  "
          f"  Ran  Hvy  Lgt  Wrk  Tia  Coa")
    print("-" * 75)

    anchor_keys = list(ANCHORS.keys())

    for rank, (name, score) in enumerate(ranked, 1):
        ref = all_data[name]["ref_games"]

        # Per-opponent result summary (W/D/L)
        per_opp = []
        for ak in anchor_keys:
            games = ref.get(ak, [])
            if not games:
                per_opp.append("  - ")
                continue
            wins  = sum(1 for g in games if g["result"] == "win")
            draws = sum(1 for g in games if g["result"] == "draw")
            total = len(games)
            if total == 1:
                r = games[0]["result"]
                per_opp.append(f"  {'W' if r=='win' else ('D' if r=='draw' else 'L')} ")
            else:
                per_opp.append(f"{wins}W{draws}D ")

        print(f"{rank:<5} {name:<{col_w}} {score:>6.1f}  {grade(score):<4}  "
              + "  ".join(per_opp))

    print()

    # Detailed breakdown per agent
    print("DETAILED BREAKDOWN")
    print("-" * 60)
    for name, score in ranked:
        ref = all_data[name]["ref_games"]
        print(f"\n{name}  (score={score}, grade={grade(score)})")
        for ak, ainfo in ANCHORS.items():
            games = ref.get(ak, [])
            if not games:
                continue
            avg  = sum(game_score(g["result"], g["ticks"]) for g in games) / len(games)
            pts  = round(avg * ainfo["weight"], 1)
            wins = sum(1 for g in games if g["result"] == "win")
            draws= sum(1 for g in games if g["result"] == "draw")
            losses=sum(1 for g in games if g["result"] not in ("win","draw"))
            ticks_avg = int(sum(g["ticks"] for g in games) / len(games))
            print(f"  vs {ainfo['name']:<14} {wins}W {draws}D {losses}L  "
                  f"avg_ticks={ticks_avg:<5}  pts={pts}/{ainfo['weight']}")
        print(f"  {'─'*40}")
        print(f"  TOTAL: {score}/100  ({grade(score)})")

    # ── Save results ──────────────────────────────────────────────────────────
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    out_path = f"{RESULTS_DIR}/benchmark_{timestamp}.json"

    output = {
        "date":    now.isoformat(),
        "map":     MAP,
        "games_per_pair": games_per_pair,
        "anchors": {
            cls: {"name": i["name"], "weight": i["weight"], "tier": i["tier"]}
            for cls, i in ANCHORS.items()
        },
        "results": [
            {
                "agent":    name,
                "score":    score,
                "grade":    grade(score),
                "per_opponent": {
                    ANCHORS[ak]["name"]: {
                        "games": all_data[name]["ref_games"].get(ak, []),
                        "weighted_pts": round(
                            (sum(game_score(g["result"], g["ticks"])
                             for g in all_data[name]["ref_games"].get(ak, []))
                             / max(1, len(all_data[name]["ref_games"].get(ak, []))))
                            * ANCHORS[ak]["weight"], 1
                        )
                    }
                    for ak in ANCHORS
                }
            }
            for name, score in ranked
        ]
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    games           = 1
    filter_agent    = None
    submissions_dir = "submissions"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--games" and i + 1 < len(args):
            games = int(args[i + 1])
            i += 2
        elif args[i] == "--agent" and i + 1 < len(args):
            filter_agent = args[i + 1]
            i += 2
        elif args[i] == "--submissions-dir" and i + 1 < len(args):
            submissions_dir = args[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            print(__doc__)
            sys.exit(1)

    run_benchmark(
        games_per_pair  = games,
        filter_agent    = filter_agent,
        submissions_dir = submissions_dir
    )