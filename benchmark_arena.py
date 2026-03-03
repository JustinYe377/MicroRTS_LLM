#!/usr/bin/env python3
"""
MicroRTS LLM Benchmark Arena v2.0

Single-elimination benchmark against 6 reference AIs in ascending difficulty.
LLMs must WIN to advance to the next opponent. A loss/draw/timeout eliminates.

Elimination order:
  1. RandomBiasedAI (easy, 10 pts)
  2. HeavyRush (medium-hard, 20 pts)
  3. LightRush (medium, 15 pts)
  4. WorkerRush (medium, 15 pts)
  5. Tiamat (hard, 20 pts)
  6. CoacAI (hard, 20 pts)
  Total: 0-100 scale

Usage:
    python3 benchmark_arena.py [--games N]

Environment Variables:
    GEMINI_API_KEY - Required for Gemini models
    DEEPSEEK_API_KEY - Required for DeepSeek cloud models
    DEEPSEEK_MODEL - Model for DeepSeek cloud (default: deepseek-chat)
    OLLAMA_MODEL - Model for ai.abstraction.ollama (default: llama3.1:8b)
    OLLAMA_MODEL_P2 - Model for ai.abstraction.ollama2 (default: qwen3:4b)
"""

import subprocess
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Configuration
CONFIG_FILE = "resources/config.properties"
RESULTS_DIR = "benchmark_results"
MAX_CYCLES = 1500  # Increased for cloud API testing
MAP = "maps/8x8/basesWorkers8x8.xml"
GAME_TIMEOUT = 900  # 15 minutes per game (hard opponents play longer)

# Reference AI anchors in single-elimination order (must win to advance)
# These are FIXED - they provide the stable baseline for all comparisons
ANCHORS = {
    "ai.RandomBiasedAI": {
        "name": "RandomBiasedAI",
        "weight": 10,  # Easy - sanity check
        "tier": "easy"
    },
    "ai.abstraction.HeavyRush": {
        "name": "HeavyRush",
        "weight": 20,  # Medium-Hard - heavy unit pressure
        "tier": "medium-hard"
    },
    "ai.abstraction.LightRush": {
        "name": "LightRush",
        "weight": 15,  # Medium - aggressive light units
        "tier": "medium"
    },
    "ai.abstraction.WorkerRush": {
        "name": "WorkerRush",
        "weight": 15,  # Medium - aggressive workers
        "tier": "medium"
    },
    "ai.competition.tiamat.Tiamat": {
        "name": "Tiamat",
        "weight": 20,  # Hard - competition-winning bot
        "tier": "hard"
    },
    "ai.coac.CoacAI": {
        "name": "CoacAI",
        "weight": 20,  # Hard - competition-winning bot
        "tier": "hard"
    },
}

# LLM contestants
# Each entry: class -> {display name, agent type, env overrides}
# Agent types: PureLLM (every tick), Hybrid (rule-based + periodic LLM),
#              Search+LLM (MCTS with LLM policy priors)
LLMS = {}

# Gemini - only include if API key is available
#if os.environ.get("GEMINI_API_KEY"):
#    LLMS["ai.abstraction.LLM_Gemini"] = {
#        "name": "gemini",
#        "display": "gemini-2.5-flash (PureLLM)",
#        "agent_type": "PureLLM",
#        "env": {"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")}
#    }

# DeepSeek Cloud API - only include if API key is available
if os.environ.get("DEEPSEEK_API_KEY"):
    _deepseek_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    LLMS["ai.abstraction.LLM_DeepSeek"] = {
        "name": "deepseek-cloud",
        "display": f"{_deepseek_model} (PureLLM-Cloud)",
        "agent_type": "PureLLM",
        "env": {
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
            "DEEPSEEK_MODEL": _deepseek_model
        }
    }

LLMS.update({
    "ai.abstraction.ollama": {
        "name": "ollama",
        "display": None,
        "agent_type": "PureLLM",
        "env": {"OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3.1:8b")}
    },
    "ai.abstraction.ollama2": {
        "name": "ollama2",
        "display": None,
        "agent_type": "PureLLM",
        "env": {"OLLAMA_MODEL_P2": os.environ.get("OLLAMA_MODEL_P2", "qwen3:4b")}
    },
    "ai.abstraction.HybridLLMRush": {
        "name": "hybrid",
        "display": None,
        "agent_type": "Hybrid",
        "env": {"OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3.1:8b")}
    },
    "ai.mcts.llmguided.LLMInformedMCTS": {
        "name": "mcts",
        "display": None,
        "agent_type": "Search+LLM",
        "env": {"OLLAMA_MODEL": os.environ.get("OLLAMA_MODEL", "llama3.1:8b")}
    },
})

# Set display names from env (model + agent type)
_ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
_ollama_model_p2 = os.environ.get("OLLAMA_MODEL_P2", "qwen3:4b")
LLMS["ai.abstraction.ollama"]["display"] = f"{_ollama_model} (PureLLM)"
LLMS["ai.abstraction.ollama2"]["display"] = f"{_ollama_model_p2} (PureLLM)"
LLMS["ai.abstraction.HybridLLMRush"]["display"] = f"{_ollama_model} (Hybrid)"
LLMS["ai.mcts.llmguided.LLMInformedMCTS"]["display"] = f"{_ollama_model} (Search+LLM)"


def update_config(ai1, ai2):
    """Update config.properties with AI settings."""
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    content = re.sub(r'^AI1=.*$', f'AI1={ai1}', content, flags=re.MULTILINE)
    content = re.sub(r'^AI2=.*$', f'AI2={ai2}', content, flags=re.MULTILINE)
    content = re.sub(r'^max_cycles=.*$', f'max_cycles={MAX_CYCLES}', content, flags=re.MULTILINE)
    content = re.sub(r'^headless=.*$', 'headless=true', content, flags=re.MULTILINE)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


def run_game(ai1, ai2):
    """Run a single game and return result."""
    update_config(ai1, ai2)

    env = os.environ.copy()
    if ai1 in LLMS:
        env.update(LLMS[ai1]["env"])
    if ai2 in LLMS:
        env.update(LLMS[ai2]["env"])

    ai1_name = LLMS.get(ai1, {}).get("display", ai1.split(".")[-1])
    ai2_name = LLMS.get(ai2, {}).get("display", ai2.split(".")[-1])

    print(f"  {ai1_name} vs {ai2_name}...", end=" ", flush=True)

    try:
        result = subprocess.run(
            ["java", "-cp", "lib/*:lib/bots/*:bin", "rts.MicroRTS", "-f", CONFIG_FILE],
            capture_output=True,
            text=True,
            timeout=GAME_TIMEOUT,
            env=env
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return {"result": "timeout", "ticks": MAX_CYCLES}
    except Exception as e:
        print(f"ERROR: {e}")
        return {"result": "error", "ticks": 0, "error": str(e)}

    # Parse result
    winner = None
    ticks = MAX_CYCLES

    winner_match = re.search(r'WINNER:\s*(-?\d+)', output)
    if winner_match:
        winner = int(winner_match.group(1))

    tick_match = re.search(r'FINAL_TICK:\s*(\d+)', output)
    if tick_match:
        ticks = int(tick_match.group(1))

    if winner is None:
        if "Player 0 wins" in output:
            winner = 0
        elif "Player 1 wins" in output:
            winner = 1

    if winner == 0:
        print(f"WIN ({ticks} ticks)")
        return {"result": "win", "ticks": ticks}
    elif winner == 1:
        print(f"LOSS ({ticks} ticks)")
        return {"result": "loss", "ticks": ticks}
    else:
        print(f"DRAW ({ticks} ticks)")
        return {"result": "draw", "ticks": ticks}


def calculate_game_score(result, ticks):
    """
    Calculate score for a single game against a reference AI.

    Returns a value 0.0 - 1.0:
      Win:  1.0 + efficiency bonus (max 1.2)
      Draw: 0.5
      Loss: 0.0
      Timeout: 0.0
    """
    if result == "win":
        base = 1.0
        # Efficiency bonus for fast wins
        if ticks < MAX_CYCLES * 0.5:
            bonus = 0.2
        elif ticks < MAX_CYCLES * 0.75:
            bonus = 0.1
        else:
            bonus = 0.0
        return min(1.2, base + bonus)
    elif result == "draw":
        return 0.5
    else:  # loss, timeout, error
        return 0.0


def calculate_benchmark_score(llm_results):
    """
    Calculate the final benchmark score (0-100) for an LLM.

    Score = Sum of (game_score × anchor_weight) for each anchor

    This score is COMPARABLE across tournaments because it's based
    only on performance against fixed reference AIs.
    """
    total_score = 0.0

    for anchor_class, anchor_info in ANCHORS.items():
        if anchor_class in llm_results:
            games = llm_results[anchor_class]
            if games:
                # Average score across games against this anchor
                avg_score = sum(
                    calculate_game_score(g["result"], g["ticks"])
                    for g in games
                ) / len(games)

                # Weight by anchor difficulty
                weighted = avg_score * anchor_info["weight"]
                total_score += weighted

    return round(total_score, 1)


def _score_to_grade(score):
    """Convert a benchmark score (0-100) to a letter grade."""
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


def _opponent_breakdown(reference_games):
    """Build per-opponent stats dict from reference game results."""
    breakdown = {}
    for anchor_class, games in reference_games.items():
        if anchor_class not in ANCHORS:
            continue
        anchor_info = ANCHORS[anchor_class]
        wins = sum(1 for g in games if g["result"] == "win")
        draws = sum(1 for g in games if g["result"] == "draw")
        losses = sum(1 for g in games if g["result"] not in ("win", "draw"))
        avg_score = sum(
            calculate_game_score(g["result"], g["ticks"]) for g in games
        ) / len(games) if games else 0.0
        weighted_pts = round(avg_score * anchor_info["weight"], 1)
        breakdown[anchor_info["name"]] = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "avg_game_score": round(avg_score, 3),
            "weighted_points": weighted_pts
        }
    return breakdown


def generate_results_markdown(benchmark_scores, all_results, eliminated_at,
                              h2h_results, games_per_pair, timestamp):
    """Generate RESULTS.md with per-opponent columns in the leaderboard."""
    sorted_scores = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)

    # Build anchor name list in order
    anchor_names = [info["name"] for info in ANCHORS.values()]

    lines = []
    lines.append("# MicroRTS LLM Benchmark Results")
    lines.append("")
    lines.append(f"## Latest Benchmark: {timestamp[:10]}")
    lines.append("")
    lines.append("### Configuration")
    lines.append("")
    lines.append("| Setting | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Map | `{MAP}` |")
    lines.append(f"| Max Cycles | {MAX_CYCLES} |")
    lines.append(f"| Games per Matchup | {games_per_pair} |")
    lines.append(f"| Arena Version | 2.0 |")
    lines.append(f"| Format | Single-elimination |")
    lines.append("")
    lines.append("### Scoring System")
    lines.append("")
    lines.append("Single-elimination: LLMs must **win** to advance. Draw/loss/timeout = eliminated.")
    lines.append("")
    lines.append("| # | Reference AI | Tier | Max Points |")
    lines.append("|---|--------------|------|------------|")
    for i, (anchor_class, info) in enumerate(ANCHORS.items(), 1):
        lines.append(f"| {i} | {info['name']} | {info['tier']} | {info['weight']} |")
    lines.append("")
    lines.append("**Per-game scoring:**")
    lines.append("- Win: 1.0 points (+ 0.2 bonus for fast wins)")
    lines.append("- Draw: 0.5 points")
    lines.append("- Loss/Timeout: 0.0 points")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Leaderboard table with per-opponent columns
    lines.append("## Leaderboard")
    lines.append("")
    header = "| Rank | Model | Score | Grade |"
    sep = "|------|-------|-------|-------|"
    for name in anchor_names:
        header += f" {name} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for rank, (llm_name, score) in enumerate(sorted_scores, 1):
        grade = _score_to_grade(score)
        row = f"| {rank} | {llm_name} | **{score}** | {grade} |"
        # Per-opponent result summary
        results = all_results.get(llm_name, {}).get("reference_games", {})
        past_elimination = False
        for anchor_class in ANCHORS:
            anchor_name = ANCHORS[anchor_class]["name"]
            games = results.get(anchor_class, [])
            if past_elimination:
                row += " -- |"
            elif games:
                wins = sum(1 for g in games if g["result"] == "win")
                draws = sum(1 for g in games if g["result"] == "draw")
                losses = len(games) - wins - draws
                row += f" {wins}W/{draws}D/{losses}L |"
                if not any(g["result"] == "win" for g in games):
                    past_elimination = True
            else:
                row += " -- |"
                past_elimination = True
        lines.append(row)

    lines.append("")
    lines.append("---")
    lines.append("")

    # Detailed per-model breakdown
    lines.append("## Detailed Results")
    lines.append("")

    for llm_name, score in sorted_scores:
        elim = eliminated_at.get(llm_name)
        elim_note = f" -- eliminated at {elim}" if elim else " -- cleared all"
        lines.append(f"### {llm_name} (Score: {score}{elim_note})")
        lines.append("")
        lines.append("| Opponent | Tier | Result | Ticks | Game Score | Weight | Points |")
        lines.append("|----------|------|--------|-------|------------|--------|--------|")

        results = all_results.get(llm_name, {}).get("reference_games", {})
        for anchor_class, anchor_info in ANCHORS.items():
            games = results.get(anchor_class, [])
            if not games:
                lines.append(
                    f"| {anchor_info['name']} | {anchor_info['tier']} | "
                    f"-- | -- | -- | {anchor_info['weight']} | 0.0 |"
                )
                continue
            for g in games:
                result_str = g["result"].upper()
                ticks = g["ticks"]
                game_score = calculate_game_score(g["result"], g["ticks"])
                weighted = game_score * anchor_info["weight"]
                lines.append(
                    f"| {anchor_info['name']} | {anchor_info['tier']} | "
                    f"{result_str} | {ticks} | {game_score:.2f} | "
                    f"{anchor_info['weight']} | {weighted:.1f} |"
                )
        lines.append("")

    lines.append("---")
    lines.append("")

    # Head-to-head
    if h2h_results:
        lines.append("## Head-to-Head Results (Supplementary)")
        lines.append("")
        lines.append("These games do not affect benchmark scores but show relative performance between LLMs.")
        lines.append("")
        lines.append("| Player 1 | Player 2 | Result | Ticks |")
        lines.append("|----------|----------|--------|-------|")
        for h in h2h_results:
            lines.append(f"| {h['player0']} | {h['player1']} | {h['result'].upper()} | {h['ticks']} |")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Grade scale
    lines.append("## Grade Scale")
    lines.append("")
    lines.append("| Grade | Score Range | Description |")
    lines.append("|-------|-------------|-------------|")
    lines.append("| A+ | 90-100 | Excellent - beats hard AIs consistently |")
    lines.append("| A | 80-89 | Very Good - competes with hard AIs |")
    lines.append("| B | 70-79 | Good - beats medium, challenges hard |")
    lines.append("| C | 60-69 | Average - beats easy and some medium |")
    lines.append("| D | 40-59 | Below Average - draws common |")
    lines.append("| F | 0-39 | Failing - losses/timeouts |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by benchmark_arena.py v2.0 (single-elimination) on {timestamp[:10]}*")
    lines.append("")

    return "\n".join(lines)


def run_tournament(games_per_pair=1):
    """Run single-elimination benchmark tournament."""
    print("=" * 60)
    print("MicroRTS LLM Benchmark Arena v2.0 (Single-Elimination)")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Map: {MAP}")
    print(f"Max Cycles: {MAX_CYCLES}")
    print(f"Games per matchup: {games_per_pair}")
    print()
    print("Single-elimination: must WIN to advance (draw/loss/timeout = out)")
    print(f"  Elimination order ({len(ANCHORS)} opponents, 100 pts total):")
    for i, (anchor_class, info) in enumerate(ANCHORS.items(), 1):
        print(f"    {i}. {info['name']} ({info['tier']}): {info['weight']} pts max")
    print()

    # Results storage
    all_results = {}
    benchmark_scores = {}
    eliminated_at = {}  # llm_name -> anchor_name where eliminated (or None)

    # Phase 1: Each LLM vs Reference AIs (single-elimination)
    print("BENCHMARK GAMES (single-elimination)")
    print("-" * 40)

    for llm_class, llm_info in LLMS.items():
        llm_name = llm_info["display"]
        all_results[llm_name] = {"reference_games": {}, "llm_games": []}
        eliminated_at[llm_name] = None
        eliminated = False

        print(f"\n{llm_name}:")

        for anchor_class, anchor_info in ANCHORS.items():
            if eliminated:
                # Skip remaining opponents after elimination
                break

            all_results[llm_name]["reference_games"][anchor_class] = []

            for game_num in range(games_per_pair):
                result = run_game(llm_class, anchor_class)
                result["game_num"] = game_num + 1
                result["opponent"] = anchor_info["name"]
                all_results[llm_name]["reference_games"][anchor_class].append(result)

            # Check if LLM won at least one game against this opponent
            games = all_results[llm_name]["reference_games"][anchor_class]
            has_win = any(g["result"] == "win" for g in games)
            if not has_win:
                eliminated = True
                eliminated_at[llm_name] = anchor_info["name"]
                print(f"  ** ELIMINATED at {anchor_info['name']} (no win) **")

        if not eliminated:
            print(f"  ** CLEARED ALL OPPONENTS **")

        # Calculate benchmark score for this LLM
        benchmark_scores[llm_name] = calculate_benchmark_score(
            all_results[llm_name]["reference_games"]
        )

    print()

    # Phase 2: LLM vs LLM (supplementary, not part of benchmark score)
    print("LLM vs LLM GAMES (supplementary)")
    print("-" * 40)

    llm_list = list(LLMS.keys())
    h2h_results = []

    for i, llm1_class in enumerate(llm_list):
        for llm2_class in llm_list[i+1:]:
            llm1_name = LLMS[llm1_class]["display"]
            llm2_name = LLMS[llm2_class]["display"]

            for game_num in range(games_per_pair):
                result = run_game(llm1_class, llm2_class)
                result["player0"] = llm1_name
                result["player1"] = llm2_name
                h2h_results.append(result)

    print()

    # Display Results
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print()

    # Sort by score
    sorted_scores = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)

    print("BENCHMARK SCORES (0-100, single-elimination)")
    print("-" * 60)
    print(f"{'Rank':<6}{'Model':<25}{'Score':<10}{'Grade':<8}{'Eliminated at'}")
    print("-" * 60)

    for rank, (llm_name, score) in enumerate(sorted_scores, 1):
        grade = _score_to_grade(score)
        elim = eliminated_at.get(llm_name)
        elim_str = elim if elim else "-- cleared all --"
        print(f"{rank:<6}{llm_name:<25}{score:<10}{grade:<8}{elim_str}")

    print()

    # Detailed breakdown
    print("DETAILED BREAKDOWN")
    print("-" * 50)

    for llm_name, results in all_results.items():
        elim = eliminated_at.get(llm_name)
        elim_note = f" (eliminated at {elim})" if elim else " (cleared all)"
        print(f"\n{llm_name}{elim_note}:")
        for anchor_class, games in results["reference_games"].items():
            anchor_name = ANCHORS[anchor_class]["name"]
            weight = ANCHORS[anchor_class]["weight"]

            for g in games:
                result_str = g["result"].upper()
                ticks = g["ticks"]
                game_score = calculate_game_score(g["result"], g["ticks"])
                weighted = game_score * weight
                print(f"  vs {anchor_name}: {result_str} ({ticks} ticks) -> {game_score:.2f} x {weight} = {weighted:.1f} pts")

    print()

    # Head-to-head summary
    print("HEAD-TO-HEAD SUMMARY (not part of benchmark score)")
    print("-" * 50)

    wins = {LLMS[c]["display"]: 0 for c in LLMS}
    losses = {LLMS[c]["display"]: 0 for c in LLMS}

    for h in h2h_results:
        if h["result"] == "win":
            wins[h["player0"]] += 1
            losses[h["player1"]] += 1
        elif h["result"] == "loss":
            losses[h["player0"]] += 1
            wins[h["player1"]] += 1

    for llm_name in wins:
        print(f"  {llm_name}: {wins[llm_name]}W - {losses[llm_name]}L")

    print()

    # Save results
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M')

    # Build per-opponent breakdown for each LLM
    opponent_breakdowns = {}
    for llm_name, results in all_results.items():
        opponent_breakdowns[llm_name] = _opponent_breakdown(
            results["reference_games"]
        )

    results_file = f"{RESULTS_DIR}/benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump({
            "version": "2.0",
            "format": "single-elimination",
            "date": now.isoformat(),
            "config": {
                "map": MAP,
                "max_cycles": MAX_CYCLES,
                "games_per_matchup": games_per_pair
            },
            "anchors": {
                cls: {"name": info["name"], "weight": info["weight"], "tier": info["tier"]}
                for cls, info in ANCHORS.items()
            },
            "benchmark_scores": benchmark_scores,
            "eliminated_at": eliminated_at,
            "opponent_breakdown": opponent_breakdowns,
            "detailed_results": all_results,
            "head_to_head": h2h_results
        }, f, indent=2)

    print(f"Results saved to {results_file}")

    # Also append to historical leaderboard
    leaderboard_file = f"{RESULTS_DIR}/leaderboard.json"
    if Path(leaderboard_file).exists():
        with open(leaderboard_file, 'r') as f:
            leaderboard = json.load(f)
    else:
        leaderboard = {"entries": []}

    # Build agent_type lookup by display name
    agent_types = {info["display"]: info["agent_type"] for info in LLMS.values()}

    for llm_name, score in benchmark_scores.items():
        leaderboard["entries"].append({
            "model": llm_name,
            "agent_type": agent_types.get(llm_name, "unknown"),
            "score": score,
            "version": "2.0",
            "date": now.isoformat(),
            "map": MAP,
            "games_per_matchup": games_per_pair,
            "opponents": opponent_breakdowns.get(llm_name, {})
        })

    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)

    print(f"Leaderboard updated: {leaderboard_file}")

    # Generate RESULTS.md
    results_md = generate_results_markdown(
        benchmark_scores, all_results, eliminated_at,
        h2h_results, games_per_pair, now.isoformat()
    )
    results_md_file = f"{RESULTS_DIR}/RESULTS.md"
    with open(results_md_file, 'w') as f:
        f.write(results_md)

    print(f"Results markdown: {results_md_file}")

    return benchmark_scores


if __name__ == "__main__":
    games = 1
    if len(sys.argv) > 1 and sys.argv[1] == "--games":
        games = int(sys.argv[2])

    run_tournament(games_per_pair=games)