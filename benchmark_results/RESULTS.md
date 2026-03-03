# MicroRTS LLM Benchmark Results

## Latest Benchmark: 2026-02-17

### Configuration

| Setting | Value |
|---------|-------|
| Map | `maps/8x8/basesWorkers8x8.xml` |
| Max Cycles | 3000 |
| Games per Matchup | 1 |
| Arena Version | 2.0 |
| Format | Single-elimination |

### Scoring System

Single-elimination: LLMs must **win** to advance. Draw/loss/timeout = eliminated.

| # | Reference AI | Tier | Max Points |
|---|--------------|------|------------|
| 1 | RandomBiasedAI | easy | 10 |
| 2 | HeavyRush | medium-hard | 20 |
| 3 | LightRush | medium | 15 |
| 4 | WorkerRush | medium | 15 |
| 5 | Tiamat | hard | 20 |
| 6 | CoacAI | hard | 20 |

**Per-game scoring:**
- Win: 1.0 points (+ 0.2 bonus for fast wins)
- Draw: 0.5 points
- Loss/Timeout: 0.0 points

---

## Leaderboard

| Rank | Model | Score | Grade | RandomBiasedAI | HeavyRush | LightRush | WorkerRush | Tiamat | CoacAI |
|------|-------|-------|-------|------|------|------|------|------|------|
| 1 | llama3.1:8b (Hybrid) | **72.0** | B | 1W/0D/0L | 1W/0D/0L | 1W/0D/0L | 1W/0D/0L | 0W/0D/1L | -- |
| 2 | llama3.1:8b (Search+LLM) | **54.0** | D | 1W/0D/0L | 1W/0D/0L | 1W/0D/0L | 0W/0D/1L | -- | -- |
| 3 | deepseek-chat (PureLLM-Cloud) | **0.0** | F | 0W/0D/1L | -- | -- | -- | -- | -- |
| 4 | llama3.1:8b (PureLLM) | **0.0** | F | 0W/0D/1L | -- | -- | -- | -- | -- |
| 5 | qwen3:4b (PureLLM) | **0.0** | F | 0W/0D/1L | -- | -- | -- | -- | -- |

---

## Detailed Results

### llama3.1:8b (Hybrid) (Score: 72.0 -- eliminated at Tiamat)

| Opponent | Tier | Result | Ticks | Game Score | Weight | Points |
|----------|------|--------|-------|------------|--------|--------|
| RandomBiasedAI | easy | WIN | 370 | 1.20 | 10 | 12.0 |
| HeavyRush | medium-hard | WIN | 195 | 1.20 | 20 | 24.0 |
| LightRush | medium | WIN | 195 | 1.20 | 15 | 18.0 |
| WorkerRush | medium | WIN | 1365 | 1.20 | 15 | 18.0 |
| Tiamat | hard | LOSS | 1240 | 0.00 | 20 | 0.0 |
| CoacAI | hard | -- | -- | -- | 20 | 0.0 |

### llama3.1:8b (Search+LLM) (Score: 54.0 -- eliminated at WorkerRush)

| Opponent | Tier | Result | Ticks | Game Score | Weight | Points |
|----------|------|--------|-------|------------|--------|--------|
| RandomBiasedAI | easy | WIN | 680 | 1.20 | 10 | 12.0 |
| HeavyRush | medium-hard | WIN | 270 | 1.20 | 20 | 24.0 |
| LightRush | medium | WIN | 578 | 1.20 | 15 | 18.0 |
| WorkerRush | medium | LOSS | 190 | 0.00 | 15 | 0.0 |
| Tiamat | hard | -- | -- | -- | 20 | 0.0 |
| CoacAI | hard | -- | -- | -- | 20 | 0.0 |

### deepseek-chat (PureLLM-Cloud) (Score: 0.0 -- eliminated at RandomBiasedAI)

| Opponent | Tier | Result | Ticks | Game Score | Weight | Points |
|----------|------|--------|-------|------------|--------|--------|
| RandomBiasedAI | easy | LOSS | 1225 | 0.00 | 10 | 0.0 |
| HeavyRush | medium-hard | -- | -- | -- | 20 | 0.0 |
| LightRush | medium | -- | -- | -- | 15 | 0.0 |
| WorkerRush | medium | -- | -- | -- | 15 | 0.0 |
| Tiamat | hard | -- | -- | -- | 20 | 0.0 |
| CoacAI | hard | -- | -- | -- | 20 | 0.0 |

### llama3.1:8b (PureLLM) (Score: 0.0 -- eliminated at RandomBiasedAI)

| Opponent | Tier | Result | Ticks | Game Score | Weight | Points |
|----------|------|--------|-------|------------|--------|--------|
| RandomBiasedAI | easy | LOSS | 875 | 0.00 | 10 | 0.0 |
| HeavyRush | medium-hard | -- | -- | -- | 20 | 0.0 |
| LightRush | medium | -- | -- | -- | 15 | 0.0 |
| WorkerRush | medium | -- | -- | -- | 15 | 0.0 |
| Tiamat | hard | -- | -- | -- | 20 | 0.0 |
| CoacAI | hard | -- | -- | -- | 20 | 0.0 |

### qwen3:4b (PureLLM) (Score: 0.0 -- eliminated at RandomBiasedAI)

| Opponent | Tier | Result | Ticks | Game Score | Weight | Points |
|----------|------|--------|-------|------------|--------|--------|
| RandomBiasedAI | easy | LOSS | 860 | 0.00 | 10 | 0.0 |
| HeavyRush | medium-hard | -- | -- | -- | 20 | 0.0 |
| LightRush | medium | -- | -- | -- | 15 | 0.0 |
| WorkerRush | medium | -- | -- | -- | 15 | 0.0 |
| Tiamat | hard | -- | -- | -- | 20 | 0.0 |
| CoacAI | hard | -- | -- | -- | 20 | 0.0 |

---

## Head-to-Head Results (Supplementary)

These games do not affect benchmark scores but show relative performance between LLMs.

| Player 1 | Player 2 | Result | Ticks |
|----------|----------|--------|-------|
| deepseek-chat (PureLLM-Cloud) | llama3.1:8b (PureLLM) | WIN | 380 |
| deepseek-chat (PureLLM-Cloud) | qwen3:4b (PureLLM) | WIN | 480 |
| deepseek-chat (PureLLM-Cloud) | llama3.1:8b (Hybrid) | LOSS | 280 |
| deepseek-chat (PureLLM-Cloud) | llama3.1:8b (Search+LLM) | LOSS | 430 |
| llama3.1:8b (PureLLM) | qwen3:4b (PureLLM) | DRAW | 3000 |
| llama3.1:8b (PureLLM) | llama3.1:8b (Hybrid) | LOSS | 185 |
| llama3.1:8b (PureLLM) | llama3.1:8b (Search+LLM) | LOSS | 345 |
| qwen3:4b (PureLLM) | llama3.1:8b (Hybrid) | LOSS | 185 |
| qwen3:4b (PureLLM) | llama3.1:8b (Search+LLM) | LOSS | 275 |
| llama3.1:8b (Hybrid) | llama3.1:8b (Search+LLM) | LOSS | 555 |

---

## Grade Scale

| Grade | Score Range | Description |
|-------|-------------|-------------|
| A+ | 90-100 | Excellent - beats hard AIs consistently |
| A | 80-89 | Very Good - competes with hard AIs |
| B | 70-79 | Good - beats medium, challenges hard |
| C | 60-69 | Average - beats easy and some medium |
| D | 40-59 | Below Average - draws common |
| F | 0-39 | Failing - losses/timeouts |

---

*Generated by benchmark_arena.py v2.0 (single-elimination) on 2026-02-17*
