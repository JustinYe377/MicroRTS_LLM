# Parker-Riggs AdaptiveRushBot

`AdaptiveRushBot` is a hybrid MicroRTS agent that tries to counter whatever
the opponent is building. It picks between three scripted rush strategies each
game and updates that choice as it gains more information about the enemy army.

The three strategies and the reasoning behind each:

- **WorkerRush** - floods the map with worker units early, applying fast swarming
  pressure before the opponent can build dedicated military.
- **LightRush** - trains cheap, fast Light units from a Barracks. Light units
  destroy Workers very efficiently and are a strong general opener.
- **HeavyRush** - trains Heavy units instead. They are slower to build but win
  direct combat against Light units, making this the right counter when the
  opponent goes LightRush.

The counter triangle is: Light beats Worker, Heavy beats Light, Worker can
sometimes pressure Heavy through sheer numbers early in the game.

### Decision making

Strategy selection works in two layers:

1. **LLM layer** - every 25 ticks the bot sends a compact game state summary to
   a local Ollama instance and asks it to pick one of the three strategies. The
   prompt includes the current tick, map dimensions, your own unit counts, and a
   per-type breakdown of visible enemy units so the model has enough context to
   reason about counters.
2. **Heuristic fallback** - if Ollama is not running, times out, or returns
   something unparseable, a hand-written rule set takes over. It checks the
   enemy composition in priority order (Heavy units visible? Go HeavyRush.
   Only workers visible? Go LightRush. Already committed to a military type?
   Stay the course.) and always produces a sensible answer.

## Files

- `AdaptiveRushBot.java` - main agent
- `metadata.json` - submission metadata

## Prerequisites

- Java JDK 17+
- Ollama installed locally: <https://ollama.ai/>
- MicroRTS repository root as working directory

## 1) Start Ollama and load a model

In a separate terminal:

```bash
ollama serve
```

Then pull the model used by this submission (default):

```bash
ollama pull llama3.1:8b
```

Optional environment variables (defaults shown):

```bash
export OLLAMA_ENDPOINT="http://localhost:11434/api/generate"
export OLLAMA_MODEL="llama3.1:8b"
```

## 2) Compile MicroRTS (full project)

From the repository root (`MicroRTS`):

```bash
find src -name '*.java' > sources.list
javac -cp "lib/*:bin" -d bin @sources.list
```

## 3) Compile this submission agent

From the repository root:

```bash
javac -cp "lib/*:bin:src" -d bin submissions/parker-riggs/AdaptiveRushBot.java
```

## 4) Configure game to use this agent

Edit `resources/config.properties` and set one side to:

```properties
AI1=ai.abstraction.submissions.parker_riggs.AdaptiveRushBot
```

Example opponent:

```properties
AI2=ai.abstraction.LightRush
```

Pick a map in the same file, for example:

```properties
map_location=maps/8x8/basesWorkers8x8.xml
```

## 5) Run games

From repository root:

```bash
javac -cp "lib/*:bin:src" -d bin submissions/parker-riggs/AdaptiveRushBot.java
java -cp "lib/*:bin" rts.MicroRTS -f resources/config.properties
```

If you want to watch the game window while still using `config.properties`, set:

```properties
headless=false
```

Then run the same command:

```bash
java -cp "lib/*:bin" rts.MicroRTS -f resources/config.properties
```

## 6) Where results are stored

Direct run (`rts.MicroRTS -f resources/config.properties`):
- Primary output is printed to the terminal where you launched the game.
- No automatic `results/` file is created by default for this single-run command.

GUI traces (optional):
- If you run the FrontEnd and check `Save Trace`, replay files are written as `trace1.xml`, `trace2.xml`, etc.
- These trace files are saved in the current working directory (typically the repository root).

Scripted loop runs:
- `./RunLoop.sh` writes per-run logs to `logs/run_YYYY-MM-DD_HH-MM-SS.log`.

Experiment/benchmark artifacts in this repo:
- Curated experiment folders are under `results/`.
- Benchmark JSON and leaderboard files are under `benchmark_results/`.

## How the heuristic works

When the LLM is unavailable the bot evaluates these rules in order and uses the
first one that fires:

1. Already have 2+ Heavy units built -- keep producing Heavy to avoid wasting ramp time.
2. Already have 2+ Light units built -- keep producing Light for the same reason.
3. Enemy Light units visible -- switch to HeavyRush, Heavy wins that fight directly.
4. Enemy Heavy units visible -- use LightRush, it applies pressure faster than workers.
5. Enemy has only workers with no military -- use LightRush, Light shreds workers cheaply.
6. Enemy attackers are within 8 tiles of our base -- use LightRush for the fastest response.
7. We have 2+ workers built -- economy is set up, time to go military.
8. Game time has passed the map-size threshold (150 ticks on 8x8, 300 on larger) -- commit.
9. Default to LightRush as a safe general opener against unknown opponents.

The time threshold is halved on 8x8 maps because those games typically resolve
in well under 500 ticks, so the transition window needs to be tighter.

## Notes

- Package name uses underscore (`parker_riggs`) to match Java naming rules.
- The LLM is only queried once every 25 ticks to keep the game loop responsive.
- If Ollama is not running the bot works entirely on the heuristic described above.
- The end-of-game log line shows what percentage of decisions came from the LLM
  vs the heuristic, which is useful for verifying Ollama is actually being used.
- `gui.frontend.FrontEnd` has a fixed AI dropdown and does not auto-list submission classes.