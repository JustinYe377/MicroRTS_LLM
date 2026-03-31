# Parker-Riggs AdaptiveRushBot

This submission contains `AdaptiveRushBot`, a hybrid MicroRTS agent that:
- Uses an LLM (via Ollama) to choose between `WorkerRush` and `LightRush`
- Falls back to built-in heuristics if the LLM is unavailable or times out

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

## Notes

- Package name uses underscore (`parker_riggs`) to match Java naming rules.
- The bot queries the LLM periodically (not every tick) for performance.
- If Ollama is not running, the bot still works using heuristic strategy switching.
- `gui.frontend.FrontEnd` has a fixed AI dropdown list and does not auto-list submission classes.