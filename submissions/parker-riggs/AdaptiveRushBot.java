package ai.abstraction.submissions.parker_riggs;

import ai.abstraction.HeavyRush;
import ai.abstraction.LightRush;
import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.List;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Locale;
import rts.GameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

/**
 * AdaptiveRushBot picks between three scripted rush strategies each game and
 * tries to counter whatever the opponent is building. The three options are
 * WorkerRush (fast swarm of worker units), LightRush (cheap fast combat units),
 * and HeavyRush (slower but tankier units that beat Light in direct fights).
 *
 * Strategy selection works in two layers. First it asks a local LLM (Ollama)
 * for a recommendation based on a compact summary of the game state. If the LLM
 * is unavailable, times out, or gives an unrecognizable answer, the bot falls
 * back to a hand-written counter-strategy heuristic so it always has something
 * reasonable to do regardless of whether Ollama is running.
 *
 * The general counter triangle is: Light beats Worker, Heavy beats Light,
 * Worker can sometimes pressure Heavy through sheer numbers early on.
 */
public class AdaptiveRushBot extends AI {

    // The three strategies the bot can pick between. Each maps to a different
    // scripted AI delegate that handles the actual unit micro for that style.
    private enum Strategy { WORKER_RUSH, LIGHT_RUSH, HEAVY_RUSH }

    // How many ticks into the game before we default to a military strategy
    // if nothing else has triggered a switch yet. On 8x8 maps this is used as-is;
    // on larger maps it is doubled at runtime since games take longer to develop.
    private static final int LIGHT_RUSH_TIME_BASE = 150;

    // Once we have at least this many workers we consider the early economy done
    // and start transitioning toward military production.
    private static final int WORKER_COUNT_THRESHOLD = 2;

    // If we already have this many combat units of one type we keep producing
    // that same type rather than switching mid-game and wasting the barracks ramp.
    private static final int MILITARY_COUNT_THRESHOLD = 2;

    // If any enemy attacker gets within this Manhattan distance of one of our
    // bases we treat it as an immediate threat and pick a fast counter response.
    private static final int ENEMY_PRESSURE_DISTANCE = 8;

    // We only ask the LLM for a new decision every this many game ticks.
    // Calling it every tick would be far too slow even on a fast local model.
    private static final int LLM_CONSULT_COOLDOWN = 25;

    // We only write a strategy log line every this many ticks to keep the
    // output readable without flooding the console on every frame.
    private static final int DECISION_LOG_COOLDOWN = 50;

    // Hard cap on how long we wait for a single LLM HTTP response. If the local
    // Ollama instance is slow or overloaded we fall back to heuristics rather
    // than blocking the game loop and causing the bot to time out.
    private static final Duration LLM_TIMEOUT = Duration.ofMillis(800);

    // Unit type table passed in at construction, needed by the delegate AIs.
    private final UnitTypeTable unitTypeTable;

    // The three delegate scripted AIs. We hand off to whichever one we picked
    // and let it handle all the unit-level micro for that strategy.
    private final WorkerRush workerRush;
    private final LightRush lightRush;
    private final HeavyRush heavyRush;

    // HTTP client used for Ollama requests. Built once and reused across ticks.
    // The endpoint and model name can be overridden with the OLLAMA_ENDPOINT
    // and OLLAMA_MODEL environment variables at runtime.
    private final HttpClient httpClient;
    private final String llmEndpoint;
    private final String llmModel;

    // The game tick when we last got a usable answer back from the LLM.
    // Initialized to a large negative number so the first tick always tries a query.
    private int lastLLMConsultTime = -1000;

    // The last tick we printed a decision log line. Separate from LLM consult time
    // so logging stays readable even when we are running on cached LLM decisions.
    private int lastDecisionLogTime = -1000;

    // Running counters used in the end-of-game summary to show how much of the
    // game was actually driven by the LLM vs the fallback heuristic.
    private int llmConsultAttempts = 0;
    private int llmConsultSuccesses = 0;
    private int llmDrivenDecisionCount = 0;
    private int heuristicDecisionCount = 0;

    // The strategy we are currently running. This gets updated each time we get
    // a fresh LLM answer or run the heuristic, and stays fixed between polls so
    // the bot does not jitter. We default to LIGHT_RUSH because it beats a pure
    // WorkerRush opponent from tick zero with no information needed.
    private Strategy cachedStrategy = Strategy.LIGHT_RUSH;

    /**
     * Constructs a new AdaptiveRushBot. MicroRTS instantiates agents via
     * reflection using this exact constructor signature, so the UnitTypeTable
     * parameter is required even though we mostly pass it straight through to
     * the delegate AIs.
     */
    public AdaptiveRushBot(UnitTypeTable unitTypeTable) {
        this.unitTypeTable = unitTypeTable;
        this.workerRush = new WorkerRush(unitTypeTable);
        this.lightRush = new LightRush(unitTypeTable);
        this.heavyRush = new HeavyRush(unitTypeTable);
        this.httpClient = HttpClient.newBuilder().connectTimeout(LLM_TIMEOUT).build();
        this.llmEndpoint = resolveEndpoint();
        this.llmModel = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
        System.out.println("[AdaptiveRushBot] LLM endpoint=" + llmEndpoint + " model=" + llmModel);
    }

    /**
     * Called by the framework between matches to clear all per-game state.
     * We reset the delegate AIs and all our counters so each game starts fresh.
     */
    @Override
    public void reset() {
        workerRush.reset();
        lightRush.reset();
        heavyRush.reset();
        lastLLMConsultTime = -1000;
        lastDecisionLogTime = -1000;
        llmConsultAttempts = 0;
        llmConsultSuccesses = 0;
        llmDrivenDecisionCount = 0;
        heuristicDecisionCount = 0;
        cachedStrategy = Strategy.LIGHT_RUSH;
    }

    @Override
    public AI clone() {
        return new AdaptiveRushBot(unitTypeTable);
    }

    /**
     * Called every game tick by the MicroRTS framework. If this player has any
     * units that can act right now we pick a strategy and hand off to the
     * appropriate delegate AI to generate the actual unit orders. If no unit
     * can act we return an empty action set which the framework ignores.
     */
    @Override
    public PlayerAction getAction(int player, GameState gameState) throws Exception {
        if (gameState.canExecuteAnyAction(player)) {
            Strategy strategy = decideStrategy(player, gameState);
            switch (strategy) {
                case HEAVY_RUSH: return heavyRush.getAction(player, gameState);
                case WORKER_RUSH: return workerRush.getAction(player, gameState);
                default: return lightRush.getAction(player, gameState);
            }
        }
        return new PlayerAction();
    }

    /**
     * Picks which strategy to run this tick. The logic has three tiers:
     *
     * First, if we got an LLM answer recently and the cooldown has not expired,
     * we reuse the cached result rather than hammering the model every tick.
     *
     * Second, when the cooldown is up, we try asking the LLM for a fresh pick.
     * If it responds with something we can parse we update the cache and use it.
     *
     * Third, if the LLM is unavailable or times out we fall through to the
     * hand-written heuristic which uses the visible enemy composition to decide.
     */
    private Strategy decideStrategy(int player, GameState gameState) {
        // Still within the cooldown window from the last successful LLM call,
        // so just reuse the cached strategy rather than querying again.
        if (lastLLMConsultTime >= 0 && gameState.getTime() - lastLLMConsultTime < LLM_CONSULT_COOLDOWN) {
            llmDrivenDecisionCount++;
            maybeLogDecision(gameState.getTime(), "LLM_CACHED", cachedStrategy);
            return cachedStrategy;
        }

        if (gameState.getTime() - lastLLMConsultTime >= LLM_CONSULT_COOLDOWN) {
            Strategy llmDecision = consultLLM(player, gameState);
            if (llmDecision != null) {
                // Got a clean answer from the LLM, update the cache and timestamp.
                cachedStrategy = llmDecision;
                lastLLMConsultTime = gameState.getTime();
                llmDrivenDecisionCount++;
                maybeLogDecision(gameState.getTime(), "LLM", cachedStrategy);
                return cachedStrategy;
            }
        }

        // LLM was unavailable or gave an unparseable response, use the heuristic.
        cachedStrategy = chooseStrategyHeuristic(player, gameState);
        heuristicDecisionCount++;
        maybeLogDecision(gameState.getTime(), "HEURISTIC", cachedStrategy);
        return cachedStrategy;
    }

    /**
     * Asks the local Ollama instance to recommend a strategy. We build a short
     * text prompt describing the current game state and ask for exactly one of
     * the three strategy tokens back. The response is parsed permissively so
     * minor formatting differences from the model do not cause false negatives.
     *
     * Returns the parsed Strategy on success, or null if anything goes wrong
     * (network error, timeout, HTTP error, unrecognized response text). The
     * caller is responsible for falling back to the heuristic on a null return.
     */
    private Strategy consultLLM(int player, GameState gameState) {
        try {
            llmConsultAttempts++;
            StrategySummary summary = summarizeState(player, gameState);
            String prompt = buildPrompt(summary, gameState.getTime());

            // Build the JSON body manually to avoid pulling in a JSON library.
            // temperature=0 keeps the output deterministic so parsing is reliable.
            String body = "{\"model\":\"" + escapeJson(llmModel) + "\","
                    + "\"prompt\":\"" + escapeJson(prompt) + "\","
                    + "\"stream\":false,\"options\":{\"temperature\":0}}";

            // We use a synchronous send here because we need the answer before
            // we can return the PlayerAction for this tick.
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(llmEndpoint))
                    .timeout(LLM_TIMEOUT)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                return null;
            }

            String llmText = extractResponseText(response.body());
            if (llmText == null) {
                return null;
            }

            // Uppercase everything for case-insensitive matching, then check for
            // the canonical underscore form first before falling back to bare keywords.
            // HEAVY is checked before LIGHT so a response like "HEAVY_RUSH" does not
            // accidentally match the LIGHT branch first.
            String normalized = llmText.toUpperCase(Locale.ROOT);
            if (normalized.contains("HEAVY_RUSH") || normalized.contains("HEAVYRUSH")) {
                llmConsultSuccesses++;
                return Strategy.HEAVY_RUSH;
            }
            if (normalized.contains("LIGHT_RUSH") || normalized.contains("LIGHTRUSH")) {
                llmConsultSuccesses++;
                return Strategy.LIGHT_RUSH;
            }
            if (normalized.contains("WORKER_RUSH") || normalized.contains("WORKERRUSH")) {
                llmConsultSuccesses++;
                return Strategy.WORKER_RUSH;
            }
            // Single keyword fallback for models that drop the underscore or add extra words.
            if (normalized.contains("HEAVY")) {
                llmConsultSuccesses++;
                return Strategy.HEAVY_RUSH;
            }
            if (normalized.contains("LIGHT")) {
                llmConsultSuccesses++;
                return Strategy.LIGHT_RUSH;
            }
            if (normalized.contains("WORKER")) {
                llmConsultSuccesses++;
                return Strategy.WORKER_RUSH;
            }
            // Could not extract a usable token from the response.
            return null;
        } catch (Exception ignored) {
            // Covers connection refused, read timeout, and any unexpected parse errors.
            // All of these degrade gracefully to the heuristic path in the caller.
            return null;
        }
    }

    /**
     * Walks the unit list once and collects the numbers that matter for strategy
     * selection. We count our own workers, light units, heavy units, and bases
     * separately, and we also break down the enemy army by type so we know
     * whether to expect light or heavy opposition. We also compute the Manhattan
     * distance from the closest enemy attacker to our nearest base so we can
     * detect early rushes and react defensively.
     *
     * This is called both by the heuristic and by the LLM prompt builder, so
     * keeping it cheap matters. Two linear passes over the unit list is fine.
     */
    private StrategySummary summarizeState(int player, GameState gameState) {
        PhysicalGameState physicalGameState = gameState.getPhysicalGameState();
        int workers = 0;
        int lightUnits = 0;
        int heavyUnits = 0;
        int bases = 0;
        int enemyWorkers = 0;
        int enemyLightUnits = 0;
        int enemyHeavyUnits = 0;
        int enemyOtherCombat = 0;
        List<Unit> ownBases = new ArrayList<>();
        int closestEnemyDistanceToBase = Integer.MAX_VALUE;

        // First pass: categorize every unit on the map by owner and type.
        // Player index -1 means a neutral resource tile, which we skip.
        for (Unit unit : physicalGameState.getUnits()) {
            UnitType type = unit.getType();
            String typeName = type.name;

            if (unit.getPlayer() == player) {
                if ("Worker".equals(typeName)) {
                    workers++;
                } else if ("Light".equals(typeName)) {
                    lightUnits++;
                } else if ("Heavy".equals(typeName)) {
                    heavyUnits++;
                } else if ("Base".equals(typeName)) {
                    bases++;
                    ownBases.add(unit);
                }
            } else if (unit.getPlayer() >= 0) {
                if ("Worker".equals(typeName)) {
                    enemyWorkers++;
                } else if ("Light".equals(typeName)) {
                    enemyLightUnits++;
                } else if ("Heavy".equals(typeName)) {
                    enemyHeavyUnits++;
                } else if (type.canAttack) {
                    // Catches Ranged and any other combat-capable unit types
                    // that are not one of the three main types.
                    enemyOtherCombat++;
                }
            }
        }

        // Second pass: find how close the nearest enemy combat unit is to any
        // of our bases. Manhattan distance is appropriate for a grid map and
        // much cheaper than running pathfinding here.
        for (Unit unit : physicalGameState.getUnits()) {
            if (unit.getPlayer() < 0 || unit.getPlayer() == player) {
                continue;
            }
            if (!unit.getType().canAttack) {
                continue;
            }
            for (Unit base : ownBases) {
                int distance = Math.abs(unit.getX() - base.getX()) + Math.abs(unit.getY() - base.getY());
                if (distance < closestEnemyDistanceToBase) {
                    closestEnemyDistanceToBase = distance;
                }
            }
        }

        return new StrategySummary(
                workers, lightUnits, heavyUnits, bases,
                enemyWorkers, enemyLightUnits, enemyHeavyUnits, enemyOtherCombat,
                closestEnemyDistanceToBase,
                gameState.getPlayer(player).getResources(),
                physicalGameState.getWidth(), physicalGameState.getHeight());
    }

    /**
     * Assembles the text prompt we send to the LLM. The instructions section
     * spells out the counter-strategy triangle so the model has the game theory
     * context it needs to make a good call. The state section follows as plain
     * key=value lines because that format tends to tokenize cleanly and leaves
     * less room for the model to go off on a tangent before giving a token.
     *
     * We include map size because 8x8 games resolve completely differently from
     * 16x16 games -- the optimal strategy on a tiny map often flips on a big one.
     */
    private String buildPrompt(StrategySummary s, int time) {
        int enemyMilitary = s.enemyLightUnits + s.enemyHeavyUnits + s.enemyOtherCombat;
        return "You are selecting a MicroRTS strategy. Reply with exactly one token: WORKER_RUSH, LIGHT_RUSH, or HEAVY_RUSH.\n"
                + "Counter-strategy rules:\n"
                + "- LIGHT_RUSH counters WORKER_RUSH (Light units destroy Workers cheaply).\n"
                + "- HEAVY_RUSH counters LIGHT_RUSH (Heavy units win direct combat vs Light).\n"
                + "- WORKER_RUSH is fast early pressure with no dedicated military.\n"
                + "Choose based on what the enemy is building. Prefer responding to visible enemy composition.\n"
                + "State:\n"
                + "time=" + time + "\n"
                + "map_width=" + s.mapWidth + " map_height=" + s.mapHeight + "\n"
                + "my_resources=" + s.resources + "\n"
                + "my_workers=" + s.workers + "\n"
                + "my_light_units=" + s.lightUnits + "\n"
                + "my_heavy_units=" + s.heavyUnits + "\n"
                + "my_bases=" + s.bases + "\n"
                + "enemy_workers=" + s.enemyWorkers + "\n"
                + "enemy_light_units=" + s.enemyLightUnits + "\n"
                + "enemy_heavy_units=" + s.enemyHeavyUnits + "\n"
                + "enemy_other_combat=" + s.enemyOtherCombat + "\n"
                + "enemy_military_total=" + enemyMilitary + "\n"
                + "closest_enemy_to_base_distance=" + s.closestEnemyDistanceToBase + "\n";
    }

    /**
     * Pulls the value of the "response" key out of the JSON object that Ollama
     * returns from /api/generate. We do this with a simple character scan rather
     * than importing a JSON library to keep the submission self-contained with
     * no extra dependencies.
     *
     * The expected shape is: {"response":"some text here", ...}
     * We find the key, then read characters until the closing quote, handling
     * backslash escapes along the way. Returns null if the key is not found or
     * the string is not properly terminated.
     */
    private String extractResponseText(String responseJson) {
        int key = responseJson.indexOf("\"response\":\"");
        if (key < 0) {
            return null;
        }

        int start = key + "\"response\":\"".length();
        StringBuilder result = new StringBuilder();
        boolean escaping = false;
        for (int i = start; i < responseJson.length(); i++) {
            char c = responseJson.charAt(i);

            if (escaping) {
                // Handle common escape sequences. We only really care about \n
                // since strategy tokens will not contain other escape chars.
                if (c == 'n') {
                    result.append('\n');
                } else {
                    result.append(c);
                }
                escaping = false;
                continue;
            }

            if (c == '\\') {
                escaping = true;
                continue;
            }

            if (c == '"') {
                // Reached the closing quote of the response string value.
                return result.toString();
            }

            result.append(c);
        }

        // Reached end of string without finding the closing quote.
        return null;
    }

    /**
     * Escapes a string value for safe insertion into a manually constructed
     * JSON body. We only need to handle the characters that can break basic
     * JSON string syntax since we control the surrounding structure.
     */
    private String escapeJson(String value) {
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
    }

    /**
     * Figures out which URL to hit for LLM queries. We support two env vars so
     * the setup stays compatible with different deployment styles:
     *
     * OLLAMA_ENDPOINT - full URL including path, used as-is if set.
     * OLLAMA_HOST     - just the base host/port, we append /api/generate.
     *
     * If neither is set we fall back to localhost on the default Ollama port.
     * Trailing slashes on the host value are stripped before appending the path.
     */
    private String resolveEndpoint() {
        String endpoint = System.getenv("OLLAMA_ENDPOINT");
        if (endpoint != null && !endpoint.trim().isEmpty()) {
            return endpoint.trim();
        }

        String host = System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434").trim();
        if (host.endsWith("/")) {
            host = host.substring(0, host.length() - 1);
        }
        return host + "/api/generate";
    }

    /**
     * Prints a one-line summary of the current strategy decision if enough
     * game ticks have passed since the last log line. The source field tells
     * you whether the decision came from the LLM, a cached LLM answer, or
     * the heuristic fallback, which makes it easy to spot in logs whether
     * Ollama is actually contributing during a run.
     */
    private void maybeLogDecision(int time, String source, Strategy strategy) {
        if (time - lastDecisionLogTime < DECISION_LOG_COOLDOWN) {
            return;
        }
        lastDecisionLogTime = time;
        System.out.println(
                "[AdaptiveRushBot] t=" + time
                        + " source=" + source
                        + " strategy=" + strategy
                        + " llm_success=" + llmConsultSuccesses
                        + "/" + llmConsultAttempts);
    }

    /**
     * Rule-based strategy picker used whenever the LLM is not available.
     * The rules are ordered from most specific to least specific so that
     * the most actionable information wins.
     *
     * The thinking behind each rule in order:
     *
     * 1. If we already built heavy units, keep going with HeavyRush. Switching
     *    strategies mid-game wastes the resources already spent on the barracks.
     *
     * 2. Same for light units -- if the barracks is already producing Light,
     *    we commit to that path.
     *
     * 3. If the enemy has Light units on the field, switch to HeavyRush because
     *    Heavy wins that direct combat decisively.
     *
     * 4. If the enemy has Heavy units on the field, Light is not a great answer
     *    but it is faster than Worker and at least applies pressure.
     *
     * 5. If the enemy appears to be running a pure WorkerRush with no dedicated
     *    military, LightRush shreds workers very efficiently.
     *
     * 6. If enemy attackers are already close to our base we pick LightRush
     *    for the fastest possible military response.
     *
     * 7. Once we have enough workers to sustain economy we stop spamming workers
     *    and switch to military production appropriate for the map size.
     *
     * 8. Past the time threshold for this map size we commit to a military
     *    strategy whether anything else triggered or not.
     *
     * 9. Default to LightRush. It beats WorkerRush and is a reasonable opener
     *    against any unknown opponent.
     */
    private Strategy chooseStrategyHeuristic(int player, GameState gameState) {
        StrategySummary s = summarizeState(player, gameState);

        // 8x8 maps play out much faster so we halve the time threshold.
        boolean smallMap = s.mapWidth <= 8;
        int timeThreshold = smallMap ? LIGHT_RUSH_TIME_BASE : LIGHT_RUSH_TIME_BASE * 2;

        // Rules 1 and 2: stay committed to whatever military type we started building.
        if (s.heavyUnits >= MILITARY_COUNT_THRESHOLD) {
            return Strategy.HEAVY_RUSH;
        }
        if (s.lightUnits >= MILITARY_COUNT_THRESHOLD) {
            return Strategy.LIGHT_RUSH;
        }

        // Rules 3 and 4: counter the enemy's visible unit composition.
        if (s.enemyLightUnits > 0) {
            return Strategy.HEAVY_RUSH;
        }
        if (s.enemyHeavyUnits > 0) {
            // Heavy beats Light in direct combat, but Light is at least fast and applies
            // pressure. WorkerRush could occasionally out-harass slow Heavy units early,
            // but Light is the safer general answer here.
            return Strategy.LIGHT_RUSH;
        }

        // Rule 5: enemy has only workers and no dedicated military units yet.
        // Light units tear through workers very cheaply so this is a strong counter.
        if (s.enemyWorkers > 0 && (s.enemyLightUnits + s.enemyHeavyUnits + s.enemyOtherCombat) == 0) {
            return Strategy.LIGHT_RUSH;
        }

        // Rule 6: enemy attackers are already dangerously close to one of our bases.
        if (s.closestEnemyDistanceToBase <= ENEMY_PRESSURE_DISTANCE) {
            return Strategy.LIGHT_RUSH;
        }

        // Rule 7: we have enough workers to sustain the economy, time to go aggressive.
        if (s.workers >= WORKER_COUNT_THRESHOLD) {
            return smallMap ? Strategy.LIGHT_RUSH : Strategy.HEAVY_RUSH;
        }

        // Rule 8: we have passed the game-time threshold for committing to military.
        if (gameState.getTime() >= timeThreshold) {
            return smallMap ? Strategy.LIGHT_RUSH : Strategy.HEAVY_RUSH;
        }

        // Rule 9: no specific trigger fired, default to LightRush as a safe opener.
        return Strategy.LIGHT_RUSH;
    }

    /**
     * A plain data holder for the game state snapshot we build each time we
     * need to make a strategy decision. Keeping it as a simple immutable object
     * makes it easy to pass the same data to both the LLM prompt builder and
     * the heuristic without calling summarizeState twice.
     */
    private static class StrategySummary {
        final int workers;
        final int lightUnits;
        final int heavyUnits;
        final int bases;
        final int enemyWorkers;
        final int enemyLightUnits;
        final int enemyHeavyUnits;
        final int enemyOtherCombat;
        final int closestEnemyDistanceToBase;
        final int resources;
        final int mapWidth;
        final int mapHeight;

        StrategySummary(int workers, int lightUnits, int heavyUnits, int bases,
                        int enemyWorkers, int enemyLightUnits, int enemyHeavyUnits, int enemyOtherCombat,
                        int closestEnemyDistanceToBase, int resources, int mapWidth, int mapHeight) {
            this.workers = workers;
            this.lightUnits = lightUnits;
            this.heavyUnits = heavyUnits;
            this.bases = bases;
            this.enemyWorkers = enemyWorkers;
            this.enemyLightUnits = enemyLightUnits;
            this.enemyHeavyUnits = enemyHeavyUnits;
            this.enemyOtherCombat = enemyOtherCombat;
            this.closestEnemyDistanceToBase = closestEnemyDistanceToBase;
            this.resources = resources;
            this.mapWidth = mapWidth;
            this.mapHeight = mapHeight;
        }
    }

    /**
     * No runtime-tunable parameters. All configuration is done through
     * environment variables or by editing the constants at the top of the class.
     */
    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }

    /**
     * Called by the framework at the end of each game. We print a summary of
     * how many strategy decisions came from the LLM vs the fallback heuristic
     * so it is easy to verify during development that Ollama is actually being
     * consulted and not silently falling back every single turn.
     */
    @Override
    public void gameOver(int winner) {
        int totalDecisions = llmDrivenDecisionCount + heuristicDecisionCount;
        if (totalDecisions <= 0) {
            System.out.println("[AdaptiveRushBot] game_over winner=" + winner + " no strategy decisions were recorded.");
            return;
        }

        double llmPercent = (100.0 * llmDrivenDecisionCount) / totalDecisions;
        double heuristicPercent = (100.0 * heuristicDecisionCount) / totalDecisions;

        System.out.printf(
                Locale.ROOT,
                "[AdaptiveRushBot] game_over winner=%d decisions=%d llm_driven=%d (%.1f%%) heuristic=%d (%.1f%%) llm_consult_success=%d/%d%n",
                winner,
                totalDecisions,
                llmDrivenDecisionCount,
                llmPercent,
                heuristicDecisionCount,
                heuristicPercent,
                llmConsultSuccesses,
                llmConsultAttempts);
    }
}