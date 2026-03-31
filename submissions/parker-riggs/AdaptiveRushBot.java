// Submission agent package declaration
package ai.abstraction.submissions.parker_riggs;

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
 * AdaptiveRushBot is a hybrid MicroRTS agent that switches between two built-in
 * scripted policies:
 * - WorkerRush for early game pressure/economy
 * - LightRush for stronger mid/late pressure
 *
 * The strategy choice is primarily delegated to an LLM (via Ollama). If the LLM
 * call fails, times out, or returns an unrecognized answer, the bot falls back to
 * deterministic heuristics so it always remains functional.
 */
public class AdaptiveRushBot extends AI {
    // Heuristic thresholds used when LLM is unavailable or ambiguous.
    // Time threshold: after this game time, default to LightRush.
    private static final int LIGHT_RUSH_TIME_THRESHOLD = 300;
    // If we have enough workers, transition to LightRush to convert economy to pressure.
    private static final int WORKER_COUNT_THRESHOLD = 6;
    // If we already have a few light units, keep using LightRush for composition consistency.
    private static final int LIGHT_COUNT_THRESHOLD = 2;
    // If enemy combat units are this close (Manhattan distance) to any base, prefer LightRush.
    private static final int ENEMY_PRESSURE_DISTANCE = 8;
    // To avoid expensive per-tick LLM calls, re-query only every N game ticks.
    private static final int LLM_CONSULT_COOLDOWN = 25;
    // Keep decisions readable in logs without printing every frame.
    private static final int DECISION_LOG_COOLDOWN = 50;
    // Tight timeout so gameplay remains responsive even if local model is slow.
    private static final Duration LLM_TIMEOUT = Duration.ofMillis(800);

    // Core RTS context and the two underlying scripted controllers.
    private final UnitTypeTable unitTypeTable;
    private final WorkerRush workerRush;
    private final LightRush lightRush;

    // LLM HTTP client and runtime model configuration.
    // Endpoint/model can be overridden with OLLAMA_ENDPOINT and OLLAMA_MODEL env vars.
    private final HttpClient httpClient;
    private final String llmEndpoint;
    private final String llmModel;

    // Tracks the last tick when we successfully accepted an LLM decision.
    private int lastLLMConsultTime = -1000;
    // Last tick when we emitted a decision source log.
    private int lastDecisionLogTime = -1000;
    // Telemetry for confidence that LLM is contributing decisions.
    private int llmConsultAttempts = 0;
    private int llmConsultSuccesses = 0;
    private int llmDrivenDecisionCount = 0;
    private int heuristicDecisionCount = 0;
    // Most recent selected strategy. Cached to keep behavior stable between LLM polls.
    private boolean cachedUseLightRush;

    /**
     * Required constructor signature for MicroRTS agents.
     * Initializes the two strategy engines and LLM connectivity settings.
     */
    public AdaptiveRushBot(UnitTypeTable unitTypeTable) {
        this.unitTypeTable = unitTypeTable;
        this.workerRush = new WorkerRush(unitTypeTable);
        this.lightRush = new LightRush(unitTypeTable);
        this.httpClient = HttpClient.newBuilder().connectTimeout(LLM_TIMEOUT).build();
        this.llmEndpoint = resolveEndpoint();
        this.llmModel = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
        System.out.println("[AdaptiveRushBot] LLM endpoint=" + llmEndpoint + " model=" + llmModel);
    }

    @Override
    /**
     * Resets internal state between matches.
     */
    public void reset() {
        workerRush.reset();
        lightRush.reset();
        lastLLMConsultTime = -1000;
        lastDecisionLogTime = -1000;
        llmConsultAttempts = 0;
        llmConsultSuccesses = 0;
        llmDrivenDecisionCount = 0;
        heuristicDecisionCount = 0;
        cachedUseLightRush = false;
    }

    @Override
    public AI clone() {
        return new AdaptiveRushBot(unitTypeTable);
    }

    @Override
    /**
     * Main game loop callback.
     *
     * If this player can issue actions now, we decide strategy (LLM-guided with
     * fallback) and delegate actual unit action generation to the selected scripted AI.
     */
    public PlayerAction getAction(int player, GameState gameState) throws Exception {
        if (gameState.canExecuteAnyAction(player)) {
            if (decideUseLightRush(player, gameState)) {
                return lightRush.getAction(player, gameState);
            }
            return workerRush.getAction(player, gameState);
        }
        return new PlayerAction();
    }

    /**
     * Strategy selection pipeline:
     * 1) Periodically consult LLM (cooldown-gated).
     * 2) If LLM returns a valid decision, use and cache it.
     * 3) Otherwise use deterministic heuristic policy.
     */
    private boolean decideUseLightRush(int player, GameState gameState) {
        // If we recently accepted an LLM decision, keep it active until next LLM poll.
        if (lastLLMConsultTime >= 0 && gameState.getTime() - lastLLMConsultTime < LLM_CONSULT_COOLDOWN) {
            llmDrivenDecisionCount++;
            maybeLogDecision(gameState.getTime(), "LLM_CACHED", cachedUseLightRush);
            return cachedUseLightRush;
        }

        if (gameState.getTime() - lastLLMConsultTime >= LLM_CONSULT_COOLDOWN) {
            Boolean llmDecision = consultLLM(player, gameState);
            if (llmDecision != null) {
                cachedUseLightRush = llmDecision;
                lastLLMConsultTime = gameState.getTime();
                llmDrivenDecisionCount++;
                maybeLogDecision(gameState.getTime(), "LLM", cachedUseLightRush);
                return cachedUseLightRush;
            }
        }

        cachedUseLightRush = shouldUseLightRush(player, gameState);
        heuristicDecisionCount++;
        maybeLogDecision(gameState.getTime(), "HEURISTIC", cachedUseLightRush);
        return cachedUseLightRush;
    }

    /**
     * Sends a compact game-state summary to Ollama and asks for one token:
     * WORKER_RUSH or LIGHT_RUSH.
     *
     * Returns:
     * - Boolean decision when parse succeeds
     * - null when request/parsing fails, so caller can fallback to heuristics
     */
    private Boolean consultLLM(int player, GameState gameState) {
        try {
            llmConsultAttempts++;
            StrategySummary summary = summarizeState(player, gameState);
            String prompt = buildPrompt(summary, gameState.getTime());

            // Minimal JSON payload for Ollama /api/generate.
            // temperature=0 to reduce output variance and improve parse reliability.
            String body = "{\"model\":\"" + escapeJson(llmModel) + "\","
                    + "\"prompt\":\"" + escapeJson(prompt) + "\","
                    + "\"stream\":false,\"options\":{\"temperature\":0}}";

            // Synchronous request because strategy must be decided before issuing actions.
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

            // Parse permissively: prefer explicit canonical tokens, then looser keyword fallback.
            String normalized = llmText.toUpperCase(Locale.ROOT);
            if (normalized.contains("LIGHT_RUSH") || normalized.contains("LIGHTRUSH")) {
                llmConsultSuccesses++;
                return true;
            }
            if (normalized.contains("WORKER_RUSH") || normalized.contains("WORKERRUSH")) {
                llmConsultSuccesses++;
                return false;
            }
            if (normalized.contains("LIGHT")) {
                llmConsultSuccesses++;
                return true;
            }
            if (normalized.contains("WORKER")) {
                llmConsultSuccesses++;
                return false;
            }
            return null;
        } catch (Exception ignored) {
            // Any network timeout/connection/parsing issue gracefully degrades to heuristic mode.
            return null;
        }
    }

    /**
     * Produces a compact, model-friendly feature summary of the current state.
     *
     * Features are intentionally simple and low-cost to compute, balancing signal
     * quality and real-time constraints.
     */
    private StrategySummary summarizeState(int player, GameState gameState) {
        PhysicalGameState physicalGameState = gameState.getPhysicalGameState();
        int workers = 0;
        int lightUnits = 0;
        int bases = 0;
        int enemyCombatUnits = 0;
        List<Unit> ownBases = new ArrayList<>();
        int closestEnemyDistanceToBase = Integer.MAX_VALUE;

        // First pass: count own key units and enemy combat presence.
        for (Unit unit : physicalGameState.getUnits()) {
            UnitType type = unit.getType();
            String typeName = type.name;

            if (unit.getPlayer() == player) {
                if ("Worker".equals(typeName)) {
                    workers++;
                } else if ("Light".equals(typeName)) {
                    lightUnits++;
                } else if ("Base".equals(typeName)) {
                    bases++;
                    ownBases.add(unit);
                }
            } else if (unit.getPlayer() >= 0 && type.canAttack) {
                enemyCombatUnits++;
            }
        }

        // Second pass: compute nearest enemy attacker distance to any own base.
        // Manhattan distance is used (grid-appropriate and cheap).
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

        return new StrategySummary(workers, lightUnits, bases, enemyCombatUnits, closestEnemyDistanceToBase, gameState.getPlayer(player).getResources());
    }

    /**
     * Builds a constrained instruction prompt for stable strategy-token output.
     */
    private String buildPrompt(StrategySummary summary, int time) {
        return "You are selecting a MicroRTS strategy. Reply with exactly one token: WORKER_RUSH or LIGHT_RUSH.\n"
                + "Choose LIGHT_RUSH when game is maturing or enemy pressure is high. Choose WORKER_RUSH for early economy pressure.\n"
                + "State:\n"
                + "time=" + time + "\n"
                + "my_resources=" + summary.resources + "\n"
                + "my_workers=" + summary.workers + "\n"
                + "my_light_units=" + summary.lightUnits + "\n"
                + "my_bases=" + summary.bases + "\n"
                + "enemy_combat_units=" + summary.enemyCombatUnits + "\n"
                + "closest_enemy_to_base_distance=" + summary.closestEnemyDistanceToBase + "\n";
    }

    /**
     * Extracts the value of the top-level "response" field from Ollama JSON.
     *
     * This is a lightweight parser tailored for expected response shape to avoid
     * adding external JSON dependencies in submission code.
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
                return result.toString();
            }

            result.append(c);
        }

        return null;
    }

    /**
     * Escapes values inserted into manually-built JSON payload strings.
     */
    private String escapeJson(String value) {
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r");
    }

    /**
     * Resolves Ollama endpoint from env vars with backward-compatible defaults:
     * - OLLAMA_ENDPOINT (full URL) takes precedence
     * - OLLAMA_HOST (base URL) maps to /api/generate
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
     * Periodic runtime trace to confirm whether strategy decisions are LLM-driven.
     */
    private void maybeLogDecision(int time, String source, boolean useLightRush) {
        if (time - lastDecisionLogTime < DECISION_LOG_COOLDOWN) {
            return;
        }
        lastDecisionLogTime = time;
        String strategy = useLightRush ? "LIGHT_RUSH" : "WORKER_RUSH";
        System.out.println(
                "[AdaptiveRushBot] t=" + time
                        + " source=" + source
                        + " strategy=" + strategy
                        + " llm_success=" + llmConsultSuccesses
                        + "/" + llmConsultAttempts);
    }

    /**
     * Deterministic fallback strategy selector (used when LLM is unavailable).
     *
     * Rules are ordered from strongest to weakest trigger:
     * - Late game -> LightRush
     * - Sufficient own eco/army -> LightRush
     * - No base (desperation/recovery) -> LightRush
     * - Enemy close to base -> LightRush
     * - Otherwise -> WorkerRush
     */
    private boolean shouldUseLightRush(int player, GameState gameState) {
        StrategySummary summary = summarizeState(player, gameState);

        if (gameState.getTime() >= LIGHT_RUSH_TIME_THRESHOLD) {
            return true;
        }

        if (summary.lightUnits >= LIGHT_COUNT_THRESHOLD || summary.workers >= WORKER_COUNT_THRESHOLD) {
            return true;
        }

        if (summary.bases == 0) {
            return true;
        }

        return summary.closestEnemyDistanceToBase <= ENEMY_PRESSURE_DISTANCE;
    }

    /**
     * Immutable container for compact strategic features used by both
     * the LLM prompt and heuristic fallback logic.
     */
    private static class StrategySummary {
        final int workers;
        final int lightUnits;
        final int bases;
        final int enemyCombatUnits;
        final int closestEnemyDistanceToBase;
        final int resources;

        StrategySummary(int workers, int lightUnits, int bases, int enemyCombatUnits, int closestEnemyDistanceToBase, int resources) {
            this.workers = workers;
            this.lightUnits = lightUnits;
            this.bases = bases;
            this.enemyCombatUnits = enemyCombatUnits;
            this.closestEnemyDistanceToBase = closestEnemyDistanceToBase;
            this.resources = resources;
        }
    }

    @Override
    /**
     * No runtime-tunable parameters are exposed for this agent.
     */
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }

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