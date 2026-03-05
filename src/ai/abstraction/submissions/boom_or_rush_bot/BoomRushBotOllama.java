package ai.abstraction.submissions.boom_or_rush_bot;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.BoomEconomy;
import ai.abstraction.LightRush;
import ai.abstraction.WorkerRush;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

public class BoomRushBotOllama extends AbstractionLayerAI {

    private enum StrategyChoice {
        WORKER_RUSH,
        LIGHT_RUSH,
        BOOM_ECONOMY
    }

    private static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String OLLAMA_MODEL =
            System.getenv().getOrDefault("OLLAMA_MODEL", "qwen3.5:9b");
    private static final boolean DEBUG_ENABLED =
            Boolean.parseBoolean(System.getenv().getOrDefault("BOOMRUSHBOT_DEBUG", "false"));
    private static final int CONSULT_INTERVAL = getEnvInt("OLLAMA_SUBMISSION_INTERVAL", 2000);
    private static final int DEBUG_TICK_INTERVAL = getEnvInt("BOOMRUSHBOT_DEBUG_TICK_INTERVAL", 100);
    private static final int CONNECT_TIMEOUT_MS = getEnvInt("OLLAMA_SUBMISSION_CONNECT_TIMEOUT_MS", 2000);
    private static final int READ_TIMEOUT_MS = getEnvInt("OLLAMA_SUBMISSION_READ_TIMEOUT_MS", 30000);

    protected UnitTypeTable utt;
    private WorkerRush workerRushAI;
    private LightRush lightRushAI;
    private BoomEconomy boomEconomyAI;

    private UnitType workerType;
    private UnitType baseType;
    private UnitType barracksType;
    private UnitType lightType;

    private StrategyChoice currentChoice = StrategyChoice.WORKER_RUSH;
    private int lastConsultationTick = Integer.MIN_VALUE / 4;

    public BoomRushBotOllama(UnitTypeTable a_utt) {
        super(new AStarPathFinding());
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        if (workerRushAI != null) {
            workerRushAI.reset();
        }
        if (lightRushAI != null) {
            lightRushAI.reset();
        }
        if (boomEconomyAI != null) {
            boomEconomyAI.reset();
        }
        lastConsultationTick = Integer.MIN_VALUE / 4;
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        lightType = utt.getUnitType("Light");

        workerRushAI = new WorkerRush(a_utt, pf);
        lightRushAI = new LightRush(a_utt, pf);
        boomEconomyAI = new BoomEconomy(a_utt, pf);

        currentChoice = StrategyChoice.WORKER_RUSH;
        lastConsultationTick = Integer.MIN_VALUE / 4;
    }

    @Override
    public AI clone() {
        BoomRushBotOllama clone = new BoomRushBotOllama(utt);
        clone.currentChoice = currentChoice;
        return clone;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (DEBUG_ENABLED && gs.getTime() % Math.max(1, DEBUG_TICK_INTERVAL) == 0) {
            debug("T=%d active=%s last_consult=%d", gs.getTime(), currentChoice, lastConsultationTick);
        }

        if (gs.getTime() == 0 || gs.getTime() - lastConsultationTick >= Math.max(1, CONSULT_INTERVAL)) {
            StrategyChoice nextChoice = askOllama(player, gs);
            if (nextChoice == null) {
                nextChoice = chooseFallback(player, gs);
            }
            if (nextChoice != currentChoice) {
                debug("T=%d switching strategy %s -> %s", gs.getTime(), currentChoice, nextChoice);
                currentChoice = nextChoice;
                getActiveAI().reset();
            } else {
                debug("T=%d keeping strategy %s", gs.getTime(), currentChoice);
            }
            lastConsultationTick = gs.getTime();
        }

        return getActiveAI().getAction(player, gs);
    }

    private AbstractionLayerAI getActiveAI() {
        switch (currentChoice) {
            case LIGHT_RUSH:
                return lightRushAI;
            case BOOM_ECONOMY:
                return boomEconomyAI;
            case WORKER_RUSH:
            default:
                return workerRushAI;
        }
    }

    private StrategyChoice askOllama(int player, GameState gs) {
        try {
            String prompt = buildPrompt(player, gs);
            debug("T=%d consulting Ollama model=%s", gs.getTime(), OLLAMA_MODEL);
            String response = callOllama(prompt);
            debug("T=%d Ollama raw response=%s", gs.getTime(), response);
            StrategyChoice parsed = parseChoice(response);
            if (parsed == null) {
                debug("T=%d Ollama response could not be parsed into a strategy", gs.getTime());
            } else {
                debug("T=%d Ollama selected %s", gs.getTime(), parsed);
            }
            return parsed;
        } catch (IOException e) {
            debug("T=%d Ollama request failed: %s", gs.getTime(), e.getMessage());
            return null;
        }
    }

    private String buildPrompt(int player, GameState gs) {
        StateSnapshot snapshot = analyzeState(player, gs);
        StringBuilder prompt = new StringBuilder();

        prompt.append("You are selecting a MicroRTS macro strategy.\n");
        prompt.append("Choose exactly one option for the next phase of play.\n");
        prompt.append("Do not include reasoning, chain-of-thought, or extra commentary.\n");
        prompt.append("Respond with JSON only: {\"strategy\":\"WORKER_RUSH\"} or ");
        prompt.append("{\"strategy\":\"LIGHT_RUSH\"} or {\"strategy\":\"BOOM_ECONOMY\"}.\n");
        prompt.append("Strategy summaries:\n");
        prompt.append("- WORKER_RUSH: immediate pressure with workers; strongest early rush.\n");
        prompt.append("- LIGHT_RUSH: build barracks and produce light units for balanced offense.\n");
        prompt.append("- BOOM_ECONOMY: prioritize workers and economy before committing to army.\n");
        prompt.append("Current state:\n");
        prompt.append("- Time: ").append(gs.getTime()).append("\n");
        prompt.append("- Map: ").append(gs.getPhysicalGameState().getWidth()).append("x");
        prompt.append(gs.getPhysicalGameState().getHeight()).append("\n");
        prompt.append("- My resources: ").append(gs.getPlayer(player).getResources()).append("\n");
        prompt.append("- My units: ").append(snapshot.myWorkers).append(" workers, ");
        prompt.append(snapshot.myLight).append(" lights, ").append(snapshot.myCombatUnits);
        prompt.append(" combat units, ").append(snapshot.myBases).append(" bases, ");
        prompt.append(snapshot.myBarracks).append(" barracks\n");
        prompt.append("- Enemy units: ").append(snapshot.enemyWorkers).append(" workers, ");
        prompt.append(snapshot.enemyCombatUnits).append(" combat units, ");
        prompt.append(snapshot.enemyBases).append(" bases, ").append(snapshot.enemyBarracks);
        prompt.append(" barracks\n");
        prompt.append("- Under pressure: ").append(snapshot.enemyCombatNearby ? "yes" : "no").append("\n");

        return prompt.toString();
    }

    private String callOllama(String prompt) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(OLLAMA_HOST + "/api/generate").openConnection();
        try {
            connection.setRequestMethod("POST");
            connection.setDoOutput(true);
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");

            JsonObject payload = new JsonObject();
            payload.addProperty("model", OLLAMA_MODEL);
            payload.addProperty("prompt", prompt);
            payload.addProperty("stream", false);
            payload.addProperty("think", false);

            byte[] body = payload.toString().getBytes(StandardCharsets.UTF_8);
            try (OutputStream os = connection.getOutputStream()) {
                os.write(body);
            }

            int status = connection.getResponseCode();
            String responseBody = readAll(status >= 200 && status < 300
                    ? connection.getInputStream()
                    : connection.getErrorStream());

            if (status < 200 || status >= 300) {
                throw new IOException("Ollama returned status " + status + ": " + responseBody);
            }

            JsonObject responseJson = JsonParser.parseString(responseBody).getAsJsonObject();
            if (!responseJson.has("response")) {
                throw new IOException("Ollama response did not include a response field");
            }

            return responseJson.get("response").getAsString();
        } finally {
            connection.disconnect();
        }
    }

    private StrategyChoice parseChoice(String response) {
        if (response == null || response.trim().isEmpty()) {
            debug("Received empty strategy response from Ollama");
            return null;
        }

        String candidate = response.trim();
        if (candidate.startsWith("{")) {
            try {
                JsonObject json = JsonParser.parseString(candidate).getAsJsonObject();
                if (json.has("strategy")) {
                    candidate = json.get("strategy").getAsString();
                }
            } catch (RuntimeException ignored) {
                // Fall back to token matching below.
            }
        }

        String normalized = candidate.toUpperCase().replace('-', '_').replace(' ', '_');
        if (normalized.contains("BOOM_ECONOMY") || normalized.contains("BOOMECONOMY") || normalized.equals("BOOM")) {
            return StrategyChoice.BOOM_ECONOMY;
        }
        if (normalized.contains("LIGHT_RUSH") || normalized.equals("LIGHT")) {
            return StrategyChoice.LIGHT_RUSH;
        }
        if (normalized.contains("WORKER_RUSH") || normalized.equals("WORKER")) {
            return StrategyChoice.WORKER_RUSH;
        }

        debug("Unrecognized strategy token from Ollama: %s", candidate);
        return null;
    }

    private StrategyChoice chooseFallback(int player, GameState gs) {
        StateSnapshot snapshot = analyzeState(player, gs);
        Player me = gs.getPlayer(player);

        if (snapshot.enemyCombatNearby && snapshot.myCombatUnits == 0) {
            debug("T=%d fallback -> WORKER_RUSH (enemy nearby, no combat units)", gs.getTime());
            return StrategyChoice.WORKER_RUSH;
        }
        if (snapshot.myBarracks > 0 || me.getResources() >= barracksType.cost) {
            debug("T=%d fallback -> LIGHT_RUSH (barracks ready or affordable)", gs.getTime());
            return StrategyChoice.LIGHT_RUSH;
        }
        if (snapshot.myWorkers < 4 && !snapshot.enemyCombatNearby) {
            debug("T=%d fallback -> BOOM_ECONOMY (building economy safely)", gs.getTime());
            return StrategyChoice.BOOM_ECONOMY;
        }
        if (gs.getTime() < 150) {
            debug("T=%d fallback -> WORKER_RUSH (early game pressure)", gs.getTime());
            return StrategyChoice.WORKER_RUSH;
        }
        debug("T=%d fallback -> LIGHT_RUSH (default late fallback)", gs.getTime());
        return StrategyChoice.LIGHT_RUSH;
    }

    private StateSnapshot analyzeState(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemyPlayer = 1 - player;
        StateSnapshot snapshot = new StateSnapshot();

        for (Unit unit : pgs.getUnits()) {
            if (unit.getPlayer() == player) {
                if (unit.getType() == workerType) {
                    snapshot.myWorkers++;
                } else if (unit.getType() == lightType) {
                    snapshot.myLight++;
                }

                if (unit.getType() == baseType) {
                    snapshot.myBases++;
                } else if (unit.getType() == barracksType) {
                    snapshot.myBarracks++;
                }

                if (unit.getType().canAttack && !unit.getType().canHarvest) {
                    snapshot.myCombatUnits++;
                }
            } else if (unit.getPlayer() == enemyPlayer) {
                if (unit.getType() == workerType) {
                    snapshot.enemyWorkers++;
                }
                if (unit.getType() == baseType) {
                    snapshot.enemyBases++;
                } else if (unit.getType() == barracksType) {
                    snapshot.enemyBarracks++;
                }
                if (unit.getType().canAttack && !unit.getType().canHarvest) {
                    snapshot.enemyCombatUnits++;
                }
            }
        }

        for (Unit myUnit : pgs.getUnits()) {
            if (myUnit.getPlayer() != player) {
                continue;
            }
            for (Unit enemyUnit : pgs.getUnits()) {
                if (enemyUnit.getPlayer() != enemyPlayer || !enemyUnit.getType().canAttack) {
                    continue;
                }
                int distance = Math.abs(myUnit.getX() - enemyUnit.getX()) + Math.abs(myUnit.getY() - enemyUnit.getY());
                if (distance <= 4) {
                    snapshot.enemyCombatNearby = true;
                    return snapshot;
                }
            }
        }

        return snapshot;
    }

    private static String readAll(InputStream stream) throws IOException {
        if (stream == null) {
            return "";
        }

        StringBuilder sb = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
            }
        }
        return sb.toString();
    }

    private static int getEnvInt(String name, int fallback) {
        try {
            return Integer.parseInt(System.getenv().getOrDefault(name, Integer.toString(fallback)));
        } catch (NumberFormatException e) {
            return fallback;
        }
    }

    private static void debug(String format, Object... args) {
        if (!DEBUG_ENABLED) {
            return;
        }
        System.out.println("[BoomRushBot] " + String.format(format, args));
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }

    private static final class StateSnapshot {
        int myWorkers;
        int myLight;
        int myCombatUnits;
        int myBases;
        int myBarracks;
        int enemyWorkers;
        int enemyCombatUnits;
        int enemyBases;
        int enemyBarracks;
        boolean enemyCombatNearby;
    }
}
