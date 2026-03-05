package ai.mcts.submissions.penguin_bot;

import ai.abstraction.HeavyRush;
import ai.abstraction.RangedRush;
import ai.abstraction.WorkerDefense;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import ai.mcts.naivemcts.NaiveMCTS;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import rts.GameState;
import rts.PlayerAction;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;
import util.Pair;

public class MCTSAgent extends NaiveMCTS {

    private static final int OPENING_END_TICK = 300;
    private static final int RUSH_ALERT_RADIUS = 8;
    private static final int BASE_DEFENSE_RADIUS = 4;
    private static final int LLM_INTERVAL =
            Integer.parseInt(System.getenv().getOrDefault("MCTS_LLM_INTERVAL", "400"));
    private static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String MODEL =
            System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    private static final Pattern JSON_OBJECT = Pattern.compile("\\{.*\\}", Pattern.DOTALL);

    private final UnitTypeTable utt;
    private final HeavyRush heavyRushPolicy;
    private final RangedRush rangedRushPolicy;
    private final WorkerDefense workerDefensePolicy;

    private int lastConsultTick = -9999;

    private float aggression = 0.45f;
    private float economyPriority = 0.50f;
    private float pace = 0.50f;
    private float riskPreference = 0.50f;
    private float boardControlPriority = 0.50f;
    private float pathRiskTolerance = 0.50f;
    private float terminalTrim = 0.50f;
    private int defendRangedCount = 1;
    private boolean attackWaveMode = true;
    private int attackWavePeriod = 120;
    private String unitLogicMode = "BALANCED";
    private String preferredUnit = "RANGED";
    private String preferredReason = "Default mixed opening";
    private Set<String> preferredActions = new HashSet<>();

    public MCTSAgent(UnitTypeTable utt) {
        super(120, -1, 105, 10,
              0.30f, 0.0f, 0.40f,
              new RangedRush(utt),
              new SimpleSqrtEvaluationFunction3(),
              true);
        this.utt = utt;
        this.heavyRushPolicy = new HeavyRush(utt);
        this.rangedRushPolicy = new RangedRush(utt);
        this.workerDefensePolicy = new WorkerDefense(utt);
        preferredActions.add("PRODUCE_HEAVY");
        preferredActions.add("PRODUCE_RANGED");
        preferredActions.add("DEFEND_BASE");
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (!gs.canExecuteAnyAction(player)) return new PlayerAction();

        if (gs.getTime() < OPENING_END_TICK) {
            return openingAction(player, gs);
        }

        if (gs.getTime() - lastConsultTick >= LLM_INTERVAL) {
            consultOllama(player, gs);
            lastConsultTick = gs.getTime();
        }

        applyStrategicBiases();
        return super.getAction(player, gs);
    }

    private PlayerAction openingAction(int player, GameState gs) throws Exception {
        PlayerAction productionAction = openingProductionAction(player, gs);
        if (!isGettingRushed(player, gs)) {
            return productionAction;
        }

        PlayerAction defenseAction = workerDefensePolicy.getAction(player, gs);
        return blendOpeningActions(player, gs, productionAction, defenseAction);
    }

    private PlayerAction openingProductionAction(int player, GameState gs) throws Exception {
        int enemyMelee = countEnemyMelee(player, gs);
        if (enemyMelee >= 2 || gs.getTime() % 80 < 40) {
            return rangedRushPolicy.getAction(player, gs);
        }
        return heavyRushPolicy.getAction(player, gs);
    }

    private boolean isGettingRushed(int player, GameState gs) {
        UnitType workerType = utt.getUnitType("Worker");
        List<Unit> myBases = new ArrayList<>();
        int myCombat = 0;
        int enemyThreat = 0;

        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && "Base".equals(u.getType().name)) myBases.add(u);
            if (u.getPlayer() == player && u.getType().canAttack && u.getType() != workerType) myCombat++;
        }

        for (Unit enemy : gs.getPhysicalGameState().getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player) continue;
            boolean isThreat = enemy.getType().canAttack || enemy.getType() == workerType;
            if (!isThreat) continue;
            int d = distanceToClosest(enemy, myBases);
            if (d <= RUSH_ALERT_RADIUS) enemyThreat++;
        }

        return enemyThreat >= 2 || (enemyThreat > 0 && enemyThreat > myCombat);
    }

    private PlayerAction blendOpeningActions(
            int player,
            GameState gs,
            PlayerAction productionAction,
            PlayerAction defenseAction) {
        Map<Long, Pair<Unit, UnitAction>> prod = toActionMap(productionAction);
        Map<Long, Pair<Unit, UnitAction>> def = toActionMap(defenseAction);
        Set<Long> defenseWorkers = selectDefenseWorkers(player, gs);

        PlayerAction blended = new PlayerAction();
        blended.setResourceUsage(new ResourceUsage());

        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() != player) continue;
            if (gs.getActionAssignment(u) != null) continue;

            Pair<Unit, UnitAction> prodUA = prod.get(u.getID());
            Pair<Unit, UnitAction> defUA = def.get(u.getID());
            Pair<Unit, UnitAction> selected = null;

            if ("Worker".equals(u.getType().name)) {
                if (defenseWorkers.contains(u.getID())) {
                    selected = defUA != null ? defUA : prodUA;
                } else {
                    selected = prodUA != null ? prodUA : defUA;
                }
            } else {
                selected = prodUA != null ? prodUA : defUA;
            }

            if (selected != null) addIfConsistent(blended, selected.m_a, selected.m_b, gs);
        }

        return blended;
    }

    private Set<Long> selectDefenseWorkers(int player, GameState gs) {
        List<Unit> workers = new ArrayList<>();
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && "Worker".equals(u.getType().name) && gs.getActionAssignment(u) == null) {
                workers.add(u);
            }
        }

        Collections.sort(workers, Comparator.comparingInt(u -> distanceToClosestEnemy(u, player, gs)));
        int nDefense = workers.size() / 2;
        Set<Long> ids = new HashSet<>();
        for (int i = 0; i < nDefense; i++) ids.add(workers.get(i).getID());
        return ids;
    }

    private void consultOllama(int player, GameState gs) {
        try {
            String prompt = buildPrompt(player, gs);
            String response = callOllama(prompt);
            parseStrategyFromResponse(response);
        } catch (Exception ignored) {
        }
    }

    private String buildPrompt(int player, GameState gs) {
        int enemy = 1 - player;
        int myHeavy = 0;
        int myRanged = 0;
        int enemyHeavy = 0;
        int enemyRanged = 0;
        int myWorkers = 0;
        int enemyWorkers = 0;
        int myResources = gs.getPlayer(player).getResources();
        UnitType baseType = utt.getUnitType("Base");
        List<Unit> myBases = new ArrayList<>();
        List<Unit> enemyBases = new ArrayList<>();
        int contestedResources = 0;
        int safeResources = 0;

        UnitType workerType = utt.getUnitType("Worker");
        UnitType heavyType = utt.getUnitType("Heavy");
        UnitType rangedType = utt.getUnitType("Ranged");

        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == heavyType) myHeavy++;
                else if (u.getType() == rangedType) myRanged++;
                else if (u.getType() == workerType) myWorkers++;
                else if (u.getType() == baseType) myBases.add(u);
            } else if (u.getPlayer() == enemy) {
                if (u.getType() == heavyType) enemyHeavy++;
                else if (u.getType() == rangedType) enemyRanged++;
                else if (u.getType() == workerType) enemyWorkers++;
                else if (u.getType() == baseType) enemyBases.add(u);
            }
        }

        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (!u.getType().isResource) continue;
            int myD = distanceToClosest(u, myBases);
            int enemyD = distanceToClosest(u, enemyBases);
            if (myD == Integer.MAX_VALUE || enemyD == Integer.MAX_VALUE) continue;
            if (Math.abs(myD - enemyD) <= 2) contestedResources++;
            if (myD + 1 < enemyD) safeResources++;
        }

        int myPressure = minMyCombatDistanceToEnemyBase(player, gs, enemyBases);
        int enemyPressure = minEnemyCombatDistanceToMyBase(player, gs, myBases);
        if (myPressure == Integer.MAX_VALUE) myPressure = 999;
        if (enemyPressure == Integer.MAX_VALUE) enemyPressure = 999;

        float boardControlDelta = (safeResources - contestedResources) / (float) (safeResources + contestedResources + 1);
        boardControlDelta = clamp01((boardControlDelta + 1.0f) * 0.5f);

        String pressureHint = enemyPressure <= 8 ? "HIGH" : (enemyPressure <= 14 ? "MEDIUM" : "LOW");
        String controlHint = boardControlDelta >= 0.60f ? "FAVORABLE" : (boardControlDelta <= 0.40f ? "UNFAVORABLE" : "CONTESTED");

        if (enemyPressure <= 8) preferredActions.add("DEFEND_BASE");
        if (myPressure + 2 < enemyPressure) preferredActions.add("ATTACK_NEAR_BASE");
        if (safeResources == 0 && contestedResources > 0) {
            preferredActions.add("HARVEST");
            preferredActions.add("RETURN");
        }

        int mapW = gs.getPhysicalGameState().getWidth();
        int mapH = gs.getPhysicalGameState().getHeight();
        String suggested = enemyRanged > enemyHeavy ? "HEAVY" : "RANGED";

        String example = "{\"preferred_unit\":\"" + suggested + "\",\"reason\":\"use board control and path safety\",\"aggression\":0.45,\"economy_priority\":0.50,\"pace\":0.50,\"risk\":0.45,\"board_control_priority\":0.60,\"path_risk_tolerance\":0.35,\"terminal_trim\":0.65,\"unit_logic_mode\":\"PRODUCE_DEFEND\",\"defend_ranged_count\":1,\"attack_wave_mode\":true,\"wave_period_ticks\":120,\"preferred_actions\":[\"PRODUCE_" + suggested + "\",\"HARVEST\",\"RETURN\",\"DEFEND_BASE\"]}";

        StringBuilder sb = new StringBuilder();
        sb.append("You are advising an RTS MCTS agent after the opening phase. Return JSON only.\n");
        sb.append("Focus on spatial board control and path risk when choosing heavy vs ranged.\n");
        sb.append("Policy constraints to respect: keep at least 1 ranged unit on defense near own base and prefer wave-based attacks.\n");
        sb.append("Choose the unit_logic_mode to strongly control behavior.\n");
        sb.append("State:\n");
        sb.append("- map: ").append(mapW).append("x").append(mapH).append("\n");
        sb.append("- time: ").append(gs.getTime()).append("\n");
        sb.append("- my_resources: ").append(myResources).append("\n");
        sb.append("- my_workers: ").append(myWorkers).append("\n");
        sb.append("- my_heavy: ").append(myHeavy).append("\n");
        sb.append("- my_ranged: ").append(myRanged).append("\n");
        sb.append("- enemy_workers: ").append(enemyWorkers).append("\n");
        sb.append("- enemy_heavy: ").append(enemyHeavy).append("\n");
        sb.append("- enemy_ranged: ").append(enemyRanged).append("\n");
        sb.append("- my_frontline_to_enemy_base: ").append(myPressure).append("\n");
        sb.append("- enemy_frontline_to_my_base: ").append(enemyPressure).append("\n");
        sb.append("- contested_resources: ").append(contestedResources).append("\n");
        sb.append("- safe_resources: ").append(safeResources).append("\n");
        sb.append("- board_control_delta_0_1: ").append(String.format("%.2f", boardControlDelta)).append("\n");
        sb.append("- pressure_hint: ").append(pressureHint).append("\n");
        sb.append("- control_hint: ").append(controlHint).append("\n");
        sb.append("JSON schema:\n");
        sb.append("{\"preferred_unit\":\"HEAVY|RANGED\",");
        sb.append("\"reason\":\"short explanation\",");
        sb.append("\"aggression\":0..1,");
        sb.append("\"economy_priority\":0..1,");
        sb.append("\"pace\":0..1,");
        sb.append("\"risk\":0..1,");
        sb.append("\"board_control_priority\":0..1,");
        sb.append("\"path_risk_tolerance\":0..1,");
        sb.append("\"terminal_trim\":0..1,");
        sb.append("\"unit_logic_mode\":\"ATTACK|PRODUCE|DEFEND|PRODUCE_ATTACK|PRODUCE_DEFEND|BALANCED\",");
        sb.append("\"defend_ranged_count\":0..3,");
        sb.append("\"attack_wave_mode\":true|false,");
        sb.append("\"wave_period_ticks\":60..240,");
        sb.append("\"preferred_actions\":[\"PRODUCE_HEAVY|PRODUCE_RANGED|HARVEST|RETURN|DEFEND_BASE|ATTACK_NEAR_BASE\"]}\n");
        sb.append("Example: ").append(example);
        return sb.toString();
    }

    private String callOllama(String prompt) throws Exception {
        URL url = new URL(OLLAMA_HOST + "/api/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setConnectTimeout(1800);
        conn.setReadTimeout(3200);
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");

        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("prompt", prompt);
        body.addProperty("stream", false);

        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.toString().getBytes(StandardCharsets.UTF_8));
        }

        byte[] raw = conn.getInputStream().readAllBytes();
        String envelope = new String(raw, StandardCharsets.UTF_8);
        JsonObject root = JsonParser.parseString(envelope).getAsJsonObject();
        return root.has("response") ? root.get("response").getAsString() : envelope;
    }

    private void parseStrategyFromResponse(String raw) {
        JsonObject strategy = parseStrategyJson(raw);
        if (strategy == null) return;

        if (strategy.has("preferred_unit")) {
            String v = strategy.get("preferred_unit").getAsString().toUpperCase();
            if ("HEAVY".equals(v) || "RANGED".equals(v)) preferredUnit = v;
        }
        if (strategy.has("reason")) preferredReason = strategy.get("reason").getAsString();
        if (strategy.has("aggression")) aggression = clamp01(strategy.get("aggression").getAsFloat());
        if (strategy.has("economy_priority")) economyPriority = clamp01(strategy.get("economy_priority").getAsFloat());
        if (strategy.has("pace")) pace = clamp01(strategy.get("pace").getAsFloat());
        if (strategy.has("risk")) riskPreference = clamp01(strategy.get("risk").getAsFloat());
        if (strategy.has("board_control_priority")) {
            boardControlPriority = clamp01(strategy.get("board_control_priority").getAsFloat());
        }
        if (strategy.has("path_risk_tolerance")) {
            pathRiskTolerance = clamp01(strategy.get("path_risk_tolerance").getAsFloat());
        }
        if (strategy.has("terminal_trim")) {
            terminalTrim = clamp01(strategy.get("terminal_trim").getAsFloat());
        }
        if (strategy.has("unit_logic_mode")) {
            String v = strategy.get("unit_logic_mode").getAsString().toUpperCase();
            if ("ATTACK".equals(v) || "PRODUCE".equals(v) || "DEFEND".equals(v)
                    || "PRODUCE_ATTACK".equals(v) || "PRODUCE_DEFEND".equals(v)
                    || "BALANCED".equals(v)) {
                unitLogicMode = v;
            }
        }
        if (strategy.has("defend_ranged_count")) {
            defendRangedCount = Math.max(0, Math.min(3, strategy.get("defend_ranged_count").getAsInt()));
        }
        if (strategy.has("attack_wave_mode")) {
            attackWaveMode = strategy.get("attack_wave_mode").getAsBoolean();
        }
        if (strategy.has("wave_period_ticks")) {
            attackWavePeriod = Math.max(60, Math.min(240, strategy.get("wave_period_ticks").getAsInt()));
        }

        if (strategy.has("preferred_actions") && strategy.get("preferred_actions").isJsonArray()) {
            Set<String> next = new HashSet<>();
            for (JsonElement e : strategy.get("preferred_actions").getAsJsonArray()) {
                String tag = e.getAsString().trim().toUpperCase();
                if (!tag.isEmpty()) next.add(tag);
            }
            if (!next.isEmpty()) preferredActions = next;
        }
    }

    private JsonObject parseStrategyJson(String raw) {
        String trimmed = raw == null ? "" : raw.trim();
        if (trimmed.isEmpty()) return null;

        try {
            JsonElement direct = JsonParser.parseString(trimmed);
            if (direct.isJsonObject()) return direct.getAsJsonObject();
        } catch (Exception ignored) {
        }

        Matcher m = JSON_OBJECT.matcher(trimmed);
        if (!m.find()) return null;
        try {
            JsonElement extracted = JsonParser.parseString(m.group());
            if (extracted.isJsonObject()) return extracted.getAsJsonObject();
        } catch (Exception ignored) {
        }
        return null;
    }

    private void applyStrategicBiases() {
        float tunedAggression = clamp01(aggression);
        float tunedEconomy = clamp01(economyPriority);
        float tunedPace = clamp01(pace);
        float tunedRisk = clamp01(riskPreference);
        float tunedBoard = clamp01(boardControlPriority);
        float tunedPath = clamp01(pathRiskTolerance);
        float tunedTrim = clamp01(terminalTrim);

        // Pace and aggression jointly tune rollout horizon.
        MAXSIMULATIONTIME = 70 + Math.round(50.0f * tunedAggression + 45.0f * tunedPace + 35.0f * tunedBoard);
        MAX_TREE_DEPTH = 8 + Math.round(4.0f * (1.0f - tunedEconomy) + 2.0f * tunedBoard);

        initial_epsilon_0 = 0.08f + 0.58f * tunedRisk + 0.12f * tunedPath;
        initial_epsilon_l = 0.20f + 0.50f * (1.0f - tunedEconomy);
        initial_epsilon_g = 0.0f;
        if (tunedTrim > 0.70f) initial_epsilon_0 = Math.min(initial_epsilon_0, 0.58f);

        if ("HEAVY".equals(preferredUnit)) {
            playoutPolicy = heavyRushPolicy;
            preferredActions.add("PRODUCE_HEAVY");
        } else {
            playoutPolicy = rangedRushPolicy;
            preferredActions.add("PRODUCE_RANGED");
        }
        preferredActions.add("DEFEND_BASE");
    }

    @Override
    public int getMostVisitedActionIdx() {
        total_actions_issued++;
        if (tree == null || tree.children == null || tree.children.isEmpty()) return -1;

        int bestIdx = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < tree.children.size(); i++) {
            double visits = tree.children.get(i).visit_count;
            if (visits <= 0) continue;
            PlayerAction action = tree.actions.get(i);
            if (!actionSatisfiesLogicMode(action)) continue;
            double avgEval = tree.children.get(i).accum_evaluation / visits;
            if (shouldTrimTerminalLosingBranch(action, avgEval, visits)) continue;
            int pref = preferenceScore(action);
            if (!preferredActions.isEmpty() && pref <= 0) continue;
            double score = visits + (pref * 1000.0) + Math.max(-150.0, avgEval * 120.0);
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        if (bestIdx == -1) return super.getMostVisitedActionIdx();
        return bestIdx;
    }

    private int preferenceScore(PlayerAction pa) {
        if (pa == null || preferredActions.isEmpty()) return 0;
        int score = 0;

        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            Unit u = uaa.m_a;
            UnitAction a = uaa.m_b;
            int type = a.getType();

            if (preferredActions.contains("HARVEST") && type == UnitAction.TYPE_HARVEST) score += 2;
            if (preferredActions.contains("RETURN") && type == UnitAction.TYPE_RETURN) score += 2;

            if (type == UnitAction.TYPE_PRODUCE && a.getUnitType() != null) {
                String produced = a.getUnitType().name.toUpperCase();
                if (preferredActions.contains("PRODUCE_HEAVY") && "HEAVY".equals(produced)) score += 4;
                if (preferredActions.contains("PRODUCE_RANGED") && "RANGED".equals(produced)) score += 4;
            }

            if (preferredActions.contains("ATTACK_NEAR_BASE")
                && type == UnitAction.TYPE_ATTACK_LOCATION
                && isActionNearAnyOwnBase(a, u.getPlayer(), BASE_DEFENSE_RADIUS)) {
                score += 3;
            }

            if (preferredActions.contains("DEFEND_BASE")
                && type == UnitAction.TYPE_MOVE
                && moveReducesDistanceToOwnBase(u, a, u.getPlayer())) {
                score += 2;
            }

            score += Math.round(boardControlDeltaScore(u, a, u.getPlayer()) * boardControlPriority);
            score -= Math.round(pathRiskPenalty(u, a, u.getPlayer()) * (1.0f - pathRiskTolerance));
        }
        score -= rangedDefensePenalty(pa);
        score += waveAttackScore(pa);
        return score;
    }

    private boolean shouldTrimTerminalLosingBranch(PlayerAction action, double avgEval, double visits) {
        if (visits < 6) return false;
        double lossCutoff = -0.90 + (0.75 * clamp01(terminalTrim));
        if (avgEval > lossCutoff) return false;
        int risk = actionPathRiskScore(action);
        if (pathRiskTolerance < 0.55f && risk >= 2) return true;
        return terminalTrim > 0.40f;
    }

    private boolean isActionNearAnyOwnBase(UnitAction a, int player, int radius) {
        if (gs_to_start_from == null) return false;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && "Base".equals(u.getType().name)) {
                int d = Math.abs(u.getX() - a.getLocationX()) + Math.abs(u.getY() - a.getLocationY());
                if (d <= radius) return true;
            }
        }
        return false;
    }

    private boolean moveReducesDistanceToOwnBase(Unit unit, UnitAction a, int player) {
        if (gs_to_start_from == null) return false;
        int nx = unit.getX();
        int ny = unit.getY();
        int dir = a.getDirection();
        if (dir >= 0 && dir < 4) {
            nx += UnitAction.DIRECTION_OFFSET_X[dir];
            ny += UnitAction.DIRECTION_OFFSET_Y[dir];
        }

        int before = Integer.MAX_VALUE;
        int after = Integer.MAX_VALUE;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && "Base".equals(u.getType().name)) {
                before = Math.min(before, Math.abs(unit.getX() - u.getX()) + Math.abs(unit.getY() - u.getY()));
                after = Math.min(after, Math.abs(nx - u.getX()) + Math.abs(ny - u.getY()));
            }
        }
        return after < before;
    }

    private int boardControlDeltaScore(Unit unit, UnitAction action, int player) {
        if (gs_to_start_from == null) return 0;
        if (action.getType() == UnitAction.TYPE_MOVE) {
            int nx = unit.getX();
            int ny = unit.getY();
            int dir = action.getDirection();
            if (dir >= 0 && dir < 4) {
                nx += UnitAction.DIRECTION_OFFSET_X[dir];
                ny += UnitAction.DIRECTION_OFFSET_Y[dir];
            }
            int before = minDistanceToEnemyBase(unit.getX(), unit.getY(), player);
            int after = minDistanceToEnemyBase(nx, ny, player);
            if (after < before) return 2;
            if (after > before) return -1;
        }
        if (action.getType() == UnitAction.TYPE_ATTACK_LOCATION) {
            int d = minDistanceToEnemyBase(action.getLocationX(), action.getLocationY(), player);
            if (d <= 6) return 2;
        }
        return 0;
    }

    private int pathRiskPenalty(Unit unit, UnitAction action, int player) {
        if (gs_to_start_from == null) return 0;
        int tx = unit.getX();
        int ty = unit.getY();
        if (action.getType() == UnitAction.TYPE_MOVE) {
            int dir = action.getDirection();
            if (dir >= 0 && dir < 4) {
                tx += UnitAction.DIRECTION_OFFSET_X[dir];
                ty += UnitAction.DIRECTION_OFFSET_Y[dir];
            }
        } else if (action.getType() == UnitAction.TYPE_ATTACK_LOCATION) {
            tx = action.getLocationX();
            ty = action.getLocationY();
        } else {
            return 0;
        }

        int enemiesCovering = 0;
        for (Unit enemy : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player) continue;
            if (!enemy.getType().canAttack) continue;
            int d = Math.abs(enemy.getX() - tx) + Math.abs(enemy.getY() - ty);
            if (d <= enemy.getAttackRange() + 1) enemiesCovering++;
        }
        return Math.min(4, enemiesCovering);
    }

    private int actionPathRiskScore(PlayerAction pa) {
        if (pa == null) return 0;
        int worst = 0;
        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            worst = Math.max(worst, pathRiskPenalty(uaa.m_a, uaa.m_b, uaa.m_a.getPlayer()));
        }
        return worst;
    }

    private int rangedDefensePenalty(PlayerAction pa) {
        if (defendRangedCount <= 0 || gs_to_start_from == null || pa == null) return 0;
        int player = pa.getActions().isEmpty() ? -1 : pa.getActions().get(0).m_a.getPlayer();
        if (player < 0) return 0;

        Map<Long, UnitAction> chosen = new HashMap<>();
        for (Pair<Unit, UnitAction> uaa : pa.getActions()) chosen.put(uaa.m_a.getID(), uaa.m_b);

        int rangedNearBase = 0;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() != player || !"Ranged".equals(u.getType().name)) continue;
            UnitAction a = chosen.get(u.getID());
            int px = projectedX(u, a);
            int py = projectedY(u, a);
            if (isPositionNearAnyOwnBase(px, py, player, BASE_DEFENSE_RADIUS + 1)) rangedNearBase++;
        }

        if (rangedNearBase >= defendRangedCount) return 0;
        return (defendRangedCount - rangedNearBase) * 12;
    }

    private int waveAttackScore(PlayerAction pa) {
        if (pa == null) return 0;
        boolean attackWindow = isAttackWaveWindow();
        int score = 0;
        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            Unit u = uaa.m_a;
            UnitAction a = uaa.m_b;
            int t = a.getType();
            boolean offensive = false;
            if (t == UnitAction.TYPE_ATTACK_LOCATION && !isActionNearAnyOwnBase(a, u.getPlayer(), BASE_DEFENSE_RADIUS)) {
                offensive = true;
            } else if (t == UnitAction.TYPE_MOVE && moveReducesDistanceToEnemyBase(u, a, u.getPlayer())) {
                offensive = true;
            }
            if (!offensive) continue;
            score += attackWindow ? 3 : -4;
        }
        return score;
    }

    private boolean actionSatisfiesLogicMode(PlayerAction pa) {
        if (pa == null || "BALANCED".equals(unitLogicMode)) return true;
        int produce = 0;
        int attack = 0;
        int defend = 0;
        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            Unit u = uaa.m_a;
            UnitAction a = uaa.m_b;
            int t = a.getType();
            if (t == UnitAction.TYPE_PRODUCE) produce++;
            if (t == UnitAction.TYPE_ATTACK_LOCATION && !isActionNearAnyOwnBase(a, u.getPlayer(), BASE_DEFENSE_RADIUS)) {
                attack++;
            } else if (t == UnitAction.TYPE_MOVE && moveReducesDistanceToEnemyBase(u, a, u.getPlayer())) {
                attack++;
            }
            if ((t == UnitAction.TYPE_ATTACK_LOCATION && isActionNearAnyOwnBase(a, u.getPlayer(), BASE_DEFENSE_RADIUS))
                    || (t == UnitAction.TYPE_MOVE && moveReducesDistanceToOwnBase(u, a, u.getPlayer()))) {
                defend++;
            }
        }

        switch (unitLogicMode) {
            case "ATTACK":
                return attack >= 1 && defend <= attack;
            case "PRODUCE":
                return produce >= 1 && attack == 0;
            case "DEFEND":
                return defend >= 1 && attack == 0;
            case "PRODUCE_ATTACK":
                return produce >= 1 && attack >= 1;
            case "PRODUCE_DEFEND":
                return produce >= 1 && defend >= 1;
            default:
                return true;
        }
    }

    private boolean isAttackWaveWindow() {
        if (!attackWaveMode || gs_to_start_from == null) return true;
        int period = Math.max(60, attackWavePeriod);
        int phase = (gs_to_start_from.getTime() / period) % 2;
        return phase == 1;
    }

    private boolean moveReducesDistanceToEnemyBase(Unit unit, UnitAction a, int player) {
        int before = minDistanceToEnemyBase(unit.getX(), unit.getY(), player);
        int after = minDistanceToEnemyBase(projectedX(unit, a), projectedY(unit, a), player);
        return after < before;
    }

    private int projectedX(Unit u, UnitAction a) {
        if (a == null || a.getType() != UnitAction.TYPE_MOVE) return u.getX();
        int dir = a.getDirection();
        if (dir >= 0 && dir < 4) return u.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
        return u.getX();
    }

    private int projectedY(Unit u, UnitAction a) {
        if (a == null || a.getType() != UnitAction.TYPE_MOVE) return u.getY();
        int dir = a.getDirection();
        if (dir >= 0 && dir < 4) return u.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
        return u.getY();
    }

    private boolean isPositionNearAnyOwnBase(int x, int y, int player, int radius) {
        if (gs_to_start_from == null) return false;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && "Base".equals(u.getType().name)) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if (d <= radius) return true;
            }
        }
        return false;
    }

    private static float clamp01(float v) {
        return Math.max(0.0f, Math.min(1.0f, v));
    }

    private int countEnemyMelee(int player, GameState gs) {
        int enemyMelee = 0;
        UnitType workerType = utt.getUnitType("Worker");
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player && u.getType().canAttack && u.getType() != workerType) {
                enemyMelee++;
            }
        }
        return enemyMelee;
    }

    private int distanceToClosest(Unit from, List<Unit> targets) {
        if (targets == null || targets.isEmpty()) return Integer.MAX_VALUE;
        int best = Integer.MAX_VALUE;
        for (Unit t : targets) {
            int d = Math.abs(from.getX() - t.getX()) + Math.abs(from.getY() - t.getY());
            if (d < best) best = d;
        }
        return best;
    }

    private int distanceToClosestEnemy(Unit me, int player, GameState gs) {
        int best = 9999;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player) {
                int d = Math.abs(me.getX() - u.getX()) + Math.abs(me.getY() - u.getY());
                best = Math.min(best, d);
            }
        }
        return best;
    }

    private int minDistanceToEnemyBase(int x, int y, int player) {
        if (gs_to_start_from == null) return 9999;
        int best = 9999;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player && "Base".equals(u.getType().name)) {
                int d = Math.abs(x - u.getX()) + Math.abs(y - u.getY());
                best = Math.min(best, d);
            }
        }
        return best;
    }

    private int minMyCombatDistanceToEnemyBase(int player, GameState gs, List<Unit> enemyBases) {
        if (enemyBases == null || enemyBases.isEmpty()) return Integer.MAX_VALUE;
        int best = Integer.MAX_VALUE;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() != player || !u.getType().canAttack || u.getType().canHarvest) continue;
            best = Math.min(best, distanceToClosest(u, enemyBases));
        }
        return best;
    }

    private int minEnemyCombatDistanceToMyBase(int player, GameState gs, List<Unit> myBases) {
        if (myBases == null || myBases.isEmpty()) return Integer.MAX_VALUE;
        int best = Integer.MAX_VALUE;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() < 0 || u.getPlayer() == player || !u.getType().canAttack || u.getType().canHarvest) continue;
            best = Math.min(best, distanceToClosest(u, myBases));
        }
        return best;
    }

    private Map<Long, Pair<Unit, UnitAction>> toActionMap(PlayerAction pa) {
        Map<Long, Pair<Unit, UnitAction>> map = new HashMap<>();
        if (pa == null) return map;
        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            map.put(uaa.m_a.getID(), uaa);
        }
        return map;
    }

    private void addIfConsistent(PlayerAction out, Unit u, UnitAction a, GameState gs) {
        ResourceUsage ru = a.resourceUsage(u, gs.getPhysicalGameState());
        if (out.consistentWith(ru, gs)) {
            out.addUnitAction(u, a);
            out.getResourceUsage().merge(ru);
        }
    }

    @Override
    public AI clone() {
        MCTSAgent cloned = new MCTSAgent(utt);
        cloned.setTimeBudget(TIME_BUDGET);
        cloned.setIterationsBudget(ITERATIONS_BUDGET);
        cloned.MAXSIMULATIONTIME = MAXSIMULATIONTIME;
        cloned.MAX_TREE_DEPTH = MAX_TREE_DEPTH;
        cloned.epsilon_l = epsilon_l;
        cloned.epsilon_g = epsilon_g;
        cloned.epsilon_0 = epsilon_0;
        cloned.initial_epsilon_l = initial_epsilon_l;
        cloned.initial_epsilon_g = initial_epsilon_g;
        cloned.initial_epsilon_0 = initial_epsilon_0;
        cloned.aggression = aggression;
        cloned.economyPriority = economyPriority;
        cloned.pace = pace;
        cloned.riskPreference = riskPreference;
        cloned.boardControlPriority = boardControlPriority;
        cloned.pathRiskTolerance = pathRiskTolerance;
        cloned.terminalTrim = terminalTrim;
        cloned.defendRangedCount = defendRangedCount;
        cloned.attackWaveMode = attackWaveMode;
        cloned.attackWavePeriod = attackWavePeriod;
        cloned.unitLogicMode = unitLogicMode;
        cloned.preferredUnit = preferredUnit;
        cloned.preferredReason = preferredReason;
        cloned.preferredActions = new HashSet<>(preferredActions);
        cloned.playoutPolicy = "HEAVY".equals(preferredUnit) ? cloned.heavyRushPolicy : cloned.rangedRushPolicy;
        return cloned;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}
