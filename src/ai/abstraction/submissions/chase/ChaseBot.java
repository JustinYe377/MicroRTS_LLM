package ai.abstraction.submissions.chase;

import ai.abstraction.AbstractAction;
import ai.abstraction.Harvest;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import rts.*;
import rts.units.*;
import util.Pair;
import java.util.*;
import java.io.*;
import java.net.*;
import com.google.gson.*;




public class ChaseBot extends AIWithComputationBudget {

    private static final int IDX_WORKER = 0;
    private static final int IDX_LIGHT = 1;
    private static final int IDX_HEAVY = 2;
    private static final int IDX_RANGED = 3;
    private static final int IDX_BASE = 4;
    private static final int IDX_BARRACKS = 5;

    private final UnitTypeTable localUtt;
    private final AStarPathFinding aStar = new AStarPathFinding();

    // [attack/defense][worker/military]
    private final AwareAI[][] strategies;

    private boolean initialized = false;
    private int mapMaxSize;

    private int[] playerUnits = new int[] {0, 0, 0, 0, 0, 0};
    private int[] enemyUnits = new int[] {0, 0, 0, 0, 0, 0};

    private Unit mainBase = null;
    private Unit enemyBase = null;
    private Unit closestEnemy = null;

    private int baseToResources = 9999;
    private int enemyToPlayerBase = 9999;
    private int playerToEnemyBase = 9999;
    private int baseToEnemyBase = 9999;
    private int realBaseToEnemy = -1;

    private int resourceThreshold = 6;
    private int awareness = 4;
    private double strategyPriority = 5.0;

    // smooth decisions to reduce oscillation every frame
    private double attackMomentum = 0.0;
    private double militaryMomentum = 0.0;

    // LLM integration (GPT-5 via Ollama proxy)
    private static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String OLLAMA_MODEL =
            System.getenv().getOrDefault("OLLAMA_MODEL", "gpt-5");
    private static final int LLM_CONSULT_INTERVAL = 200;
    private int lastLLMConsultTick = -999;
    private double llmAttackBias = 0.0;
    private double llmMilitaryBias = 0.0;
    private String llmUnitPreference = "balanced";

    public ChaseBot(AwareAI[][] s, int timeBudget, int iterationsBudget, UnitTypeTable utt) {
        super(timeBudget, iterationsBudget);
        this.localUtt = utt;
        this.strategies = s;
    }

    public ChaseBot(UnitTypeTable utt) {
        this(
                new AwareAI[][] {
                        {new AWorkDefense(utt), new AMilDefense(utt)},
                        {new AWorkRush(utt), new AMilRush(utt)}
                },
                100,
                -1,
                utt
        );
    }

    @Override
    public void reset() {
        initialized = false;
        attackMomentum = 0.0;
        militaryMomentum = 0.0;
        mainBase = null;
        enemyBase = null;
        closestEnemy = null;
        lastLLMConsultTick = -999;
        llmAttackBias = 0.0;
        llmMilitaryBias = 0.0;
        llmUnitPreference = "balanced";
    }

    @Override
    public AI clone() {
        return new ChaseBot(TIME_BUDGET, ITERATIONS_BUDGET, localUtt);
    }

    private ChaseBot(int timeBudget, int iterationsBudget, UnitTypeTable utt) {
        this(
                new AwareAI[][] {
                        {new AWorkDefense(utt), new AMilDefense(utt)},
                        {new AWorkRush(utt), new AMilRush(utt)}
                },
                timeBudget,
                iterationsBudget,
                utt
        );
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("TimeBudget", int.class, 100));
        parameters.add(new ParameterSpecification("IterationsBudget", int.class, -1));
        return parameters;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (!initialized) {
            mapMaxSize = Math.max(gs.getPhysicalGameState().getHeight(), gs.getPhysicalGameState().getWidth());
            strategyPriority = calibrateStrategyPriority(mapMaxSize);
            initialized = true;
        }

        if (!gs.canExecuteAnyAction(player)) {
            return new PlayerAction();
        }

        updateUnitDistribution(player, gs);
        refreshMapDistances(player, gs);
        consultLLM(player, gs);

        AwareAI macroStrategy = getMacroStrategy(player, gs);
        calibrateStrategy(macroStrategy, gs, player);
        return macroStrategy.getAction(player, gs);
    }

    private double calibrateStrategyPriority(int mapSize) {
        if (mapSize <= 8) {
            return 10.0;
        }
        if (mapSize <= 16) {
            return 6.0;
        }
        if (mapSize <= 32) {
            return 4.0;
        }
        return 3.0;
    }

    private AwareAI getMacroStrategy(int player, GameState gs) {
        int resources = gs.getPlayer(player).getResources();

        int myMilitary = weightedMilitaryCount(playerUnits);
        int enemyMilitary = weightedMilitaryCount(enemyUnits);
        int militaryLead = myMilitary - enemyMilitary;

        int myEconomy = playerUnits[IDX_WORKER] + (2 * playerUnits[IDX_BASE]) + playerUnits[IDX_BARRACKS];
        int enemyEconomy = enemyUnits[IDX_WORKER] + (2 * enemyUnits[IDX_BASE]) + enemyUnits[IDX_BARRACKS];
        int economyLead = myEconomy - enemyEconomy;

        boolean underThreat = enemyToPlayerBase <= 8;
        boolean enemyCollapsed = enemyUnits[IDX_BASE] == 0 && enemyUnits[IDX_WORKER] <= 2;
        boolean canReachEnemy = realBaseToEnemy >= 0;

        double attackScore =
                (0.9 * militaryLead)
                        + (0.5 * economyLead)
                        + (resources >= 8 ? 1.5 : 0.0)
                        + (canReachEnemy ? 1.0 : -1.0)
                        + (enemyCollapsed ? 3.0 : 0.0)
                        + (underThreat ? -3.0 : 0.0)
                        + (playerUnits[IDX_BASE] == 0 ? -2.0 : 0.0)
                        + llmAttackBias;

        double militaryScore =
                (resources >= resourceThreshold ? 2.0 : -1.0)
                        + (playerUnits[IDX_BARRACKS] > 0 ? 1.5 : -1.5)
                        + (enemyMilitary > 0 ? 1.2 : 0.0)
                        + (mapMaxSize >= 24 ? 0.6 : 0.0)
                        + (playerUnits[IDX_WORKER] <= 1 ? -2.5 : 0.0)
                        + (playerUnits[IDX_BASE] == 0 ? -4.0 : 0.0)
                        + llmMilitaryBias;

        attackMomentum = 0.70 * attackMomentum + 0.30 * attackScore;
        militaryMomentum = 0.70 * militaryMomentum + 0.30 * militaryScore;

        int attackIdx = attackMomentum >= 0.0 ? 1 : 0;
        int unitModeIdx = militaryMomentum >= 0.0 ? 1 : 0;

        return strategies[attackIdx][unitModeIdx];
    }

    // Tune internal parameters for the selected strategy each frame.
    private void calibrateStrategy(AwareAI strategy, GameState gs, int player) {
        Player p = gs.getPlayer(player);
        int resources = p.getResources();

        int harvestUnits = computeHarvestTarget(gs);
        int totalBarracks = computeBarracksTarget(resources);
        int[] unitProduction = calibrateUnitProduction(gs);

        if (mapMaxSize >= 48 && realBaseToEnemy >= 0) {
            resourceThreshold = 10;
        } else if (playerUnits[IDX_BARRACKS] < totalBarracks) {
            resourceThreshold = mapMaxSize <= 16 ? 5 * (totalBarracks - playerUnits[IDX_BARRACKS]) : 8 * (totalBarracks - playerUnits[IDX_BARRACKS]);
        } else {
            resourceThreshold = 4 + (3 * playerUnits[IDX_BARRACKS]);
        }

        strategy.setnHarvest(harvestUnits);
        strategy.setTotBarracks(totalBarracks);
        strategy.setUnitProduction(unitProduction);
        strategy.setResourceTreshold(resourceThreshold);
        strategy.setPlayerUnits(playerUnits);
        strategy.setEnemyUnits(enemyUnits);
        strategy.setUnitAwareness(awareness);
        strategy.setStrategyPriority(strategyPriority);
    }

    private int computeHarvestTarget(GameState gs) {
        if (mainBase == null) {
            return 0;
        }

        int maxHarvest = Math.max(2, Math.min(6, 1 + (mapMaxSize / 12)));
        int pressure = enemyToPlayerBase <= 10 ? 1 : 0;
        int nearbyResources = countNearbyResources(mainBase, gs.getPhysicalGameState(), 8);

        int target = 2 + (nearbyResources >= 3 ? 1 : 0) + (mapMaxSize >= 32 ? 1 : 0) - pressure;
        target = Math.max(1, Math.min(maxHarvest, target));

        return Math.min(target, playerUnits[IDX_WORKER]);
    }

    private int computeBarracksTarget(int resources) {
        int target = 1;
        if (mapMaxSize >= 24) {
            target++;
        }
        if (resources >= 12 && playerUnits[IDX_WORKER] >= 4 && mapMaxSize >= 32) {
            target++;
        }
        return Math.min(target, 3);
    }

    private int[] calibrateUnitProduction(GameState gs) {
        int[] units = new int[] {1, 1, 1}; // [Light, Heavy, Ranged]

        int myFrontline = playerUnits[IDX_LIGHT] + playerUnits[IDX_HEAVY] + playerUnits[IDX_RANGED];
        int enemyFrontline = enemyUnits[IDX_LIGHT] + enemyUnits[IDX_HEAVY] + enemyUnits[IDX_RANGED];

        if (myFrontline < enemyFrontline) {
            units[IDX_LIGHT - 1] += 1;
            units[IDX_HEAVY - 1] += 1;
        }

        if (enemyUnits[IDX_LIGHT] > enemyUnits[IDX_HEAVY]) {
            units[IDX_HEAVY - 1] += 2; // heavies trade well into light swarms
        }
        if (enemyUnits[IDX_HEAVY] > enemyUnits[IDX_LIGHT]) {
            units[IDX_RANGED - 1] += 2; // ranged punishes slow heavy units
        }
        if (enemyUnits[IDX_RANGED] > 0) {
            units[IDX_LIGHT - 1] += 1; // lights can collapse ranged quickly
        }

        boolean blockedPath = false;
        if (mainBase != null && closestEnemy != null) {
            int d = aStar.findDistToPositionInRange(mainBase, closestEnemy.getPosition(gs.getPhysicalGameState()), 1, gs, gs.getResourceUsage());
            blockedPath = d < 0;
        }

        if (!canReachEnemyBase() || blockedPath || mapMaxSize >= 32) {
            units[IDX_RANGED - 1] += 2;
        }

        if (playerUnits[IDX_BARRACKS] == 0) {
            units[IDX_HEAVY - 1] += 1;
        }

        // Apply LLM unit preference
        if ("light".equals(llmUnitPreference)) {
            units[IDX_LIGHT - 1] += 2;
        } else if ("heavy".equals(llmUnitPreference)) {
            units[IDX_HEAVY - 1] += 2;
        } else if ("ranged".equals(llmUnitPreference)) {
            units[IDX_RANGED - 1] += 2;
        }

        return units;
    }

    private boolean canReachEnemyBase() {
        return realBaseToEnemy >= 0;
    }

    private int weightedMilitaryCount(int[] units) {
        return units[IDX_WORKER] + (2 * units[IDX_LIGHT]) + (3 * units[IDX_HEAVY]) + (2 * units[IDX_RANGED]);
    }

    private void refreshMapDistances(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        int closestResourceDist = 9999;
        int closestEnemyToBase = 9999;
        int closestAllyToEnemyBase = 9999;
        Unit nearestEnemy = null;

        for (Unit u : pgs.getUnits()) {
            if (mainBase != null && u.getType().isResource) {
                int d = manhattan(mainBase, u);
                if (d < closestResourceDist) {
                    closestResourceDist = d;
                }
            }

            if (mainBase != null && u.getPlayer() >= 0 && u.getPlayer() != player) {
                int d = manhattan(mainBase, u);
                if (d < closestEnemyToBase) {
                    closestEnemyToBase = d;
                    nearestEnemy = u;
                }
            }

            if (enemyBase != null && u.getPlayer() == player) {
                int d = manhattan(enemyBase, u);
                if (d < closestAllyToEnemyBase) {
                    closestAllyToEnemyBase = d;
                }
            }
        }

        closestEnemy = nearestEnemy;
        baseToResources = closestResourceDist;
        enemyToPlayerBase = closestEnemyToBase;
        playerToEnemyBase = closestAllyToEnemyBase;

        if (mainBase != null && enemyBase != null) {
            baseToEnemyBase = manhattan(mainBase, enemyBase);
        } else {
            baseToEnemyBase = 9999;
        }

        if (mainBase != null) {
            realBaseToEnemy = distRealUnitToEnemy(mainBase, gs.getPlayer(player), gs);
        } else {
            realBaseToEnemy = -1;
        }
    }

    private int manhattan(Unit a, Unit b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    private int countNearbyResources(Unit center, PhysicalGameState pgs, int radius) {
        int count = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                int d = Math.abs(center.getX() - u.getX()) + Math.abs(center.getY() - u.getY());
                if (d <= radius) {
                    count++;
                }
            }
        }
        return count;
    }

    private void updateUnitDistribution(int player, GameState gs) {
        playerUnits = new int[] {0, 0, 0, 0, 0, 0};
        enemyUnits = new int[] {0, 0, 0, 0, 0, 0};

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit myBase = null;
        Unit enemyMainBase = null;

        for (Unit u : pgs.getUnits()) {
            int unitIdx = -1;
            String name = u.getType().name;

            if ("Worker".equals(name)) {
                unitIdx = IDX_WORKER;
            } else if ("Light".equals(name)) {
                unitIdx = IDX_LIGHT;
            } else if ("Heavy".equals(name)) {
                unitIdx = IDX_HEAVY;
            } else if ("Ranged".equals(name)) {
                unitIdx = IDX_RANGED;
            } else if ("Base".equals(name)) {
                unitIdx = IDX_BASE;
                if (u.getPlayer() == player && myBase == null) {
                    myBase = u;
                } else if (u.getPlayer() >= 0 && u.getPlayer() != player && enemyMainBase == null) {
                    enemyMainBase = u;
                }
            } else if ("Barracks".equals(name)) {
                unitIdx = IDX_BARRACKS;
            }

            if (unitIdx < 0) {
                continue;
            }

            if (u.getPlayer() == player) {
                playerUnits[unitIdx]++;
            } else if (u.getPlayer() >= 0) {
                enemyUnits[unitIdx]++;
            }
        }

        mainBase = myBase;
        enemyBase = enemyMainBase;
    }

    private int distRealUnitToEnemy(Unit base, Player player, GameState gs) {
        if (base == null) {
            return -1;
        }

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int closestDistance = Integer.MAX_VALUE;

        for (Unit enemy : pgs.getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player.getID()) {
                continue;
            }

            int d = aStar.findDistToPositionInRange(base, enemy.getPosition(pgs), 1, gs, gs.getResourceUsage());
            if (d >= 0 && d < closestDistance) {
                closestDistance = d;
            }
        }

        return closestDistance == Integer.MAX_VALUE ? -1 : closestDistance;
    }

    // ========== LLM Strategic Consultation (GPT-5) ==========

    private void consultLLM(int player, GameState gs) {
        int currentTick = gs.getTime();
        if (currentTick - lastLLMConsultTick < LLM_CONSULT_INTERVAL) return;
        lastLLMConsultTick = currentTick;

        try {
            String prompt = buildLLMPrompt(player, gs);
            String response = queryOllamaAPI(prompt);
            if (response != null) {
                parseLLMAdvice(response);
            }
        } catch (Exception e) {
            // LLM failure is non-fatal; keep using existing bias values
            System.err.println("[ChaseBot] LLM consultation failed: " + e.getMessage());
        }
    }

    private String buildLLMPrompt(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int maxCycles = 5000;
        int resources = gs.getPlayer(player).getResources();
        String pathStatus = realBaseToEnemy >= 0
                ? "open (distance " + realBaseToEnemy + ")"
                : "blocked";

        return "You are a strategic advisor for an RTS game. Analyze this game state and recommend strategy.\n\n" +
                "Map: " + pgs.getWidth() + "x" + pgs.getHeight() +
                ", Turn: " + gs.getTime() + "/" + maxCycles + "\n" +
                "My units: Workers=" + playerUnits[IDX_WORKER] +
                ", Light=" + playerUnits[IDX_LIGHT] +
                ", Heavy=" + playerUnits[IDX_HEAVY] +
                ", Ranged=" + playerUnits[IDX_RANGED] +
                ", Bases=" + playerUnits[IDX_BASE] +
                ", Barracks=" + playerUnits[IDX_BARRACKS] + "\n" +
                "Enemy units: Workers=" + enemyUnits[IDX_WORKER] +
                ", Light=" + enemyUnits[IDX_LIGHT] +
                ", Heavy=" + enemyUnits[IDX_HEAVY] +
                ", Ranged=" + enemyUnits[IDX_RANGED] +
                ", Bases=" + enemyUnits[IDX_BASE] +
                ", Barracks=" + enemyUnits[IDX_BARRACKS] + "\n" +
                "My resources: " + resources + "\n" +
                "Closest enemy to my base: " + enemyToPlayerBase + " tiles\n" +
                "My closest unit to enemy base: " + playerToEnemyBase + " tiles\n" +
                "Path to enemy: " + pathStatus + "\n\n" +
                "Unit counters: Light(fast) beats Ranged, Heavy(tanky) beats Light, Ranged(range 3) beats Heavy.\n" +
                "Workers cost 1, Light cost 2, Heavy cost 3, Ranged cost 2. Barracks cost 5.\n\n" +
                "Respond in JSON only:\n" +
                "{\"thinking\":\"brief analysis\",\"attack_bias\":<float -3.0 to 3.0>,\"military_bias\":<float -3.0 to 3.0>,\"unit_preference\":\"light|heavy|ranged|balanced\"}\n" +
                "attack_bias: positive=attack aggressively, negative=defend and build up\n" +
                "military_bias: positive=prioritize military training, negative=prioritize economy\n" +
                "unit_preference: which unit type to prioritize building";
    }

    private String queryOllamaAPI(String prompt) throws Exception {
        URL url = new URL(OLLAMA_HOST + "/api/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setConnectTimeout(5000);
        conn.setReadTimeout(30000);

        JsonObject body = new JsonObject();
        body.addProperty("model", OLLAMA_MODEL);
        body.addProperty("prompt", prompt);
        body.addProperty("format", "json");
        body.addProperty("stream", false);

        OutputStream os = conn.getOutputStream();
        os.write(body.toString().getBytes("UTF-8"));
        os.flush();
        os.close();

        int responseCode = conn.getResponseCode();
        if (responseCode != 200) {
            conn.disconnect();
            return null;
        }

        BufferedReader reader = new BufferedReader(
                new InputStreamReader(conn.getInputStream(), "UTF-8"));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) sb.append(line);
        reader.close();
        conn.disconnect();

        JsonObject ollamaResp = JsonParser.parseString(sb.toString()).getAsJsonObject();
        return ollamaResp.has("response") ? ollamaResp.get("response").getAsString() : null;
    }

    private void parseLLMAdvice(String responseJson) {
        try {
            JsonObject advice = JsonParser.parseString(responseJson).getAsJsonObject();

            if (advice.has("attack_bias")) {
                llmAttackBias = Math.max(-3.0, Math.min(3.0,
                        advice.get("attack_bias").getAsDouble()));
            }
            if (advice.has("military_bias")) {
                llmMilitaryBias = Math.max(-3.0, Math.min(3.0,
                        advice.get("military_bias").getAsDouble()));
            }
            if (advice.has("unit_preference")) {
                String pref = advice.get("unit_preference").getAsString().toLowerCase();
                if ("light".equals(pref) || "heavy".equals(pref)
                        || "ranged".equals(pref) || "balanced".equals(pref)) {
                    llmUnitPreference = pref;
                }
            }

            System.out.println("[ChaseBot] LLM advice: attack=" + llmAttackBias +
                    " military=" + llmMilitaryBias + " units=" + llmUnitPreference);
        } catch (Exception e) {
            System.err.println("[ChaseBot] Failed to parse LLM response: " + e.getMessage());
        }
    }
}




abstract class AwareAI extends AIWithComputationBudget{

    protected PathFinding pf;
    protected ScoutPathFinding spf = new ScoutPathFinding();

    public int getnHarvest() {
        return nHarvest;
    }

    public void setnHarvest(int nHarvest) {
        this.nHarvest = nHarvest;
    }

    public int getUnitAwareness() {
        return unitAwareness;
    }

    public void setUnitAwareness(int unitAwareness) {
        this.unitAwareness = unitAwareness;
    }

    public int getTotBarracks() {
        return totBarracks;
    }

    public void setTotBarracks(int totBarracks) {
        this.totBarracks = totBarracks;
    }

    public double getStrategyPriority() {
        return strategyPriority;
    }

    public void setStrategyPriority(double strategyPriority) {
        this.strategyPriority = strategyPriority;
    }

    public int[] getPlayerUnits() {
        return playerUnits;
    }

    public void setPlayerUnits(int[] playerUnits) {
        this.playerUnits = playerUnits;
    }

    public int[] getEnemyUnits() {
        return enemyUnits;
    }

    public void setEnemyUnits(int[] enemyUnits) {
        this.enemyUnits = enemyUnits;
    }

    public int[] getUnitProduction() {
        return unitProduction;
    }

    public void setUnitProduction(int[] unitProduction) {
        this.unitProduction = unitProduction;
    }

    public int getResourceTreshold() {
        return resourceTreshold;
    }

    public void setResourceTreshold(int resourceTreshold) {
        this.resourceTreshold = resourceTreshold;
    }

    int resourceTreshold = 5;

    protected int nHarvest = 2;
    int unitAwareness = 3;
    int totBarracks = 1;
    int heavyCounter = 0;

    double strategyPriority = 0.0;

    int[] playerUnits = new int[] {0, 0, 0, 0, 0, 0};
    int[] enemyUnits = new int[] {0, 0, 0, 0, 0, 0};
    int[] unitProduction = new int[] {1, 1, 1};

    protected UnitTypeTable utt;
    UnitType workerType;
    UnitType baseType;
    UnitType barracksType;
    UnitType rangedType;
    UnitType lightType;
    UnitType heavyType;

    PlayerAction pa;

    public AwareAI(PathFinding a_pf, int timeBudget, int iterationsBudget) {
        super(timeBudget, iterationsBudget);
        pf = a_pf;
    }

    public AwareAI(PathFinding a_pf){
        super(-1, -1);
        pf = a_pf;
    }

    public void reset() {
        //TODO
    }

    public void reset(UnitTypeTable a_utt)
    {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        rangedType = utt.getUnitType("Ranged");
        lightType = utt.getUnitType("Light");
        heavyType = utt.getUnitType("Heavy");
    }

    public PlayerAction getAction(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);
        pa = new PlayerAction();

        List<Unit> workers = new ArrayList<>();

        ResourceUsage ru = new ResourceUsage();
        ResourceUsage r = gs.getResourceUsage();
        pa.setResourceUsage(r);

        for(Unit u : gs.getUnits()){
            if (u.getPlayer() != player) {
                continue;
            }

            UnitAction a = null;

            if(u.getType() == baseType && gs.getActionAssignment(u) == null)
                a = baseBehavior(u, p, gs);

            if(u.getType() == barracksType && gs.getActionAssignment(u) == null)
                a = barracksBehavior(u, p, gs);

            if(u.getType().canHarvest)
                workers.add(u);

            if(u.getType().canAttack && !u.getType().canHarvest  && gs.getActionAssignment(u) == null){
                a = meleeUnitBehavior(u, p, gs, ru);
            }

            if(a != null){
                ResourceUsage r_a = a.resourceUsage(u, pgs);

                if(pa.consistentWith(r_a, gs)){
                    ru.merge(r_a);
                    pa.addUnitAction(u, a);
                    pa.getResourceUsage().merge(r_a);
                }
            }
        }

        List<Pair<Unit, UnitAction>> w_a = workersBehavior(workers, p, gs, ru);

        if(w_a != null && !w_a.isEmpty()){
            for(Pair<Unit, UnitAction> pair : w_a){
                if(pair.m_b != null){
                    ResourceUsage r_a = pair.m_b.resourceUsage(pair.m_a, pgs);

                    if(pa.consistentWith(r_a, gs)){
                        ru.merge(r_a);
                        pa.addUnitAction(pair.m_a, pair.m_b);
                        pa.getResourceUsage().merge(r_a);
                    }
                }
            }
        }

        pa.fillWithNones(gs, player, 10);
        return pa;
    }

    int manhattanDistance(int x, int y, int x2, int y2) {
        return Math.abs(x-x2) + Math.abs(y-y2);
    }

    double sqrDistance(int x, int y, int x2, int y2) {
        int dx = x - x2;
        int dy = y - y2;

        return Math.sqrt(dx * dx + dy * dy);
    }

    public int realDistance(Unit u1, Unit u2, GameState gs) {
        AStarPathFinding aStar = new AStarPathFinding();

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int d = aStar.findDistToPositionInRange(u1, u2.getPosition(pgs), 1, gs, gs.getResourceUsage());

        if(d == -1)
            return 9999;
        else
            return d;
    }

    public int posDistance(int a, int b, int w){
        return manhattanDistance(a%w, b%w, a/w, b/w);
    }

    public UnitAction Move(Unit unit, int dest, GameState gs, ResourceUsage ru){
        UnitAction move = pf.findPath(unit, dest, gs, ru);
//        System.out.println("AStarAttak returns: " + move);
        if (move!=null && gs.isUnitActionAllowed(unit, move))
            return move;
        if(move==null){
            //TODO: still get closer with other methods
        }
        return null;
    }

    public UnitAction Train(Unit unit, UnitType type, GameState gs){
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int x = unit.getX();
        int y = unit.getY();
        int best_direction = -1;
        int best_score = -1;

        if (y>0 && gs.free(x,y-1)) {
            int score = evaluateTrain(x,y-1, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_UP;
            }
        }
        if (x<pgs.getWidth()-1 && gs.free(x+1,y)) {
            int score = evaluateTrain(x+1,y, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_RIGHT;
            }
        }
        if (y<pgs.getHeight()-1 && gs.free(x,y+1)) {
            int score = evaluateTrain(x,y+1, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_DOWN;
            }
        }
        if (x>0 && gs.free(x-1,y)) {
            int score = evaluateTrain(x-1,y, type, unit.getPlayer(), pgs);
            if (score>best_score || best_direction==-1) {
                best_score = score;
                best_direction = UnitAction.DIRECTION_LEFT;
            }
        }

        if (best_direction!=-1) {
            UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE,best_direction, type);
            if (gs.isUnitActionAllowed(unit, ua)) return ua;
        }

        return null;
    }

    public int evaluateTrain(int x, int y, UnitType type, int player, PhysicalGameState pgs) {
        int distance = 0;
        boolean first = true;

        if (type.canHarvest && playerUnits[0] < nHarvest) {
            // evaluateTrain is minus distance to closest resource
            for(Unit u:pgs.getUnits()) {
                if (u.getType().isResource) {
                    int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                    if (first || d<distance) {
                        distance = d;
                        first = false;
                    }
                }
            }
        } else {
            // evaluateTrain is minus distance to closest enemy
            for(Unit u:pgs.getUnits()) {
                if (u.getPlayer()>=0 && u.getPlayer()!=player) {
                    int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                    if (first || d<distance) {
                        distance = d;
                        first = false;
                    }
                }
            }
        }

        return -distance;
    }

    public UnitAction Build(Unit unit, UnitType type, int x, int y, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        UnitAction move = pf.findPathToAdjacentPosition(unit, x+y*pgs.getWidth(), gs, ru);
        if (move!=null) {
            if (gs.isUnitActionAllowed(unit, move)) return move;
            return null;
        }

        // build:
        UnitAction ua = null;
        if (x == unit.getX() &&
                y == unit.getY()-1) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_UP,type);
        if (x == unit.getX()+1 &&
                y == unit.getY()) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_RIGHT,type);
        if (x == unit.getX() &&
                y == unit.getY()+1) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_DOWN,type);
        if (x == unit.getX()-1 &&
                y == unit.getY()) ua = new UnitAction(UnitAction.TYPE_PRODUCE,UnitAction.DIRECTION_LEFT,type);
        if (ua!=null && gs.isUnitActionAllowed(unit, ua)) return ua;

        return null;
    }

    public int evaluateBaseBuild(Unit builder, int pos, int player, PhysicalGameState pgs){
        int distance = 0, enemyDistance = 0, resourceDistance = 0;

        int x = pos % pgs.getWidth();
        int y = pos / pgs.getWidth();
        int posDistance = Math.abs(builder.getX() - x) + Math.abs(builder.getY() - y);

        for (Unit u : pgs.getUnits()){
            if (u.getType().isResource) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if((resourceDistance == 0 || d<resourceDistance) && d > 1){
                    resourceDistance = d;
                }
            }

            if (u.getPlayer()>=0 && u.getPlayer()!=player) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if (enemyDistance == 0 || d>enemyDistance) {
                    enemyDistance = d;
                }
            }
        }

        distance = enemyDistance - resourceDistance - posDistance;

        return distance;
    }

    public int evaluateBarrackBuild(int pos, int player, PhysicalGameState pgs){
        int distance = 0, enemyDistance = 0, baseDistance = 0;

        int x = pos % pgs.getWidth();
        int y = pos / pgs.getWidth();

        if(badSpot(x, y, pgs))
            return distance;

        for (Unit u : pgs.getUnits()){
            if (u.getType() == baseType && u.getPlayer() == player) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if((baseDistance == 0 || d>baseDistance) && d > 1){
                    baseDistance = d;
                }
            }

            if (u.getPlayer()>=0 && u.getPlayer()!=player) {
                int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
                if (enemyDistance == 0 || (d>enemyDistance && pgs.getWidth()<64) || (d<enemyDistance && pgs.getWidth()>=64)) {
                    enemyDistance = d;
                }
            }
        }

        if(pgs.getWidth()<64)
            distance = enemyDistance + baseDistance;
        else
            distance = baseDistance - enemyDistance;

        return distance;
    }

    public int evaluateAttackTarget(Unit unit, Unit target, PhysicalGameState pgs){
        int d = Math.abs(unit.getX() - target.getX()) + Math.abs(unit.getY() - target.getY());
        int score = d * 10;

        // Prefer high-value and vulnerable targets.
        if (oneShots(unit, target)) score -= 20;
        score += target.getHitPoints() * 2;

        if (target.getType() == rangedType) score -= 14;
        if (target.getType() == barracksType) score -= 8;
        if (target.getType() == baseType) score -= 6;
        if (target.getType() == workerType) score -= 4;

        // Workers should not chase deep military trades.
        if (unit.getType() == workerType && target.getType().canAttack) score += 8;

        // Light units are better at collapsing ranged targets.
        if (unit.getType() == lightType && target.getType() == rangedType) score -= 8;

        // Heavy units are better used against static structures.
        if (unit.getType() == heavyType && (target.getType() == baseType || target.getType() == barracksType)) {
            score -= 8;
        }

        return score;
    }

    public boolean oneShots(Unit unit, Unit target){
        return target.getHitPoints()<=unit.getMaxDamage();
    }



    public List<Integer> GetFreePositionsAround(Unit u, PhysicalGameState pgs){
        List<Integer> positions = new ArrayList<>();
        int[] diffX = new int[] {-1, 0, +1, -1, +1, -1, 0, +1};
        int[] diffY = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};

        if(u.getX() == 0){
            diffX = new int[] {0, +1, +2, +1, +2, 0, +1, +2};
            diffY = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
        }
        else if(u.getX() == pgs.getWidth() - 1){
            diffX = new int[] {0, -1, -2, -1, -2, 0, -1, -2};
            diffY = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
        }
        else if(u.getY() == 0){
            diffX = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
            diffY = new int[] {0, +1, +2, +1, +2, 0, +1, +2};
        }
        else if(u.getY() == pgs.getHeight() - 1){
            diffX = new int[] {-1, -1, -1, 0, 0, +1, +1, +1};
            diffY = new int[] {0, -1, -2, -1, -2, 0, -1, -2};
        }

        for(int i = 0; i < diffX.length; i++){
            int x = u.getX() + diffX[i], y = u.getY() + diffY[i];

            if(x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight())
                continue;

            if(pgs.getUnitAt(x, y) == null && pgs.getTerrain(x, y) != PhysicalGameState.TERRAIN_WALL && !AdjacentToResource(x, y, pgs)){
                positions.add(x + y * pgs.getWidth());
            }
        }

        return positions;
    }

    public boolean AdjacentToResource(int x, int y, PhysicalGameState pgs){
        int[] diffX = new int[] {-1, 0, +1, 0};
        int[] diffY = new int[] {0, -1, 0, +1,};

        for(int i = 0; i < diffX.length; i++){
            int nx = x + diffX[i], ny = y + diffY[i];

            if(nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight())
                continue;

            Unit aU = pgs.getUnitAt(nx, ny);

            if(aU != null){
                if(aU.getType().isResource)
                    return true;
            }
        }

        return false;
    }

    //Checks if the selected spot is either near a Resource or on a chokepoint
    public boolean badSpot(int x, int y, PhysicalGameState pgs){
        int[] diffX = new int[] {-1, 0, +1, 0, -1, -1, +1, +1};
        int[] diffY = new int[] {0, -1, 0, +1, -1, +1, -1, +1};

        List<Integer> walls = new ArrayList<>();

        for(int i = 0; i < diffX.length; i++){
            int nx = x + diffX[i], ny = y + diffY[i];

            if(nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight())
                continue;

            Unit aU = pgs.getUnitAt(nx, ny);

            if(aU != null){
                if(aU.getType().isResource)
                    return true;
            }

            if(pgs.getTerrain(nx,ny) == PhysicalGameState.TERRAIN_WALL){
                walls.add(nx + ny * pgs.getWidth());
                if(walls.size() > 1){
                    int wall = walls.get(walls.size()-1);
                    for (int other: walls) {
                        if(posDistance(wall, other, pgs.getWidth()) > 1)
                            return true;
                    }
                }
            }

        }

        return false;
    }

    public boolean isChokepoint(int x, int y, PhysicalGameState pgs){
        int[] diffX = new int[] {-1, 0, +1, 0, -1, -1, +1, +1};
        int[] diffY = new int[] {0, -1, 0, +1, -1, +1, -1, +1};

        List<Integer> walls = new ArrayList<>();

        for(int i = 0; i < diffX.length; i++){
            int nx = x + diffX[i], ny = y + diffY[i];

            if(nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight())
                continue;

            if(pgs.getTerrain(nx,ny) == PhysicalGameState.TERRAIN_WALL){
                walls.add(nx + ny * pgs.getWidth());
                if(walls.size() > 1){
                    int wall = walls.get(walls.size()-1);
                    for (int other: walls) {
                        if(posDistance(wall, other, pgs.getWidth()) > 1)
                            return true;
                    }
                }
            }

        }

        return false;

    }

    public boolean isPlayerPredominant(Unit unit, int width, int height, GameState gs) {
        boolean mine;

        int allies = 0, enemies = 0;

        for (Unit u : gs.getUnits()) {
            if ((Math.abs(u.getX() - unit.getX()) <= width && Math.abs(u.getY() - unit.getY()) <= height)) {
                if(u.getPlayer() == unit.getPlayer())
                    allies++;

                if(u.getPlayer() != unit.getPlayer() && u.getPlayer() >= 0)
                    enemies++;
            }
        }

        if(enemies > allies)
            mine = false;
        else
            mine = true;

        return mine;
    }

    public boolean inRange(Unit u1, Unit u2, int range){
        return ((Math.abs(u1.getX() - u2.getX()) <= range && Math.abs(u1.getY() - u2.getY()) <= range));
    }

    public Unit GetClosestBase(Unit u, PhysicalGameState pgs){
        Unit closestBase = null;
        int distance = 0;

        for(Unit other : pgs.getUnits()){
            if(other.getType() == baseType && other.getPlayer() == u.getPlayer()){
                int d = Math.abs(u.getX() - other.getX()) + Math.abs(u.getY() - other.getY());
                if(closestBase == null || d < distance){
                    closestBase = other;
                    distance = d;
                }
            }
        }

        return closestBase;
    }

    public UnitAction Harvest(Unit unit, Unit target, Unit base, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        if (unit.getResources()==0) {
            if (target == null) return null;
            // go get resources:
            UnitAction move = pf.findPathToAdjacentPosition(unit, target.getX()+target.getY()*gs.getPhysicalGameState().getWidth(), gs, ru);
            if (move!=null) {
                if (gs.isUnitActionAllowed(unit, move)) return move;
                return null;
            }

            // harvest:
            if (target.getX() == unit.getX() &&
                    target.getY() == unit.getY()-1) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_UP);
            if (target.getX() == unit.getX()+1 &&
                    target.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_RIGHT);
            if (target.getX() == unit.getX() &&
                    target.getY() == unit.getY()+1) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_DOWN);
            if (target.getX() == unit.getX()-1 &&
                    target.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_HARVEST,UnitAction.DIRECTION_LEFT);
        } else {
            // return resources:
            if (base == null) return null;
            UnitAction move = pf.findPathToAdjacentPosition(unit, base.getX()+base.getY()*gs.getPhysicalGameState().getWidth(), gs, ru);
            if (move!=null) {
                if (gs.isUnitActionAllowed(unit, move)) return move;
                return null;
            }

            // harvest:
            if (base.getX() == unit.getX() &&
                    base.getY() == unit.getY()-1) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_UP);
            if (base.getX() == unit.getX()+1 &&
                    base.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_RIGHT);
            if (base.getX() == unit.getX() &&
                    base.getY() == unit.getY()+1) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_DOWN);
            if (base.getX() == unit.getX()-1 &&
                    base.getY() == unit.getY()) return new UnitAction(UnitAction.TYPE_RETURN,UnitAction.DIRECTION_LEFT);
        }
        return null;
    }

    public UnitAction Attack(Unit unit, Unit target, GameState gs, ResourceUsage ru) {
        if(target == null)
            return null;

        if (inAttackRange(unit, target) && !willEscapeAttack(unit, target, gs)) {
            return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,target.getX(),target.getY());
        } else if(futureInAttackRange(unit, target, gs)){
            return new UnitAction(UnitAction.TYPE_NONE, 1);
        }  else {
            UnitAction move = pf.findPathToPositionInRange(unit, target.getX()+target.getY()*gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
            if (move!=null && gs.isUnitActionAllowed(unit, move))
                return move;

            if(move == null)
                return Approach(unit, target, gs, ru);

            return null;
        }
    }

    protected UnitAction rangedBehavior(Unit ranged, Player p, GameState gs, ResourceUsage ru){
        Unit closestEnemy = null;
        Unit bestTarget = null;
        int closestDistance = 0;
        int bestTargetScore = Integer.MAX_VALUE;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = manhattanDistance(ranged.getX(), ranged.getY(), u2.getX(), u2.getY());
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }

                int t = evaluateAttackTarget(ranged, u2, pgs);
                if (bestTarget==null || t<bestTargetScore) {
                    bestTarget = u2;
                    bestTargetScore = t;
                }
            }
        }
        if (bestTarget == null || closestEnemy == null) {
            return Attack(ranged, null, gs, ru);
        }

        int enemyThreatRange = closestEnemy.getType().canAttack ? closestEnemy.getAttackRange() : 0;
        boolean threatened = closestEnemy.getType().canAttack && closestDistance <= enemyThreatRange + 1;

        if (threatened && !oneShots(ranged, bestTarget)) {
            UnitAction kite = kiteFromEnemy(ranged, closestEnemy, bestTarget, gs, ru);
            if (kite != null) {
                return kite;
            }
        }

        return Attack(ranged, bestTarget, gs, ru);
    }

    private UnitAction kiteFromEnemy(Unit ranged, Unit enemy, Unit preferredTarget, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Integer> positions = PositionsInRadius(ranged, 1, pgs);
        if (positions.isEmpty()) return null;

        int currentEnemyDistance = manhattanDistance(ranged.getX(), ranged.getY(), enemy.getX(), enemy.getY());
        int bestScore = Integer.MIN_VALUE;
        int bestPos = -1;

        for (int pos : positions) {
            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();
            int enemyDistance = manhattanDistance(x, y, enemy.getX(), enemy.getY());
            int targetDistance = manhattanDistance(x, y, preferredTarget.getX(), preferredTarget.getY());

            if (enemyDistance <= currentEnemyDistance) continue;
            if (targetDistance > ranged.getAttackRange() + 2) continue;
            if (!pf.pathExists(ranged, pos, gs, ru)) continue;

            int score = enemyDistance * 4 - targetDistance;
            if (score > bestScore) {
                bestScore = score;
                bestPos = pos;
            }
        }

        if (bestPos >= 0) {
            return Move(ranged, bestPos, gs, ru);
        }
        return null;
    }

    protected UnitAction aggroMeleeBehevior(Unit unit, Player p, GameState gs, ResourceUsage ru){
        Unit closestTarget = null;
        Unit favoriteTarget = null;
        int closestDistance = 0;
        int favoriteDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(unit, u2, pgs);
                if (closestTarget==null || d<closestDistance) {
                    closestTarget = u2;
                    closestDistance = d;
                }
                if(((u2.getType()==barracksType || u2.getType()==baseType) && unit.getType()==heavyType) ||
                        (u2.getType()==rangedType && unit.getType()==lightType)){
                    if (favoriteTarget==null || d<favoriteDistance) {
                        favoriteTarget = u2;
                        favoriteDistance = d;
                    }
                }
            }
        }
        if(favoriteTarget==null && closestTarget!=null)
            return Attack(unit,closestTarget,gs,ru);
        if(favoriteTarget==closestTarget)
            return Attack(unit, favoriteTarget, gs, ru);
        if(closestTarget!=null && closestDistance < 4)
            return Attack(unit,closestTarget,gs,ru);
        if(favoriteTarget!=null)
            return Attack(unit, favoriteTarget, gs, ru);
        return Attack(unit,null,gs,ru);
    }

    int combatScore(Unit u, Unit e, PhysicalGameState pgs) {
        int score = manhattanDistance(u.getX(), u.getY(), e.getX(), e.getY());

        if ((u.getType() == rangedType || u.getType() == lightType) && e.getType() == rangedType && pgs.getWidth() > 9)
            score -= 2;

        if (pgs.getWidth() >= 16 && (u.getType() == heavyType || u.getType() == rangedType)
                && (e.getType() == barracksType)) //todo - remove? todo base
            score -= pgs.getWidth();

        return score;
    }

    public UnitAction Scout(Unit unit, Unit target, GameState gs, ResourceUsage ru){
        if(target == null)
            return null;
        if (inAttackRange(unit, target) && !willEscapeAttack(unit, target, gs)) {
            return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,target.getX(),target.getY());
        } else if(futureInAttackRange(unit, target, gs)){
            return new UnitAction(UnitAction.TYPE_NONE, 1);
        }
        else{
            UnitAction move = spf.findPathToPositionInRange(unit, target.getX()+target.getY()*gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
            if (move!=null && gs.isUnitActionAllowed(unit, move)){
                if(move.getType() == UnitAction.TYPE_HARVEST && isChokepoint(unit.getX(), unit.getY(), gs.getPhysicalGameState()) && !isPlayerPredominant(unit, 5, 5, gs)){
                    return null;
                }
                else
                    return move;
            }


            if(move == null)
                return Approach(unit, target, gs, ru);

            return null;
        }
    }

    public UnitAction Idle(Unit unit, GameState gs, ResourceUsage ru){
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if (!unit.getType().canAttack)
            return null;

        for(Unit target:pgs.getUnits()) {
            if (target.getPlayer()!=-1 && target.getPlayer()!=unit.getPlayer()) {
                if (inAttackRange(unit, target)) {
                    return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,target.getX(),target.getY());
                }
            }
        }
        return null;
    }

    public UnitAction Approach(Unit unit, Unit target, GameState gs, ResourceUsage ru){
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Integer> positions = PositionsInRadius(unit, 2, pgs);
        int pos = 0;
        int closestDistance = 0;
        int modifier = 0;

        if(target.getType().canAttack){
            if(target.getType()==rangedType)
                modifier = 4;
            else
                modifier = 1;
        }

        int currentDistance = Math.abs(target.getX() - unit.getX()) + Math.abs(target.getY() - unit.getY()) - modifier;
        boolean found = false;

        if(positions.isEmpty())
            return null;

        if(target.getType().canAttack){
            if(target.getType()==rangedType)
                modifier = 4;
            else
                modifier = 1;
        }

        for(int p : positions){
            if(pf.pathExists(unit, p, gs, ru)){
                int x = p % pgs.getWidth();
                int y = p/ pgs.getWidth();
                int d = Math.abs(target.getX() - x) + Math.abs(target.getY() - y) - modifier;

                if((!found||d<closestDistance) && d < currentDistance && d > 0){
                    closestDistance = d;
                    pos = p;
                    found = true;
                }
            }
        }

        if(found){
            UnitAction move = pf.findPath(unit, pos, gs, ru);
            if(move!=null && gs.isUnitActionAllowed(unit, move))
                return move;
        }

        return null;
    }

    public List<Integer> PositionsInRadius(Unit u, int radius, PhysicalGameState pgs){
        List<Integer> positions = new ArrayList<>();
        int[] diffX = new int[] {-1, 0, +1, 0};
        int[] diffY = new int[] {0, -1, 0, +1,};

        if(radius == 2){
            diffX = new int[] {-1, 0, +1, 0, -1, -1, +1, +1, -2, 0, +2, 0};
            diffY = new int[] {0, -1, 0, +1, -1, +1, -1, +1, 0, -2, 0, +2};
        }

        for(int i = 0; i < diffX.length; i++){
            int x = u.getX() + diffX[i], y = u.getY() + diffY[i];

            if(x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight())
                continue;

            if(pgs.getUnitAt(x, y) == null && pgs.getTerrain(x, y) != PhysicalGameState.TERRAIN_WALL && !AdjacentToResource(x, y, pgs)){
                positions.add(x + y * pgs.getWidth());
            }
        }

        return positions;
    }

    public boolean Between(Unit unit, Unit first, Unit last){
        int minX = Math.min(first.getX(), last.getX());
        int maxX = Math.max(first.getX(), last.getX());

        int minY = Math.min(first.getY(), last.getY());
        int maxY = Math.max(first.getY(), last.getY());

        return (minX <= unit.getX() && unit.getX() <= maxX && minY <= unit.getY() && unit.getY() <= maxY);
    }

    public List<Unit> getAdjacentUnits(Unit u, PhysicalGameState pgs, boolean friendly){
        List<Unit> units = new ArrayList<>();
        int[] diffX = new int[] {-1, 0, +1, 0};
        int[] diffY = new int[] {0, -1, 0, +1,};

        for(int i = 0; i < diffX.length; i++){
            int x = u.getX() + diffX[i], y = u.getY() + diffY[i];

            if(x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight())
                continue;

            Unit u2 = pgs.getUnitAt(x, y);

            if(u2 != null){
                if(u2.getType().canAttack && u2.getPlayer() == u.getPlayer() && friendly)
                    units.add(u2);

                if(u2.getType().canAttack && u2.getPlayer() != u.getPlayer() && !friendly)
                    units.add(u2);
            }
        }

        return units;
    }

    public boolean conflictingMove(Unit unit, UnitAction unitAction, List<Pair<Unit, UnitAction>> list, PhysicalGameState pgs){
        List<Unit> adjacentUnits = getAdjacentUnits(unit, pgs, true);

        if(unitAction == null)
            return false;

        if(unitAction.getType()!=UnitAction.TYPE_MOVE)
            return false;

        if(!adjacentUnits.isEmpty()){
            for (Unit au: adjacentUnits) {
                for(Pair<Unit, UnitAction> pair : list){
                    if(pair.m_b==null)
                        continue;

                    if(pair.m_a == au && pair.m_b.getType() == UnitAction.TYPE_MOVE)
                        return true;
                }
            }
        }

        return false;
    }

    int futurePos(Unit unit, GameState gs){
        int x = unit.getX(), y = unit.getY();
        PhysicalGameState pgs = gs.getPhysicalGameState();

        UnitActionAssignment aa = gs.getActionAssignment(unit);
        if (aa == null){
            return x + y * pgs.getWidth();
        }

        else if (aa.action.getType() == UnitAction.TYPE_MOVE){
            int nx = x;
            int ny = y;

            switch (aa.action.getDirection()) {
                case UnitAction.DIRECTION_DOWN:
                    ny = (ny == pgs.getHeight()- 1) ? ny : ny + 1;
                    break;
                case UnitAction.DIRECTION_UP:
                    ny = (ny == 0) ? ny : ny - 1;
                    break;
                case UnitAction.DIRECTION_RIGHT:
                    nx = (nx == pgs.getWidth() - 1) ? nx : nx + 1;
                    break;
                case UnitAction.DIRECTION_LEFT:
                    nx = (nx == 0) ? nx : nx - 1;
                    break;
                default:
                    break;
            }

            return nx + ny * pgs.getWidth();
        }

        return x + y * pgs.getWidth();
    }

    boolean futureInAttackRange(Unit u1, Unit u2, GameState gs){
        int futurePos = futurePos(u2, gs);

        int w = gs.getPhysicalGameState().getWidth();
        int x2 = futurePos%w;
        int y2 = futurePos/w;

        boolean inRange = sqrDistance(u1.getX(), u1.getY(), x2, y2) <= u1.getAttackRange();

        return inRange;
    }

    boolean inAttackRange(Unit u1, Unit u2) {
        return sqrDistance(u1.getX(), u1.getY(), u2.getX(), u2.getY()) <= u1.getAttackRange();
    }

    boolean willEscapeAttack(Unit attacker, Unit runner, GameState gs) {
        UnitActionAssignment aa = gs.getActionAssignment(runner);
        if (aa == null)
            return false;
        if (aa.action.getType() != UnitAction.TYPE_MOVE)
            return false;
        int eta = aa.action.ETA(runner) - (gs.getTime() - aa.time);
        return eta <= attacker.getAttackTime();
    }

    protected abstract UnitAction workerBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru);
    protected abstract UnitAction barracksBehavior(Unit u, Player p, GameState gs);
    protected abstract UnitAction baseBehavior(Unit u, Player p, GameState gs);
    protected abstract UnitAction meleeUnitBehavior(Unit u, Player p, GameState gs, ResourceUsage ru);
    protected abstract UnitAction scoutBehavior(Unit scout, Player p, GameState gs, ResourceUsage ru);
    protected abstract UnitAction builderBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru);
    protected abstract List<Pair<Unit, UnitAction>> workersBehavior(List<Unit> workers, Player p, GameState gs, ResourceUsage ru);
}




class AWorkDefense extends AwareAI{
    public AWorkDefense(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public AWorkDefense(UnitTypeTable a_utt){
        this(a_utt, new AStarPathFinding());
    }

    @Override
    public AI clone() {
        return new AWorkDefense(utt, pf);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return null;
    }

    @Override
    protected UnitAction workerBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestBase = null;
        Unit closestResource = null;
        int closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestResource==null || d<closestDistance) {
                    closestResource = u2;
                    closestDistance = d;
                }
            }
        }
        closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestBase==null || d<closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }
        }
        if(closestBase!=null||closestResource!=null)
            return Harvest(worker, closestResource, closestBase, gs, ru);

        return scoutBehavior(worker, p, gs, ru);
    }

    @Override
    protected UnitAction barracksBehavior(Unit u, Player p, GameState gs) {
        int value = 0;
        UnitType type = lightType;
        UnitType[] types = new UnitType[] {lightType, heavyType, rangedType};
        for(int i = 0; i < unitProduction.length; i++){
            if(unitProduction[i] > value){
                value = unitProduction[i];
                type = types[i];
            }
        }
        if (p.getResources()>=type.cost)
            return Train(u, type, gs);
        return null;
    }

    @Override
    protected UnitAction baseBehavior(Unit u, Player p, GameState gs) {
        if (p.getResources()>=workerType.cost)
            return Train(u, workerType, gs);
        return null;
    }

    @Override
    protected UnitAction meleeUnitBehavior(Unit u, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit closestMeleeEnemy = null;
        int closestDistance = 0;
        int enemyDistance = 0;
        int mybase = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        int threshold = Math.max(pgs.getHeight(), pgs.getWidth());

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(u, u2, pgs);
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
            else if(u2.getPlayer()==p.getID() && u2.getType() == baseType){
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if(mybase == 0 || d < mybase)
                    mybase = d;
            }
        }
        if (closestEnemy!=null && (closestDistance < threshold/2 || mybase < threshold/2)) {
            return Attack(u,closestEnemy,gs,ru);
        }
        else
        {
            return Attack(u,null,gs,ru);
        }
    }

    @Override
    protected UnitAction scoutBehavior(Unit scout, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit base = null;
        int closestEnemyDistance = 0;
        int closestBaseDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        int threshold = Math.max(pgs.getHeight(), pgs.getWidth());

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(scout, u2, pgs);
                if (closestEnemy==null || d<closestEnemyDistance) {
                    closestEnemy = u2;
                    closestEnemyDistance = d;
                }
            }
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - scout.getX()) + Math.abs(u2.getY() - scout.getY());
                if (base==null || d<closestBaseDistance) {
                    base = u2;
                    closestBaseDistance = d;
                }
            }
        }

        if(closestEnemy!=null && (closestEnemyDistance < threshold/2 || closestBaseDistance < threshold/2)){
            if(pf.findPathToAdjacentPosition(scout, closestEnemy.getPosition(pgs), gs, ru)!=null
                    || manhattanDistance(scout.getX(), scout.getY(), closestEnemy.getX(), closestEnemy.getY()) == 1){
                if(isChokepoint(scout.getX(), scout.getY(), pgs)){
                    //might make them stall regardless, could look into it
                    if(isPlayerPredominant(scout, 5, 5, gs))
                        return Attack(scout, closestEnemy, gs, ru);
                    else
                        return Idle(scout, gs, ru);
                }
                else
                    return Attack(scout, closestEnemy, gs, ru);
            }
            else{
                if(scout.getResources() == 0)
                    return Scout(scout,closestEnemy,gs,ru);
                else if (base!=null){
                    if(pf.findPathToAdjacentPosition(scout, base.getPosition(pgs), gs, ru)!=null
                            || manhattanDistance(scout.getX(), scout.getY(), base.getX(), base.getY()) == 1)
                        return Harvest(scout, null, base, gs, ru);
                    else
                        return Scout(scout,closestEnemy,gs,ru);
                }
            }
        }else
        {
            return Attack(scout,null,gs,ru);
        }

        return Scout(scout,null,gs,ru);
    }

    @Override
    protected UnitAction builderBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if(playerUnits[4] == 0 && p.getResources() >= baseType.cost){
            int highscore = 0, pos = 0;

            List<Integer> positions = GetFreePositionsAround(worker, pgs);

            if(positions.isEmpty())
                return workerBehavior(worker, p, gs, ru);

            for(int i : positions){
                int score = evaluateBaseBuild(worker, i, p.getID(), pgs);
                if(score > highscore){
                    highscore = score;
                    pos = i;
                }
            }

            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();

            return Build(worker, baseType, x, y, gs, ru);
        }

        return workerBehavior(worker, p, gs, ru);
    }

    @Override
    protected List<Pair<Unit, UnitAction>> workersBehavior(List<Unit> workers, Player p, GameState gs, ResourceUsage ru) {
        int nbases = 0;
        int nbarracks = 0;
        int resourcesUsed = 0;
        Unit hw;
        UnitAction unitAction;
        List<Unit> harvestWorkers = new LinkedList<>();
        List<Unit> freeWorkers = new LinkedList<>(workers);
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Pair<Unit, UnitAction>> list = new ArrayList<>();

        if (workers.isEmpty()) return list;

        nbases = playerUnits[4];
        nbarracks = playerUnits[5];

        List<Integer> reservedPositions = new LinkedList<>();
        if (nbases==0 && !freeWorkers.isEmpty()) {
            // build a base:
            if (p.getResources()>=baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                //buildIfNotAlreadyBuilding(u,baseType,u.getX(),u.getY(),reservedPositions,p,pgs);
                if(gs.getActionAssignment(u) == null){
                    resourcesUsed+=baseType.cost;
                    unitAction = builderBehavior(u, p, gs, ru);

                    if(conflictingMove(u, unitAction, list, pgs))
                        unitAction = Idle(u, gs, ru);

                    list.add(new Pair<>(u, unitAction));
                }
            }
        }

        if (freeWorkers.size()>0){
            for(int i = 1; i <=nHarvest; i++){
                if(!freeWorkers.isEmpty()){
                    hw = freeWorkers.remove(0);
                    harvestWorkers.add(hw);
                }
            }
        }

        // harvest with the harvest worker:
        for (Unit harvestWorker : harvestWorkers) {
            if(gs.getActionAssignment(harvestWorker) != null)
                continue;

            Unit closestBase = null;
            Unit closestResource = null;
            Unit closestEnemy = null;

            int closestResourceDistance = 0;
            int closestBaseDistance = 0;
            int closestEnemyDistance = 0;

            for(Unit u2:pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestResource==null || d<closestResourceDistance) {
                        closestResource = u2;
                        closestResourceDistance = d;
                    }
                }

                if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestBase==null || d<closestBaseDistance) {
                        closestBase = u2;
                        closestBaseDistance = d;
                    }
                }

                if(u2.getPlayer() >= 0 && u2.getPlayer()!=harvestWorker.getPlayer()){
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestEnemy==null || d<closestEnemyDistance) {
                        closestEnemy = u2;
                        closestEnemyDistance = d;
                    }
                }

            }

            //if you are close to a base and an enemy is close to that, attack him
            if(closestEnemy != null && closestEnemyDistance <= 2 && closestBaseDistance <= 2)
                list.add(new Pair<>(harvestWorker, Attack(harvestWorker, closestEnemy, gs, ru)));

            else if (closestResource!=null && closestBase!=null) {
                if(closestEnemy!=null){
                    if(Between(closestEnemy, harvestWorker, closestResource) && harvestWorker.getResources()<1){
                        //this could be changed directly to Attack
                        unitAction = scoutBehavior(harvestWorker, p, gs, ru);
                    }
                    else
                        unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);
                }
                else
                    unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else if(closestBase!=null && harvestWorker.getResources()>0){
                unitAction = Harvest(harvestWorker, null, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else
                freeWorkers.add(harvestWorker);
        }

        for(Unit u:freeWorkers){
            if(gs.getActionAssignment(u) == null){
                unitAction = scoutBehavior(u, p, gs, ru);

                if(conflictingMove(u, unitAction, list, pgs))
                    unitAction = Idle(u, gs, ru);

                list.add(new Pair<>(u, unitAction));
            }
        }

        return list;
    }
}



class AWorkRush extends AwareAI{
    public AWorkRush(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public AWorkRush(UnitTypeTable a_utt){
        this(a_utt, new AStarPathFinding());
    }

    @Override
    public AI clone() {
        return new AWorkRush(utt, pf);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return null;
    }

    @Override
    protected UnitAction workerBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestBase = null;
        Unit closestResource = null;
        int closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestResource==null || d<closestDistance) {
                    closestResource = u2;
                    closestDistance = d;
                }
            }
        }
        closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestBase==null || d<closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }
        }

        if(closestBase!=null||closestResource!=null)
            return Harvest(worker, closestResource, closestBase, gs, ru);

        return scoutBehavior(worker, p, gs, ru);
    }

    @Override
    protected UnitAction barracksBehavior(Unit u, Player p, GameState gs) {
        int value = 0;
        UnitType type = lightType;
        UnitType[] types = new UnitType[] {lightType, heavyType, rangedType};
        for(int i = 0; i < unitProduction.length; i++){
            if(unitProduction[i] > value){
                value = unitProduction[i];
                type = types[i];
            }
        }
        if (p.getResources()>=type.cost)
            return Train(u, type, gs);
        return null;
    }

    @Override
    protected UnitAction baseBehavior(Unit u, Player p, GameState gs) {
        if (p.getResources()>=workerType.cost)
            return Train(u, workerType, gs);
        return null;
    }

    @Override
    protected UnitAction meleeUnitBehavior(Unit u, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        int closestDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        //if(pgs.getWidth() >= 16 && u.getType()==heavyType) return heavyUnitBehavior(u, p, gs, ru);

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(u, u2, pgs);
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
        }
        if (closestEnemy!=null) {
            return Attack(u,closestEnemy,gs,ru);
        }
        else
        {
            return Attack(u,null,gs,ru);
        }
    }

    protected UnitAction heavyUnitBehavior(Unit heavy, Player p, GameState gs, ResourceUsage ru){
        Unit closestEnemy = null;
        Unit closestBuilding = null;
        int closestDistance = 0;
        int buildingDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(heavy, u2, pgs);
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }

                if(u2.getType()==barracksType || u2.getType()==baseType){
                    int l = evaluateAttackTarget(heavy, u2, pgs);
                    if (closestBuilding==null || l<buildingDistance) {
                        closestBuilding = u2;
                        buildingDistance = l;
                    }
                }
            }
        }

        if(closestDistance!=0){
            closestEnemy = closestBuilding;
        }

        if (closestEnemy!=null) {
            return Attack(heavy,closestEnemy,gs,ru);
        }
        else
        {
            return Attack(heavy,null,gs,ru);
        }
    }

    @Override
    protected UnitAction scoutBehavior(Unit scout, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit base = null;
        int closestEnemyDistance = 0;
        int closestBaseDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(scout, u2, pgs);
                if (closestEnemy==null || d<closestEnemyDistance) {
                    closestEnemy = u2;
                    closestEnemyDistance = d;
                }
            }
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - scout.getX()) + Math.abs(u2.getY() - scout.getY());
                if (base==null || d<closestBaseDistance) {
                    base = u2;
                    closestBaseDistance = d;
                }
            }
        }

        if(closestEnemy!=null){
            if(pf.findPathToAdjacentPosition(scout, closestEnemy.getPosition(pgs), gs, ru)!=null
                    || manhattanDistance(scout.getX(), scout.getY(), closestEnemy.getX(), closestEnemy.getY()) == 1){
                if(isChokepoint(scout.getX(), scout.getY(), pgs)){
                    if(isPlayerPredominant(scout, 5, 5, gs))
                        return Attack(scout, closestEnemy, gs, ru);
                    else
                        return Idle(scout, gs, ru);
                }
                else
                    return Attack(scout, closestEnemy, gs, ru);
            }
            else{
                if(scout.getResources() == 0)
                    return Scout(scout,closestEnemy,gs,ru);
                else if (base!=null){
                    if(pf.findPathToAdjacentPosition(scout, base.getPosition(pgs), gs, ru)!=null
                            || manhattanDistance(scout.getX(), scout.getY(), base.getX(), base.getY()) == 1)
                        return Harvest(scout, null, base, gs, ru);
                    else
                        return Scout(scout,closestEnemy,gs,ru);
                }
            }
        }
        return Scout(scout,null,gs,ru);
    }

    @Override
    protected UnitAction builderBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if(playerUnits[4] == 0 && p.getResources() >= baseType.cost){
            int highscore = 0, pos = 0;

            List<Integer> positions = GetFreePositionsAround(worker, pgs);

            if(positions.isEmpty())
                return workerBehavior(worker, p, gs, ru);

            for(int i : positions){
                int score = evaluateBaseBuild(worker, i, p.getID(), pgs);
                if(score > highscore){
                    highscore = score;
                    pos = i;
                }
            }

            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();

            return Build(worker, baseType, x, y, gs, ru);
        }

        return workerBehavior(worker, p, gs, ru);
    }

    @Override
    protected List<Pair<Unit, UnitAction>> workersBehavior(List<Unit> workers, Player p, GameState gs, ResourceUsage ru) {
        int nbases = 0;
        int nbarracks = 0;
        int resourcesUsed = 0;
        Unit hw;
        UnitAction unitAction;
        List<Unit> harvestWorkers = new LinkedList<>();
        List<Unit> freeWorkers = new LinkedList<>(workers);
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Pair<Unit, UnitAction>> list = new ArrayList<>();

        if (workers.isEmpty()) return list;

        nbases = playerUnits[4];
        nbarracks = playerUnits[5];

        List<Integer> reservedPositions = new LinkedList<>();
        if (nbases==0 && !freeWorkers.isEmpty()) {
            // build a base:
            if (p.getResources()>=baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                //buildIfNotAlreadyBuilding(u,baseType,u.getX(),u.getY(),reservedPositions,p,pgs);
                if(gs.getActionAssignment(u) == null){
                    resourcesUsed+=baseType.cost;
                    unitAction = builderBehavior(u, p, gs, ru);

                    if(conflictingMove(u, unitAction, list, pgs))
                        unitAction = Idle(u, gs, ru);

                    list.add(new Pair<>(u, unitAction));
                }
            }
        }

        if (freeWorkers.size()>0){
            for(int i = 1; i <=nHarvest; i++){
                if(!freeWorkers.isEmpty()){
                    hw = freeWorkers.remove(0);
                    harvestWorkers.add(hw);
                }
            }
        }

        // harvest with the harvest worker:
        for (Unit harvestWorker : harvestWorkers) {
            if(gs.getActionAssignment(harvestWorker) != null)
                continue;

            Unit closestBase = null;
            Unit closestResource = null;
            Unit closestEnemy = null;

            int closestResourceDistance = 0;
            int closestBaseDistance = 0;
            int closestEnemyDistance = 0;

            for(Unit u2:pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestResource==null || d<closestResourceDistance) {
                        closestResource = u2;
                        closestResourceDistance = d;
                    }
                }

                if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestBase==null || d<closestBaseDistance) {
                        closestBase = u2;
                        closestBaseDistance = d;
                    }
                }

                if(u2.getPlayer() >= 0 && u2.getPlayer()!=harvestWorker.getPlayer()){
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestEnemy==null || d<closestEnemyDistance) {
                        closestEnemy = u2;
                        closestEnemyDistance = d;
                    }
                }
            }

            //if you are close to a base and an enemy is close to that, attack him
            if(closestEnemy != null && closestEnemyDistance <= 2 && closestBaseDistance <= 2)
                list.add(new Pair<>(harvestWorker, Attack(harvestWorker, closestEnemy, gs, ru)));

            else if (closestResource!=null && closestBase!=null) {
                if(closestEnemy!=null){
                    if(Between(closestEnemy, harvestWorker, closestResource) && harvestWorker.getResources()<1){
                        //this could be changed directly to Attack
                        unitAction = scoutBehavior(harvestWorker, p, gs, ru);
                    }
                    else
                        unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);
                }
                else
                    unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else if(closestBase!=null && harvestWorker.getResources()>0){
                unitAction = Harvest(harvestWorker, null, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else
                freeWorkers.add(harvestWorker);
        }

        for(Unit u:freeWorkers){
            if(gs.getActionAssignment(u) == null){
                unitAction = scoutBehavior(u, p, gs, ru);

                if(conflictingMove(u, unitAction, list, pgs))
                    unitAction = Idle(u, gs, ru);

                list.add(new Pair<>(u, unitAction));
            }
        }
        return list;
    }

}



class AMilDefense extends AwareAI{
    Random r = new Random();

    public AMilDefense(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public AMilDefense(UnitTypeTable a_utt){
        this(a_utt, new AStarPathFinding());
    }

    @Override
    public AI clone() {
        return new AMilDefense(utt, pf);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return null;
    }

    @Override
    protected UnitAction workerBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestBase = null;
        Unit closestResource = null;
        int closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestResource==null || d<closestDistance) {
                    closestResource = u2;
                    closestDistance = d;
                }
            }
        }
        closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestBase==null || d<closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }
        }

        if(closestBase!=null||closestResource!=null)
            return Harvest(worker, closestResource, closestBase, gs, ru);

        return scoutBehavior(worker, p, gs, ru);
    }

    @Override
    protected UnitAction barracksBehavior(Unit u, Player p, GameState gs) {
        int unit = 0, value = 0;
        UnitType type = heavyType;
        UnitType[] types = new UnitType[] {lightType, heavyType, rangedType};
        for(int i = 0; i < unitProduction.length; i++){
            if(unitProduction[i] > value){
                value = unitProduction[i];
                unit = i;
                type = types[i];
            }
        }

        if(p.getResources() >= type.cost)
            return Train(u, type, gs);

        return null;
    }

    @Override
    protected UnitAction baseBehavior(Unit u, Player p, GameState gs) {
        if ((p.getResources()>=workerType.cost && playerUnits[0] < nHarvest) ||
                ((p.getResources()>=workerType.cost && p.getResources() > resourceTreshold)))
            return Train(u, workerType, gs);
        return null;
    }

    @Override
    protected UnitAction meleeUnitBehavior(Unit u, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit closestMeleeEnemy = null;
        int closestDistance = 0;
        int enemyDistance = 0;
        int mybase = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        int threshold = Math.max(pgs.getHeight(), pgs.getWidth());

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(u, u2, pgs);
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
            else if(u2.getPlayer()==p.getID() && u2.getType() == baseType){
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if(mybase == 0 || d < mybase)
                    mybase = d;
            }
        }
        if (closestEnemy!=null && (closestDistance < threshold/2 || mybase < threshold/2)) {
            return Attack(u,closestEnemy,gs,ru);
        }
        else
        {
            return Attack(u,null,gs,ru);
        }
    }

    @Override
    protected UnitAction scoutBehavior(Unit scout, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit base = null;
        int closestEnemyDistance = 0;
        int closestBaseDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        int threshold = Math.max(pgs.getHeight(), pgs.getWidth());

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(scout, u2, pgs);
                if (closestEnemy==null || d<closestEnemyDistance) {
                    closestEnemy = u2;
                    closestEnemyDistance = d;
                }
            }
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - scout.getX()) + Math.abs(u2.getY() - scout.getY());
                if (base==null || d<closestBaseDistance) {
                    base = u2;
                    closestBaseDistance = d;
                }
            }
        }

        if(closestEnemy!=null && (closestEnemyDistance < threshold/2 || closestBaseDistance < threshold/2)){
            if(pf.findPathToAdjacentPosition(scout, closestEnemy.getPosition(pgs), gs, ru)!=null
                    || manhattanDistance(scout.getX(), scout.getY(), closestEnemy.getX(), closestEnemy.getY()) == 1){
                if(isChokepoint(scout.getX(), scout.getY(), pgs)){
                    //might make them stall regardless, could look into it
                    if(isPlayerPredominant(scout, 5, 5, gs))
                        return Attack(scout, closestEnemy, gs, ru);
                    else
                        return Idle(scout, gs, ru);
                }
                else
                    return Attack(scout, closestEnemy, gs, ru);
            }
            else{
                if(scout.getResources() == 0)
                    return Scout(scout,closestEnemy,gs,ru);
                else if (base!=null){
                    if(pf.findPathToAdjacentPosition(scout, base.getPosition(pgs), gs, ru)!=null
                            || manhattanDistance(scout.getX(), scout.getY(), base.getX(), base.getY()) == 1)
                        return Harvest(scout, null, base, gs, ru);
                    else
                        return Scout(scout,closestEnemy,gs,ru);
                }
            }
        }else
        {
            return Attack(scout,null,gs,ru);
        }

        return Scout(scout,null,gs,ru);
    }

    @Override
    protected UnitAction builderBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if(playerUnits[4] == 0 && p.getResources() >= baseType.cost){
            int highscore = 0, pos = 0;

            List<Integer> positions = GetFreePositionsAround(worker, pgs);

            if(positions.isEmpty())
                return workerBehavior(worker, p, gs, ru);

            for(int i : positions){
                int score = evaluateBaseBuild(worker, i, p.getID(), pgs);
                if(score > highscore){
                    highscore = score;
                    pos = i;
                }
            }
            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();

            return Build(worker, baseType, x, y, gs, ru);
        }

        if(playerUnits[5] < totBarracks && p.getResources() >= barracksType.cost){
            int highscore = 0, pos = 0;
            List<Integer> positions;

            Unit base = GetClosestBase(worker, pgs);

            if(base== null)
                positions = GetFreePositionsAround(worker, pgs);
            else
                positions = GetFreePositionsAround(base, pgs);

            if(positions.isEmpty())
                return workerBehavior(worker, p, gs, ru);

            for(int i : positions){
                int score = evaluateBarrackBuild(i, p.getID(), pgs);
                if(score > highscore || highscore == 0){
                    highscore = score;
                    pos = i;
                }
            }

            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();

            return Build(worker, barracksType, x, y, gs, ru);
        }

        return workerBehavior(worker, p, gs, ru);
    }

    @Override
    protected List<Pair<Unit, UnitAction>> workersBehavior(List<Unit> workers, Player p, GameState gs, ResourceUsage ru) {
        int nbases = 0;
        int nbarracks = 0;
        int resourcesUsed = 0;
        UnitAction unitAction;
        Unit hw;
        List<Unit> harvestWorkers = new LinkedList<>();
        List<Unit> freeWorkers = new LinkedList<>(workers);
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Pair<Unit, UnitAction>> list = new ArrayList<>();

        if (workers.isEmpty()) return list;

        nbases = playerUnits[4];
        nbarracks = playerUnits[5];

        List<Integer> reservedPositions = new LinkedList<>();
        if (nbases==0 && !freeWorkers.isEmpty()) {
            // build a base:
            if (p.getResources()>=baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                //buildIfNotAlreadyBuilding(u,baseType,u.getX(),u.getY(),reservedPositions,p,pgs);
                if(gs.getActionAssignment(u) == null){
                    resourcesUsed+=baseType.cost;
                    unitAction = builderBehavior(u, p, gs, ru);

                    if(conflictingMove(u, unitAction, list, pgs))
                        unitAction = Idle(u, gs, ru);

                    list.add(new Pair<>(u, unitAction));
                }
            }
        }

        if (freeWorkers.size()>0){
            for(int i = 1; i <=nHarvest; i++){
                if(!freeWorkers.isEmpty()){
                    hw = freeWorkers.remove(0);
                    harvestWorkers.add(hw);
                }
            }
        }

        if (harvestWorkers.size() >= 1 && nbarracks < totBarracks && p.getResources() > barracksType.cost){
            Unit u;

            if(harvestWorkers.size()==1 && freeWorkers.size() > 0)
                u = freeWorkers.remove(0);
            else
                u = harvestWorkers.remove(0);
            //buildIfNotAlreadyBuilding(u, barracksType, u.getX(), u.getY(), reservedPositions, p, pgs);

            if(gs.getActionAssignment(u) == null){
                resourcesUsed += barracksType.cost;
                unitAction = builderBehavior(u, p, gs, ru);

                if(conflictingMove(u, unitAction, list, pgs))
                    unitAction = Idle(u, gs, ru);

                list.add(new Pair<>(u, unitAction));
            }
        }

        // harvest with the harvest worker:
        for (Unit harvestWorker : harvestWorkers) {
            if(gs.getActionAssignment(harvestWorker) != null)
                continue;

            Unit closestBase = null;
            Unit closestResource = null;
            Unit closestEnemy = null;

            int closestResourceDistance = 0;
            int closestBaseDistance = 0;
            int closestEnemyDistance = 0;

            for(Unit u2:pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestResource==null || d<closestResourceDistance) {
                        closestResource = u2;
                        closestResourceDistance = d;
                    }
                }

                if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestBase==null || d<closestBaseDistance) {
                        closestBase = u2;
                        closestBaseDistance = d;
                    }
                }

                if(u2.getPlayer() >= 0 && u2.getPlayer()!=harvestWorker.getPlayer()){
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestEnemy==null || d<closestEnemyDistance) {
                        closestEnemy = u2;
                        closestEnemyDistance = d;
                    }
                }
            }

            //if you are close to a base and an enemy is close to that, attack him
            if(closestEnemy != null && closestEnemyDistance <= 2 && closestBaseDistance <= 2)
                list.add(new Pair<>(harvestWorker, Attack(harvestWorker, closestEnemy, gs, ru)));

            else if (closestResource!=null && closestBase!=null) {
                if(closestEnemy!=null){
                    if(Between(closestEnemy, harvestWorker, closestResource) && harvestWorker.getResources()<1){
                        //this could be changed directly to Attack
                        unitAction = scoutBehavior(harvestWorker, p, gs, ru);
                    }
                    else
                        unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);
                }
                else
                    unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else if(closestBase!=null && harvestWorker.getResources()>0){
                unitAction = Harvest(harvestWorker, null, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else
                freeWorkers.add(harvestWorker);
        }

        for(Unit u:freeWorkers){
            if(gs.getActionAssignment(u) == null){
                unitAction = scoutBehavior(u, p, gs, ru);

                if(conflictingMove(u, unitAction, list, pgs))
                    unitAction = Idle(u, gs, ru);

                list.add(new Pair<>(u, unitAction));
            }
        }

        return list;
    }
}



class AMilRush extends AwareAI{

    public AMilRush(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public AMilRush(UnitTypeTable a_utt){
        this(a_utt, new AStarPathFinding());
    }

    @Override
    public AI clone() {
        return new AMilRush(utt, pf);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return null;
    }

    @Override
    protected UnitAction workerBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestBase = null;
        Unit closestResource = null;
        int closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestResource==null || d<closestDistance) {
                    closestResource = u2;
                    closestDistance = d;
                }
            }
        }
        closestDistance = 0;
        for(Unit u2:pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - worker.getX()) + Math.abs(u2.getY() - worker.getY());
                if (closestBase==null || d<closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }
        }

        if(closestBase!=null||closestResource!=null)
            return Harvest(worker, closestResource, closestBase, gs, ru);

        return scoutBehavior(worker, p, gs, ru);
    }

    @Override
    protected UnitAction barracksBehavior(Unit u, Player p, GameState gs) {
        int unit = 0, value = 0;
        UnitType type = heavyType;
        UnitType[] types = new UnitType[] {lightType, heavyType, rangedType};
        for(int i = 0; i < unitProduction.length; i++){
            if(unitProduction[i] > value){
                value = unitProduction[i];
                unit = i;
                type = types[i];
            }
        }

        if(p.getResources() >= type.cost)
            return Train(u, type, gs);

        return null;
    }

    @Override
    protected UnitAction baseBehavior(Unit u, Player p, GameState gs) {
        if ((p.getResources()>=workerType.cost && playerUnits[0] < nHarvest) ||
                ((p.getResources()>=workerType.cost && p.getResources() > resourceTreshold)))
            return Train(u, workerType, gs);
        return null;
    }

    @Override
    protected UnitAction meleeUnitBehavior(Unit u, Player p, GameState gs, ResourceUsage ru) {
        if(u.getType()==rangedType)
            return rangedBehavior(u, p, gs, ru);
        if(u.getType()==heavyType || u.getType()==lightType)
            return aggroMeleeBehevior(u, p, gs, ru);
        return Attack(u,null,gs,ru);
        /*Unit closestEnemy = null;
        Unit closestMeleeEnemy = null;
        int closestDistance = 0;
        int enemyDistance = 0;
        int mybase = 0;
        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(u, u2, pgs);
                if (closestEnemy==null || d<closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
        }
        if (closestEnemy!=null) {
            return Attack(u,closestEnemy,gs,ru);
        }
        else
        {
            return Attack(u,null,gs,ru);
        }*/
    }

    @Override
    protected UnitAction scoutBehavior(Unit scout, Player p, GameState gs, ResourceUsage ru) {
        Unit closestEnemy = null;
        Unit base = null;
        int closestEnemyDistance = 0;
        int closestBaseDistance = 0;

        PhysicalGameState pgs = gs.getPhysicalGameState();

        for(Unit u2:pgs.getUnits()) {
            if (u2.getPlayer()>=0 && u2.getPlayer()!=p.getID()) {
                int d = evaluateAttackTarget(scout, u2, pgs);
                if (closestEnemy==null || d<closestEnemyDistance) {
                    closestEnemy = u2;
                    closestEnemyDistance = d;
                }
            }
            if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                int d = Math.abs(u2.getX() - scout.getX()) + Math.abs(u2.getY() - scout.getY());
                if (base==null || d<closestBaseDistance) {
                    base = u2;
                    closestBaseDistance = d;
                }
            }
        }

        if(closestEnemy!=null){
            //System.out.println("Sees enemy at: " + closestEnemy.getX() + " " + closestEnemy.getY());
            if(pf.findPathToAdjacentPosition(scout, closestEnemy.getPosition(pgs), gs, ru)!=null
                    || manhattanDistance(scout.getX(), scout.getY(), closestEnemy.getX(), closestEnemy.getY()) == 1){
                if(isChokepoint(scout.getX(), scout.getY(), pgs)){
                    if(isPlayerPredominant(scout, 5, 5, gs))
                        return Attack(scout, closestEnemy, gs, ru);
                    else
                        return Idle(scout, gs, ru);
                }
                else
                    return Attack(scout, closestEnemy, gs, ru);
            }
            else{
                //System.out.println("Resources: " + scout.getResources());
                if(scout.getResources() == 0)
                    return Scout(scout,closestEnemy,gs,ru);
                else if (base!=null){
                    //System.out.println("base at " + base.getX() + " " + base.getY());
                    if(pf.findPathToAdjacentPosition(scout, base.getPosition(pgs), gs, ru)!=null
                            || manhattanDistance(scout.getX(), scout.getY(), base.getX(), base.getY()) == 1){
                        //System.out.println("Go back");
                        return Harvest(scout, null, base, gs, ru);
                    }
                    else
                        return Scout(scout,closestEnemy,gs,ru);
                }

            }
        }

        return Scout(scout,null,gs,ru);
    }

    @Override
    protected UnitAction builderBehavior(Unit worker, Player p, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        if(playerUnits[4] == 0 && p.getResources() >= baseType.cost){
            int highscore = 0, pos = 0;

            List<Integer> positions = GetFreePositionsAround(worker, pgs);

            if(positions.isEmpty())
                return workerBehavior(worker, p, gs, ru);

            for(int i : positions){
                int score = evaluateBaseBuild(worker, i, p.getID(), pgs);
                if(score > highscore){
                    highscore = score;
                    pos = i;
                }
            }

            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();

            return Build(worker, baseType, x, y, gs, ru);
        }

        if(playerUnits[5] < totBarracks && p.getResources() >= barracksType.cost){
            int highscore = 0, pos = 0;
            List<Integer> positions;

            Unit base = GetClosestBase(worker, pgs);

            if(base == null)
                positions = GetFreePositionsAround(worker, pgs);
            else
                positions = GetFreePositionsAround(base, pgs);

            if(positions.isEmpty())
                return workerBehavior(worker, p, gs, ru);

            for(int i : positions){
                int score = evaluateBarrackBuild(i, p.getID(), pgs);
                if(score > highscore || highscore == 0){
                    highscore = score;
                    pos = i;
                }
            }

            int x = pos % pgs.getWidth();
            int y = pos / pgs.getWidth();

            return Build(worker, barracksType, x, y, gs, ru);
        }

        return workerBehavior(worker, p, gs, ru);
    }

    @Override
    protected List<Pair<Unit, UnitAction>> workersBehavior(List<Unit> workers, Player p, GameState gs, ResourceUsage ru) {
        int nbases = 0;
        int nbarracks = 0;
        int resourcesUsed = 0;
        Unit hw;
        UnitAction unitAction;
        List<Unit> harvestWorkers = new LinkedList<>();
        List<Unit> freeWorkers = new LinkedList<>(workers);
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Pair<Unit, UnitAction>> list = new ArrayList<>();

        if (workers.isEmpty()) return list;

        nbases = playerUnits[4];
        nbarracks = playerUnits[5];

        List<Integer> reservedPositions = new LinkedList<>();
        if (nbases==0 && !freeWorkers.isEmpty()) {
            // build a base:
            if (p.getResources()>=baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                //buildIfNotAlreadyBuilding(u,baseType,u.getX(),u.getY(),reservedPositions,p,pgs);
                if(gs.getActionAssignment(u) == null){
                    resourcesUsed+=baseType.cost;
                    unitAction = builderBehavior(u, p, gs, ru);

                    if(conflictingMove(u, unitAction, list, pgs))
                        unitAction = Idle(u, gs, ru);

                    list.add(new Pair<>(u, unitAction));
                }
            }
        }

        if (freeWorkers.size()>0){
            for(int i = 1; i <=nHarvest; i++){
                if(!freeWorkers.isEmpty()){
                    hw = freeWorkers.remove(0);
                    harvestWorkers.add(hw);
                }
            }
        }

        if (harvestWorkers.size() >= 1 && nbarracks < totBarracks && p.getResources() > barracksType.cost){
            Unit u;

            if(harvestWorkers.size()==1 && freeWorkers.size() > 0)
                u = freeWorkers.remove(0);
            else
                u = harvestWorkers.remove(0);

            //buildIfNotAlreadyBuilding(u, barracksType, u.getX(), u.getY(), reservedPositions, p, pgs);
            if(gs.getActionAssignment(u) == null){
                resourcesUsed += barracksType.cost;
                unitAction = builderBehavior(u, p, gs, ru);

                if(conflictingMove(u, unitAction, list, pgs))
                    unitAction = Idle(u, gs, ru);

                list.add(new Pair<>(u, unitAction));
            }
        }

        // harvest with the harvest worker:
        for (Unit harvestWorker : harvestWorkers) {
            if(gs.getActionAssignment(harvestWorker) != null)
                continue;

            Unit closestBase = null;
            Unit closestResource = null;
            Unit closestEnemy = null;

            int closestResourceDistance = 0;
            int closestBaseDistance = 0;
            int closestEnemyDistance = 0;

            for(Unit u2:pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestResource==null || d<closestResourceDistance) {
                        closestResource = u2;
                        closestResourceDistance = d;
                    }
                }

                if (u2.getType().isStockpile && u2.getPlayer()==p.getID()) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestBase==null || d<closestBaseDistance) {
                        closestBase = u2;
                        closestBaseDistance = d;
                    }
                }

                if(u2.getPlayer() >= 0 && u2.getPlayer()!=harvestWorker.getPlayer()){
                    int d = Math.abs(u2.getX() - harvestWorker.getX()) + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestEnemy==null || d<closestEnemyDistance) {
                        closestEnemy = u2;
                        closestEnemyDistance = d;
                    }
                }
            }

            //if you are close to a base and an enemy is close to that, attack him
            if(closestEnemy != null && closestEnemyDistance <= 2 && closestBaseDistance <= 2)
                list.add(new Pair<>(harvestWorker, Attack(harvestWorker, closestEnemy, gs, ru)));

            else if (closestResource!=null && closestBase!=null) {

                if(closestEnemy!=null){
                    if(Between(closestEnemy, harvestWorker, closestResource) && harvestWorker.getResources()<1){
                        //this could be changed directly to Attack
                        unitAction = scoutBehavior(harvestWorker, p, gs, ru);
                    }
                    else
                        unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);
                }
                else
                    unitAction = Harvest(harvestWorker, closestResource, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else if(closestBase!=null && harvestWorker.getResources()>0){
                unitAction = Harvest(harvestWorker, null, closestBase, gs, ru);

                if(conflictingMove(harvestWorker, unitAction, list, pgs))
                    unitAction = Idle(harvestWorker, gs, ru);

                list.add(new Pair<>(harvestWorker, unitAction));
            }
            else
                freeWorkers.add(harvestWorker);
        }

        for(Unit u:freeWorkers){
            if(gs.getActionAssignment(u) == null){
                unitAction = scoutBehavior(u, p, gs, ru);

                if(conflictingMove(u, unitAction, list, pgs))
                    unitAction = Idle(u, gs, ru);

                list.add(new Pair<>(u, unitAction));
            }
        }

        return list;
    }
}



class ScoutPathFinding{
    Boolean free[][];
    int closed[];
    int open[];  // open list
    int heuristic[];     // heuristic value of the elements in 'open'
    int parents[];
    int cost[];     // cost of reaching a given position so far
    int inOpenOrClosed[];
    int openinsert = 0;

    public UnitAction findPath(Unit start, int targetpos, GameState gs, ResourceUsage ru) {
        return findPathToPositionInRange(start,targetpos,0,gs,ru);
    }


    /*
     * This function is like the previous one, but doesn't try to reach 'target', but just to
     * reach a position that is at most 'range' far away from 'target'
     */
    public UnitAction findPathToPositionInRange(Unit start, int targetpos, int range, GameState gs, ResourceUsage ru) {
        if (!runScoutAStar(start, targetpos, range, gs, ru))
            return null;

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int w = pgs.getWidth();
        int h = pgs.getHeight();

        int pos = open[openinsert];
        int parent = parents[openinsert];

        int last = pos;
//      System.out.println("- Path from " + start.getX() + "," + start.getY() + " to " + targetpos%w + "," + targetpos/w + " (range " + range + ") in " + iterations + " iterations");
        while(parent!=pos) {
            last = pos;
            pos = parent;
            parent = closed[pos];
//			System.out.println("    " + pos%w + "," + pos/w);
        }

        int x = last%w;
        int y = last/w;
        Unit nu = pgs.getUnitAt(x, y);

        if (last == pos+w) {
            if(nu != null){
                if(nu.getType().isResource)
                    return new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_DOWN);
            }

            return new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_DOWN);
        }

        if (last == pos-1){
            if(nu != null){
                if(nu.getType().isResource)
                    return new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_LEFT);
            }

            return new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_LEFT);
        }

        if (last == pos-w){
            if(nu != null){
                if(nu.getType().isResource)
                    return new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_UP);
            }

            return new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_UP);
        }
        if (last == pos+1){
            if(nu != null){
                if(nu.getType().isResource)
                    return new UnitAction(UnitAction.TYPE_HARVEST, UnitAction.DIRECTION_RIGHT);
            }

            return new UnitAction(UnitAction.TYPE_MOVE, UnitAction.DIRECTION_RIGHT);
        }
        return null;
    }

    public int findStepToPositionInRange(Unit start, int targetpos, int range, GameState gs, ResourceUsage ru) {
        if (!runScoutAStar(start, targetpos, range, gs, ru))
            return -1;

        PhysicalGameState pgs = gs.getPhysicalGameState();
        int w = pgs.getWidth();
        int h = pgs.getHeight();

        int pos = open[openinsert];
        int parent = parents[openinsert];

        int last = pos;
//      System.out.println("- Path from " + start.getX() + "," + start.getY() + " to " + targetpos%w + "," + targetpos/w + " (range " + range + ") in " + iterations + " iterations");
        while(parent!=pos) {
            last = pos;
            pos = parent;
            parent = closed[pos];
//			System.out.println("    " + pos%w + "," + pos/w);
        }

        if (last == pos+w) return UnitAction.DIRECTION_DOWN;
        if (last == pos-1) return UnitAction.DIRECTION_LEFT;
        if (last == pos-w) return UnitAction.DIRECTION_UP;
        if (last == pos+1) return UnitAction.DIRECTION_RIGHT;

        return -1;
    }

    /*
     * This function is like the previous one, but doesn't try to reach 'target', but just to
     * reach a position adjacent to 'target'
     */
    public UnitAction findPathToAdjacentPosition(Unit start, int targetpos, GameState gs, ResourceUsage ru) {
        return findPathToPositionInRange(start, targetpos, 1, gs, ru);
    }

    public boolean pathExists(Unit start, int targetpos, GameState gs, ResourceUsage ru) {
        return start.getPosition(gs.getPhysicalGameState()) == targetpos
                || findPath(start, targetpos, gs, ru) != null;
    }


    public boolean pathToPositionInRangeExists(Unit start, int targetpos, int range, GameState gs, ResourceUsage ru) {
        int x = targetpos%gs.getPhysicalGameState().getWidth();
        int y = targetpos/gs.getPhysicalGameState().getWidth();
        int d = (x-start.getX())*(x-start.getX()) + (y-start.getY())*(y-start.getY());
        return d <= range * range
                || findPathToPositionInRange(start, targetpos, range, gs, ru) != null;
    }

    // and keep the "open" list sorted:
    void addToOpen(int x, int y, int newPos, int oldPos, int h) {
        cost[newPos] = cost[oldPos]+1;

        // find the right position for the insert:
        for(int i = openinsert-1;i>=0;i--) {
            if (heuristic[i]+cost[open[i]]>=h+cost[newPos]) {
//                System.out.println("Inserting at " + (i+1) + " / " + openinsert);
                // shift all the elements:
                System.arraycopy(open, i, open, i + 1, openinsert - i);
                System.arraycopy(heuristic, i, heuristic, i + 1, openinsert - i);
                System.arraycopy(parents, i, parents, i + 1, openinsert - i);

                // insert at i+1:
                open[i+1] = newPos;
                heuristic[i+1] = h;
                parents[i+1] = oldPos;
                openinsert++;
                inOpenOrClosed[newPos] = 1;
                return;
            }
        }
        // i = -1;
//        System.out.println("Inserting at " + 0 + " / " + openinsert);
        // shift all the elements:
        System.arraycopy(open, 0, open, 1, openinsert);
        System.arraycopy(heuristic, 0, heuristic, 1, openinsert);
        System.arraycopy(parents, 0, parents, 1, openinsert);

        // insert at 0:
        open[0] = newPos;
        heuristic[0] = h;
        parents[0] = oldPos;
        openinsert++;
        inOpenOrClosed[newPos] = 1;
    }


    int manhattanDistance(int x, int y, int x2, int y2) {
        return Math.abs(x-x2) + Math.abs(y-y2);
    }

    public int findDistToPositionInRange(Unit start, int targetpos, int range, GameState gs, ResourceUsage ru) {
        if (!runScoutAStar(start, targetpos, range, gs, ru))
            return -1;

        int pos = open[openinsert];
        int parent = parents[openinsert];

        int dist = 0;
        while(parent!=pos) {
            pos = parent;
            parent = closed[pos];
            dist++;
            //System.out.println("    " + pos%w + "," + pos/w);
        }
        return dist;
    }

    /**
     * Runs specialized A* search that ignores resources. Calling functions can, after running this, figure out either
     * the action to take to walk along the shortest path, or the cost of the shortest
     * path.
     *
     * @param start
     * @param targetpos
     * @param range
     * @param gs
     * @param ru
     * @return Did we successfully complete our search?
     */
    private boolean runScoutAStar(Unit start, int targetpos, int range, GameState gs, ResourceUsage ru) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int w = pgs.getWidth();
        int h = pgs.getHeight();
        if (free==null || free.length < w || free[0].length < h) {
            free = new Boolean[w][h];
            closed = new int[w*h];
            open = new int[w*h];
            heuristic = new int[w*h];
            parents = new int[w*h];
            inOpenOrClosed = new int[w*h];
            cost = new int[w*h];
        }

        for (int x = 0; x < w; ++x) {
            Arrays.fill(free[x], null);
        }
        Arrays.fill(closed, -1);
        Arrays.fill(inOpenOrClosed, 0);

        if (ru!=null) {
            for(int pos:ru.getPositionsUsed()) {
                free[pos%w][pos/w] = false;
            }
        }
        int targetx = targetpos%w;
        int targety = targetpos/w;
        int sq_range = range*range;
        int startPos = start.getY()*w + start.getX();

        assert(targetx>=0);
        assert(targetx<w);
        assert(targety>=0);
        assert(targety<h);
        assert(start.getX()>=0);
        assert(start.getX()<w);
        assert(start.getY()>=0);
        assert(start.getY()<h);

        openinsert = 0;
        open[openinsert] = startPos;
        heuristic[openinsert] = manhattanDistance(start.getX(), start.getY(), targetx, targety);
        parents[openinsert] = startPos;
        inOpenOrClosed[startPos] = 1;
        cost[startPos] = 0;
        openinsert++;
        while(openinsert>0) {
            openinsert--;
            int pos = open[openinsert];
            int parent = parents[openinsert];
            if (closed[pos]!=-1) continue;
            closed[pos] = parent;

            int x = pos%w;
            int y = pos/w;

            if (((x-targetx)*(x-targetx)+(y-targety)*(y-targety))<=sq_range) {
                // path found: return to let the calling code compute either action or cost
                return true;
            }
            if (y>0 && inOpenOrClosed[pos-w] == 0) {
                if (free[x][y-1]==null){
                    Unit u = gs.getPhysicalGameState().getUnitAt(x, y-1);
                    boolean isResource = false;
                    if(u!=null){
                        if(u.getType().isResource)
                            isResource = true;
                    }
                    free[x][y-1]= (gs.free(x, y-1) || isResource);
                }
                assert(free[x][y-1]!=null);
                if (free[x][y-1]) {
                    addToOpen(x,y-1,pos-w,pos,manhattanDistance(x, y-1, targetx, targety));
                }
            }
            if (x<pgs.getWidth()-1 && inOpenOrClosed[pos+1] == 0) {
                if (free[x+1][y]==null){
                    Unit u = gs.getPhysicalGameState().getUnitAt(x+1, y);
                    boolean isResource = false;
                    if(u!=null){
                        if(u.getType().isResource)
                            isResource = true;
                    }
                    free[x+1][y]= (gs.free(x+1, y) || isResource);
                }
                assert(free[x+1][y]!=null);
                if (free[x+1][y]) {
                    addToOpen(x+1,y,pos+1,pos,manhattanDistance(x+1, y, targetx, targety));
                }
            }
            if (y<pgs.getHeight()-1 && inOpenOrClosed[pos+w] == 0) {
                if (free[x][y+1]==null){
                    Unit u = gs.getPhysicalGameState().getUnitAt(x, y+1);
                    boolean isResource = false;
                    if(u!=null){
                        if(u.getType().isResource)
                            isResource = true;
                    }
                    free[x][y+1]= (gs.free(x, y+1) || isResource);
                }
                assert(free[x][y+1]!=null);
                if (free[x][y+1]) {
                    addToOpen(x,y+1,pos+w,pos,manhattanDistance(x, y+1, targetx, targety));
                }
            }
            if (x>0 && inOpenOrClosed[pos-1] == 0) {
                if (free[x-1][y]==null){
                    Unit u = gs.getPhysicalGameState().getUnitAt(x-1, y);
                    boolean isResource = false;
                    if(u!=null){
                        if(u.getType().isResource)
                            isResource = true;
                    }
                    free[x-1][y]= (gs.free(x-1, y) || isResource);
                }
                assert(free[x-1][y]!=null);
                if (free[x-1][y]) {
                    addToOpen(x-1,y,pos-1,pos,manhattanDistance(x-1, y, targetx, targety));
                }
            }
        }

        return false;
    }
}

