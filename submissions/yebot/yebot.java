/*
 * yebot — Macro-LLM + Hard-coded Micro
 *
 * Architecture:
 *   MICRO (Java, every tick):
 *     - Small maps (≤12): worker rush (matches built-in WorkerRush exactly)
 *     - Large maps (>12): eco start → barracks → army
 *     - Counter-unit targeting for combat units
 *
 *   MACRO (LLM, synchronous every 200 ticks):
 *     - Reads game state summary (unit counts, resources, map size)
 *     - Picks one of: WORKER_RUSH, ECON_HEAVY, ECON_RANGED, COUNTER_MIX, ALL_IN
 *     - Java micro adapts production and aggression based on macro plan
 *     - If LLM fails/slow, Java picks a sensible default macro automatically
 *
 * @author Ye
 * Team: yebot
 */
package ai.abstraction.submissions.yebot;

import ai.abstraction.AbstractAction;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.Harvest;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import com.google.gson.*;
import rts.*;
import rts.units.*;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class yebot extends AbstractionLayerAI {

    // ═══════════════════════════════════════════════════════════════════════════
    //  CONFIG
    // ═══════════════════════════════════════════════════════════════════════════
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "qwen3:8b";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int LLM_TIMEOUT  = 5000;
    private static final int LLM_INTERVAL = 200;

    // ═══════════════════════════════════════════════════════════════════════════
    //  UNIT TYPES
    // ═══════════════════════════════════════════════════════════════════════════
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ═══════════════════════════════════════════════════════════════════════════
    //  MACRO STRATEGY (LLM-controlled, synchronous)
    // ═══════════════════════════════════════════════════════════════════════════
    private String macroStrategy = "DEFAULT";
    private int lastLLMTick = -LLM_INTERVAL;

    // ═══════════════════════════════════════════════════════════════════════════
    //  LLM SYSTEM PROMPT
    // ═══════════════════════════════════════════════════════════════════════════
    private static final String SYSTEM_PROMPT =
        "You are a MicroRTS macro strategist. Given the game state, choose ONE strategy.\n"
      + "UNITS: Worker(HP=1,dmg=1,cost=1) Light(HP=4,dmg=2,cost=2) "
      + "Heavy(HP=8,dmg=4,cost=3) Ranged(HP=3,dmg=1,range=3,cost=2)\n"
      + "COUNTER LOGIC: Light beats Worker. Heavy beats Light. Ranged beats Heavy. Workers swarm Ranged.\n"
      + "STRATEGIES:\n"
      + "- WORKER_RUSH: Send all workers to attack. Best on small maps or when ahead in workers.\n"
      + "- ECON_HEAVY: Build barracks, produce Heavies. Good vs Light-heavy enemy.\n"
      + "- ECON_RANGED: Build barracks, produce Ranged. Good vs Heavy-heavy enemy.\n"
      + "- COUNTER_MIX: Produce whatever counters enemy composition.\n"
      + "- ALL_IN: Stop eco, send everything to attack. Use when you have army advantage.\n"
      + "OUTPUT FORMAT (JSON only): {\"thinking\":\"brief reason\",\"strategy\":\"STRATEGY_NAME\"}\n";

    // ═══════════════════════════════════════════════════════════════════════════
    //  CONSTRUCTORS
    // ═══════════════════════════════════════════════════════════════════════════

    public yebot(UnitTypeTable a_utt) { this(a_utt, new AStarPathFinding()); }

    public yebot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        macroStrategy = "DEFAULT";
        lastLLMTick   = -LLM_INTERVAL;
    }

    public void reset(UnitTypeTable a_utt) {
        utt          = a_utt;
        workerType   = utt.getUnitType("Worker");
        lightType    = utt.getUnitType("Light");
        heavyType    = utt.getUnitType("Heavy");
        rangedType   = utt.getUnitType("Ranged");
        baseType     = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
    }

    @Override
    public AI clone() { return new yebot(utt, pf); }

    // ═══════════════════════════════════════════════════════════════════════════
    //  MAIN LOOP
    // ═══════════════════════════════════════════════════════════════════════════

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);
        int tick = gs.getTime();
        int mapW = pgs.getWidth();

        // ── Every 200 ticks: call LLM, get macro plan ─────────────────────────
        if (tick - lastLLMTick >= LLM_INTERVAL) {
            lastLLMTick = tick;
            try {
                String stateText = buildMacroStateText(player, gs, pgs);
                String response  = callLLM(stateText);
                String parsed    = parseMacroStrategy(response);
                if (parsed != null) {
                    macroStrategy = parsed;
                    System.out.println("[yebot] t=" + tick + " LLM → " + parsed);
                }
            } catch (Exception e) {
                System.err.println("[yebot] LLM error: " + e.getMessage());
            }
        }

        // ── Resolve strategy ──────────────────────────────────────────────────
        String strategy = resolveStrategy(mapW, tick, player, gs, pgs);

        // ── Execute ───────────────────────────────────────────────────────────
        switch (strategy) {
            case "WORKER_RUSH":
                executeWorkerRush(player, p, gs, pgs);
                break;
            case "ALL_IN":
                executeAllIn(player, p, gs, pgs);
                break;
            case "ECON_HEAVY":
                executeEcon(player, p, gs, pgs, heavyType);
                break;
            case "ECON_RANGED":
                executeEcon(player, p, gs, pgs, rangedType);
                break;
            case "COUNTER_MIX":
                executeEcon(player, p, gs, pgs, pickCounterUnit(pgs, player));
                break;
            default:
                executeEcon(player, p, gs, pgs, heavyType);
                break;
        }

        return translateActions(player, gs);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  RESOLVE STRATEGY
    // ═══════════════════════════════════════════════════════════════════════════

    private String resolveStrategy(int mapW, int tick, int player,
                                    GameState gs, PhysicalGameState pgs) {
        if (!"DEFAULT".equals(macroStrategy)) return macroStrategy;
        if (mapW <= 12) return "WORKER_RUSH";
        if (tick < 200) return "ECON_HEAVY";
        return autoCounter(pgs, player);
    }

    private String autoCounter(PhysicalGameState pgs, int player) {
        int eH = 0, eL = 0, eR = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player) {
                if (u.getType() == heavyType) eH++;
                else if (u.getType() == lightType) eL++;
                else if (u.getType() == rangedType) eR++;
            }
        }
        if (eH >= eL && eH >= eR && eH > 0) return "ECON_RANGED";
        if (eL >= eH && eL >= eR && eL > 0) return "ECON_HEAVY";
        return "ECON_HEAVY";
    }

    private UnitType pickCounterUnit(PhysicalGameState pgs, int player) {
        int eH = 0, eL = 0, eR = 0, eW = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player) {
                if (u.getType() == heavyType) eH++;
                else if (u.getType() == lightType) eL++;
                else if (u.getType() == rangedType) eR++;
                else if (u.getType() == workerType) eW++;
            }
        }
        if (eL > eH && eL > eR) return heavyType;
        if (eH >= eL && eH >= eR && eH > 0) return rangedType;
        if (eW > eH + eL + eR) return lightType;
        return heavyType;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  STRATEGY: WORKER RUSH
    //  Copied from built-in WorkerRush with proper harvest checking
    // ═══════════════════════════════════════════════════════════════════════════

    private void executeWorkerRush(int player, Player p, GameState gs,
                                     PhysicalGameState pgs) {
        // Base: train workers nonstop
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (p.getResources() >= workerType.cost) train(u, workerType);
            }
        }

        // Non-worker combat units: attack nearest
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                meleeAttack(u, p, pgs);
            }
        }

        // Workers
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canHarvest && u.getPlayer() == player) workers.add(u);
        }
        workerRushBehavior(workers, p, gs, pgs);
    }

    /**
     * Exact same logic as built-in WorkerRush.workersBehavior:
     * build base if none, 1 harvester with proper getAbstractAction check, rest attack.
     */
    private void workerRushBehavior(List<Unit> workers, Player p,
                                      GameState gs, PhysicalGameState pgs) {
        int nbases = 0;
        int resourcesUsed = 0;
        Unit harvestWorker = null;
        List<Unit> freeWorkers = new LinkedList<>(workers);

        if (workers.isEmpty()) return;

        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == p.getID()) nbases++;
        }

        List<Integer> reservedPositions = new LinkedList<>();

        // Build base if none
        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, baseType, u.getX(), u.getY(),
                        reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        // Assign one harvester
        if (!freeWorkers.isEmpty()) harvestWorker = freeWorkers.remove(0);

        // Harvest — check getAbstractAction to avoid canceling in-progress harvest
        if (harvestWorker != null) {
            Unit closestBase = null;
            Unit closestResource = null;
            int closestDistance = 0;

            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX())
                          + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestResource == null || d < closestDistance) {
                        closestResource = u2;
                        closestDistance = d;
                    }
                }
            }
            closestDistance = 0;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isStockpile && u2.getPlayer() == p.getID()) {
                    int d = Math.abs(u2.getX() - harvestWorker.getX())
                          + Math.abs(u2.getY() - harvestWorker.getY());
                    if (closestBase == null || d < closestDistance) {
                        closestBase = u2;
                        closestDistance = d;
                    }
                }
            }

            boolean harvestWorkerFree = true;
            if (harvestWorker.getResources() > 0) {
                if (closestBase != null) {
                    AbstractAction aa = getAbstractAction(harvestWorker);
                    if (!(aa instanceof Harvest)) {
                        harvest(harvestWorker, null, closestBase);
                    }
                    harvestWorkerFree = false;
                }
            } else {
                if (closestResource != null && closestBase != null) {
                    AbstractAction aa = getAbstractAction(harvestWorker);
                    if (!(aa instanceof Harvest)) {
                        harvest(harvestWorker, closestResource, closestBase);
                    }
                    harvestWorkerFree = false;
                }
            }

            if (harvestWorkerFree) freeWorkers.add(harvestWorker);
        }

        // All remaining workers: attack nearest enemy
        for (Unit u : freeWorkers) meleeAttack(u, p, pgs);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  STRATEGY: ALL IN
    // ═══════════════════════════════════════════════════════════════════════════

    private void executeAllIn(int player, Player p, GameState gs,
                                PhysicalGameState pgs) {
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (p.getResources() >= workerType.cost) train(u, workerType);
            }
        }
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType().canAttack
                    && gs.getActionAssignment(u) == null) {
                meleeAttack(u, p, pgs);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  STRATEGY: ECON BUILD (heavy, ranged, or counter)
    // ═══════════════════════════════════════════════════════════════════════════

    private void executeEcon(int player, Player p, GameState gs,
                              PhysicalGameState pgs, UnitType combatUnit) {
        int nbases = 0, nbarracks = 0, nworkers = 0;
        int resourcesUsed = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == baseType) nbases++;
                else if (u.getType() == barracksType) nbarracks++;
                else if (u.getType() == workerType) nworkers++;
            }
        }

        // Base: train workers (cap 3 per base)
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (nworkers < nbases * 3
                        && p.getResources() - resourcesUsed >= workerType.cost) {
                    train(u, workerType);
                    resourcesUsed += workerType.cost;
                }
            }
        }

        // Barracks: train combat
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (p.getResources() - resourcesUsed >= combatUnit.cost) {
                    train(u, combatUnit);
                    resourcesUsed += combatUnit.cost;
                }
            }
        }

        // Combat units: attack nearest
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                meleeAttack(u, p, pgs);
            }
        }

        // Workers
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canHarvest && u.getPlayer() == player) workers.add(u);
        }
        econWorkerBehavior(workers, p, gs, pgs, nbarracks, resourcesUsed);
    }

    private void econWorkerBehavior(List<Unit> workers, Player p, GameState gs,
                                      PhysicalGameState pgs, int nbarracks,
                                      int resourcesUsed) {
        int nbases = 0;
        List<Unit> freeWorkers = new LinkedList<>(workers);
        if (workers.isEmpty()) return;

        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == p.getID()) nbases++;
        }

        List<Integer> reservedPositions = new LinkedList<>();

        // Build base if none
        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, baseType, u.getX(), u.getY(),
                        reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        // Build barracks if none
        if (nbarracks == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= barracksType.cost + resourcesUsed) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, barracksType, u.getX(), u.getY(),
                        reservedPositions, p, pgs);
                resourcesUsed += barracksType.cost;
            }
        }

        // 2 harvesters, rest attack
        int maxHarvesters = Math.min(2, freeWorkers.size());
        List<Unit> harvesters = new ArrayList<>();
        for (int i = 0; i < maxHarvesters && !freeWorkers.isEmpty(); i++) {
            harvesters.add(freeWorkers.remove(0));
        }

        // Harvest properly with getAbstractAction check
        for (Unit hw : harvesters) {
            if (!doHarvest(hw, p, pgs)) {
                freeWorkers.add(hw); // can't harvest, go attack
            }
        }

        // Remaining workers attack
        for (Unit w : freeWorkers) meleeAttack(w, p, pgs);
    }

    /**
     * Proper harvest with getAbstractAction check — never cancels in-progress harvest.
     * Returns false if can't harvest (no base or no resource).
     */
    private boolean doHarvest(Unit hw, Player p, PhysicalGameState pgs) {
        Unit closestBase = null;
        Unit closestResource = null;
        int closestDistance = 0;

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - hw.getX()) + Math.abs(u2.getY() - hw.getY());
                if (closestResource == null || d < closestDistance) {
                    closestResource = u2;
                    closestDistance = d;
                }
            }
        }
        closestDistance = 0;
        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer() == p.getID()) {
                int d = Math.abs(u2.getX() - hw.getX()) + Math.abs(u2.getY() - hw.getY());
                if (closestBase == null || d < closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }
        }

        if (hw.getResources() > 0) {
            if (closestBase != null) {
                AbstractAction aa = getAbstractAction(hw);
                if (!(aa instanceof Harvest)) {
                    harvest(hw, null, closestBase);
                }
                return true;
            }
            return false;
        } else {
            if (closestResource != null && closestBase != null) {
                AbstractAction aa = getAbstractAction(hw);
                if (!(aa instanceof Harvest)) {
                    harvest(hw, closestResource, closestBase);
                }
                return true;
            }
            return false;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  MELEE ATTACK — attack nearest enemy (same as built-in WorkerRush)
    // ═══════════════════════════════════════════════════════════════════════════

    private void meleeAttack(Unit u, Player p, PhysicalGameState pgs) {
        Unit closestEnemy = null;
        int closestDistance = 0;
        for (Unit u2 : pgs.getUnits()) {
            if (u2.getPlayer() >= 0 && u2.getPlayer() != p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestEnemy == null || d < closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
        }
        if (closestEnemy != null) {
            attack(u, closestEnemy);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  LLM MACRO — state text + parse
    // ═══════════════════════════════════════════════════════════════════════════

    private String buildMacroStateText(int player, GameState gs,
                                        PhysicalGameState pgs) {
        int myW = 0, myB = 0, myBr = 0, myH = 0, myR = 0, myL = 0;
        int eW = 0, eB = 0, eBr = 0, eH = 0, eR = 0, eL = 0;
        int res = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) { res++; continue; }
            if (u.getPlayer() == player) {
                if (u.getType() == workerType) myW++;
                else if (u.getType() == baseType) myB++;
                else if (u.getType() == barracksType) myBr++;
                else if (u.getType() == heavyType) myH++;
                else if (u.getType() == rangedType) myR++;
                else if (u.getType() == lightType) myL++;
            } else if (u.getPlayer() >= 0) {
                if (u.getType() == workerType) eW++;
                else if (u.getType() == baseType) eB++;
                else if (u.getType() == barracksType) eBr++;
                else if (u.getType() == heavyType) eH++;
                else if (u.getType() == rangedType) eR++;
                else if (u.getType() == lightType) eL++;
            }
        }

        return "Turn=" + gs.getTime()
             + " Map=" + pgs.getWidth() + "x" + pgs.getHeight()
             + " Resources=" + gs.getPlayer(player).getResources()
             + "\nMY: W=" + myW + " B=" + myB + " Br=" + myBr
             + " H=" + myH + " R=" + myR + " L=" + myL
             + "\nENEMY: W=" + eW + " B=" + eB + " Br=" + eBr
             + " H=" + eH + " R=" + eR + " L=" + eL
             + "\nMapRes=" + res
             + "\nChoose the best strategy.";
    }

    private String parseMacroStrategy(String response) {
        try {
            response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
            int s = response.indexOf("{"), e = response.lastIndexOf("}") + 1;
            if (s < 0 || e <= s) return null;

            JsonObject json = JsonParser.parseString(response.substring(s, e)).getAsJsonObject();
            if (json.has("thinking"))
                System.out.println("[yebot] LLM thinks: " + json.get("thinking").getAsString());
            if (json.has("strategy")) {
                String strat = json.get("strategy").getAsString().toUpperCase().trim();
                switch (strat) {
                    case "WORKER_RUSH": case "ECON_HEAVY": case "ECON_RANGED":
                    case "COUNTER_MIX": case "ALL_IN":
                        return strat;
                }
            }
        } catch (Exception ex) {
            System.err.println("[yebot] Parse macro error: " + ex.getMessage());
        }
        return null;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  LLM HTTP CALL
    // ═══════════════════════════════════════════════════════════════════════════

    private String callLLM(String stateText) {
        try {
            URL url = new URL(API_URL);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(LLM_TIMEOUT);
            conn.setReadTimeout(LLM_TIMEOUT);

            JsonObject req = new JsonObject();
            req.addProperty("model", OLLAMA_MODEL);

            JsonArray msgs = new JsonArray();
            JsonObject sys = new JsonObject();
            sys.addProperty("role", "system");
            sys.addProperty("content", SYSTEM_PROMPT);
            msgs.add(sys);
            JsonObject usr = new JsonObject();
            usr.addProperty("role", "user");
            usr.addProperty("content", stateText);
            msgs.add(usr);
            req.add("messages", msgs);

            JsonObject fmt = new JsonObject();
            fmt.addProperty("type", "json_object");
            req.add("response_format", fmt);
            req.addProperty("temperature", 0.3);
            req.addProperty("max_tokens", 256);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(req.toString().getBytes(StandardCharsets.UTF_8));
            }

            if (conn.getResponseCode() == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    JsonObject resp = JsonParser.parseString(sb.toString()).getAsJsonObject();
                    JsonArray choices = resp.getAsJsonArray("choices");
                    if (choices != null && choices.size() > 0)
                        return choices.get(0).getAsJsonObject()
                                .getAsJsonObject("message").get("content").getAsString();
                }
            }
        } catch (Exception e) {
            System.err.println("[yebot] callLLM: " + e.getMessage());
        }
        return "{}";
    }

    @Override
    public List<ParameterSpecification> getParameters() { return new ArrayList<>(); }
}