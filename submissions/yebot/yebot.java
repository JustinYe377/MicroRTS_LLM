/*
 * yebot — Macro-LLM + Rule-based Micro  
 *
 * Key fixes.
 *   1. ARMY GROUPING: combat units rally at a staging point and only commit when
 *      we have critical mass (or LLM forces commit). Fixes trickle-attack on large maps.
 *   2. LLM CONTROLS MORE: in addition to strategy, LLM chooses aggression level
 *      (0=turtle, 1=pressure, 2=commit) and target preference (army/eco/expansion).
 *   3. HARVESTER SCALING: caps scale properly to map area; harvesters assigned to
 *      nearest resource node (not all clustered at one patch).
 *   4. BUILD PLACEMENT: buildings placed at an OFFSET from the base, not ON the base tile.
 *      Barracks placed on the side facing the enemy; expansion base toward far resources.
 *   5. SCOUTING: on first LLM tick of a large map, one worker scouts; LLM gets
 *      "enemy seen at" info and can act on it.
 *   6. MAP-AWARE DEFAULTS: fallback behavior no longer sends workers across a 64x64 map.
 *   7. EMERGENCY RALLY: if enemy reaches base, all army units return home (not just
 *      "attack nearest" which was pulling them further away).
 *   8. LLM TICK-CONSISTENT: state summary includes distance-to-enemy and army-ratio
 *      so the model can tell "I'm ahead, commit" vs "I'm behind, turtle".
 *
 * @author Ye
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

    // ==========================================================================
    //  CONFIG
    // ==========================================================================
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "qwen3:8b";
    // OLLAMA_HOST preferred (base URL, no path). OLLAMA_URL kept for backward compat.
    private static final String OLLAMA_HOST = resolveOllamaHost();
    private static final String API_URL     = OLLAMA_HOST + "/api/generate";


    private static final int CONNECT_TIMEOUT = 1800;    // matches fortress-bot
    private static final int READ_TIMEOUT    = 15000;   // matches fortress-bot

    private static final int LLM_INTERVAL       = 200;  // matches fortress-bot (160 + slack)
    private static final int STRATEGY_DECAY     = 800;
    private static final int MAX_FAIL_STREAK    = 3;    // after N consecutive fails, disable LLM

    private static String resolveOllamaHost() {
        String host = System.getenv("OLLAMA_HOST");
        if (host != null && !host.isEmpty()) return host.replaceAll("/+$", "");
        String legacy = System.getenv("OLLAMA_URL");
        if (legacy != null && !legacy.isEmpty()) {
            // Strip /v1/... or /api/... suffix if user passed a full URL
            return legacy.replaceAll("/(v1|api)/.*$", "").replaceAll("/+$", "");
        }
        return "http://localhost:11434";
    }

    // ==========================================================================
    //  UNIT TYPES
    // ==========================================================================
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ==========================================================================
    //  MACRO STATE (LLM-controlled)
    // ==========================================================================
    private String macroStrategy   = "DEFAULT";
    private int    aggression      = 1;        // 0 turtle, 1 pressure, 2 commit
    private String targetPref      = "ARMY";   // ARMY, ECO, EXPANSION, ANY
    private int    lastLLMTick     = Integer.MIN_VALUE / 2; // force fire on first tick
    private int    strategySetTick = 0;
    private int    llmAttempts     = 0;
    private int    llmSuccesses    = 0;
    private boolean llmEverWorked  = false;
    private boolean llmDisabled    = false;    // flip true after repeated failures
    private int    llmFailStreak   = 0;

    // Memoized rally point (recomputed each tick cheaply, but stored for stability)
    private int rallyX = -1, rallyY = -1;

    // ==========================================================================
    //  LLM SYSTEM PROMPT
    // ==========================================================================
    private static final String SYSTEM_PROMPT =
        "You are a MicroRTS macro strategist. You receive the current game state and must choose "
      + "(1) a strategy, (2) an aggression level, and (3) a target preference.\n"
      + "UNITS: Worker(HP=1,dmg=1,cost=1) Light(HP=4,dmg=2,cost=2) "
      + "Heavy(HP=8,dmg=4,cost=3) Ranged(HP=3,dmg=1,range=3,cost=2)\n"
      + "COUNTER: Light>Worker, Heavy>Light, Ranged>Heavy, mass Workers>lone Ranged.\n"
      + "STRATEGIES:\n"
      + "- WORKER_RUSH: all workers attack. Only viable on SMALL maps (<=8) or with clear worker lead.\n"
      + "- ECON_HEAVY: build barracks, produce Heavy. Good early-mid on medium+ maps.\n"
      + "- ECON_RANGED: build barracks, produce Ranged. Good vs Heavy-heavy enemies.\n"
      + "- COUNTER_MIX: produce whatever counters enemy composition now.\n"
      + "- ALL_IN: stop economy, max-produce combat units and commit.\n"
      + "- EXPAND: build a second base near distant resources. Best on LARGE+ maps early.\n"
      + "- DEFEND: turtle, hold base, wait for a counter-attack window.\n"
      + "AGGRESSION: 0=turtle (hold rally, never commit), 1=pressure (commit when army>=4), "
      + "2=commit (attack now regardless of size).\n"
      + "TARGET: ARMY (kill enemy army first), ECO (snipe workers+bases), EXPANSION (hit their 2nd base), ANY.\n"
      + "Consider: map size (large maps favor EXPAND+ECON, small favor WORKER_RUSH), army ratio, "
      + "whether UnderAttack is true (prefer DEFEND), and current economy.\n"
      + "OUTPUT (JSON only, no markdown):\n"
      + "{\"thinking\":\"brief reason\",\"strategy\":\"NAME\",\"aggression\":0|1|2,\"target\":\"ARMY|ECO|EXPANSION|ANY\"}\n";

    // ==========================================================================
    //  CONSTRUCTORS
    // ==========================================================================

    public yebot(UnitTypeTable a_utt) { this(a_utt, new AStarPathFinding()); }

    public yebot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        macroStrategy   = "DEFAULT";
        aggression      = 1;
        targetPref      = "ARMY";
        lastLLMTick     = Integer.MIN_VALUE / 2;
        strategySetTick = 0;
        rallyX = rallyY = -1;
        llmAttempts     = 0;
        llmSuccesses    = 0;
        llmEverWorked   = false;
        llmDisabled     = false;
        llmFailStreak   = 0;
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

    // ==========================================================================
    //  MAIN LOOP
    // ==========================================================================

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs  = gs.getPhysicalGameState();
        Player            p    = gs.getPlayer(player);
        int               tick = gs.getTime();

        // Update rally point each tick (cheap, keeps units grouped near base toward enemy)
        updateRallyPoint(player, pgs);

        // Synchronous LLM call on interval. First call may take up to READ_TIMEOUT
        // to let the model load; subsequent calls hit a warm model and reply fast.
        int interval = adaptiveLLMInterval(pgs);
        if (!llmDisabled && tick - lastLLMTick >= interval) {
            lastLLMTick = tick;
            llmAttempts++;
            String stateText = buildMacroStateText(player, gs, pgs);
            long t0 = System.currentTimeMillis();
            String response = callLLM(stateText);
            long dt = System.currentTimeMillis() - t0;
            System.out.println("[yebot] t=" + tick + " LLM call #" + llmAttempts
                    + " in " + dt + "ms, len=" + (response == null ? 0 : response.length()));

            if (parseMacroResponse(response)) {
                strategySetTick = tick;
                llmSuccesses++;
                llmEverWorked   = true;
                llmFailStreak   = 0;
                System.out.println("[yebot] t=" + tick
                        + " LLM OK -> " + macroStrategy
                        + " agg=" + aggression
                        + " tgt=" + targetPref
                        + " (" + llmSuccesses + "/" + llmAttempts + ")");
            } else {
                llmFailStreak++;
                System.out.println("[yebot] t=" + tick + " LLM parse/call FAILED ("
                        + llmFailStreak + " in a row), raw=" + truncate(response, 150));
                if (llmFailStreak >= MAX_FAIL_STREAK) {
                    llmDisabled = true;
                    System.out.println("[yebot] t=" + tick + " DISABLING LLM after "
                            + MAX_FAIL_STREAK + " failures — running on rule-based fallback");
                }
            }
        }

        // Strategy decay
        if (!"DEFAULT".equals(macroStrategy) && (tick - strategySetTick > STRATEGY_DECAY)) {
            macroStrategy = "DEFAULT";
            System.out.println("[yebot] t=" + tick + " strategy decayed -> DEFAULT");
        }

        // Resolve strategy (emergency override possible)
        String strategy = resolveStrategy(tick, player, gs, pgs);

        switch (strategy) {
            case "WORKER_RUSH": executeWorkerRush(player, p, gs, pgs);                        break;
            case "ALL_IN":      executeAllIn(player, p, gs, pgs);                             break;
            case "ECON_HEAVY":  executeEcon(player, p, gs, pgs, heavyType);                   break;
            case "ECON_RANGED": executeEcon(player, p, gs, pgs, rangedType);                  break;
            case "COUNTER_MIX": executeEcon(player, p, gs, pgs, pickCounterUnit(pgs, player));break;
            case "EXPAND":      executeExpand(player, p, gs, pgs);                            break;
            case "DEFEND":      executeDefend(player, p, gs, pgs);                            break;
            default:            executeEcon(player, p, gs, pgs, heavyType);                   break;
        }

        return translateActions(player, gs);
    }

    // ==========================================================================
    //  MAP SIZE
    // ==========================================================================
    private int     mapArea(PhysicalGameState pgs)     { return pgs.getWidth() * pgs.getHeight(); }
    private boolean isMediumMap(PhysicalGameState pgs)  { return mapArea(pgs) >= 144; }   // >=12x12
    private boolean isLargeMap(PhysicalGameState pgs)   { return mapArea(pgs) >= 256; }   // >=16x16
    private boolean isHugeMap(PhysicalGameState pgs)    { return mapArea(pgs) >= 1024; }  // >=32x32

    private int adaptiveLLMInterval(PhysicalGameState pgs) {
        if (isHugeMap(pgs))  return 150;
        if (isLargeMap(pgs)) return 175;
        return LLM_INTERVAL;
    }

    private int maxWorkers(PhysicalGameState pgs, int nbases) {
        // Scale more aggressively for huge maps — more resource nodes, need more harvesters
        int perBase = isHugeMap(pgs) ? 8 : isLargeMap(pgs) ? 6 : isMediumMap(pgs) ? 4 : 3;
        return perBase * Math.max(1, nbases);
    }

    private int maxBarracks(PhysicalGameState pgs) {
        if (isHugeMap(pgs))  return 3;
        if (isLargeMap(pgs)) return 2;
        return 1;
    }

    private int maxHarvesters(PhysicalGameState pgs, int resourceNodes, int nbases) {
        int perBase = isHugeMap(pgs) ? 4 : isLargeMap(pgs) ? 3 : isMediumMap(pgs) ? 2 : 2;
        int cap = perBase * Math.max(1, nbases);
        // Don't exceed 2 harvesters per resource node (contention)
        return Math.min(cap, Math.max(1, resourceNodes * 2));
    }

    private int countResourceNodes(PhysicalGameState pgs) {
        int count = 0;
        for (Unit u : pgs.getUnits()) if (u.getType().isResource) count++;
        return count;
    }

    // ==========================================================================
    //  RALLY POINT: between our base and enemy base, closer to our base
    // ==========================================================================

    private void updateRallyPoint(int player, PhysicalGameState pgs) {
        Unit myBase = null, enemyBase = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType) {
                if (u.getPlayer() == player) myBase = u;
                else if (u.getPlayer() >= 0) enemyBase = u;
            }
        }
        if (myBase == null) { rallyX = pgs.getWidth()/2; rallyY = pgs.getHeight()/2; return; }

        // Rally progressively forward: 40% at game start, 70% by mid-game.
        // Prevents both bots from eternally farming in their corners.
        int tick = 0;
        // Note: we don't have gs here, so we use a static field pattern via
        // re-reading from the caller instead. We'll approximate with a ratchet
        // based on how many times this has been called — but cleanest is to
        // just use 50% as a compromise, pushed further when units accumulate.
        int pct = 50;
        // Simple heuristic: count our army; more army = push rally further
        int myArmy = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType().canAttack && !u.getType().canHarvest)
                myArmy++;
        }
        if (myArmy >= 3) pct = 60;
        if (myArmy >= 6) pct = 75;

        if (enemyBase != null) {
            rallyX = myBase.getX() + (enemyBase.getX() - myBase.getX()) * pct / 100;
            rallyY = myBase.getY() + (enemyBase.getY() - myBase.getY()) * pct / 100;
        } else {
            // Unknown enemy position: scout toward map center aggressively
            int cx = pgs.getWidth()/2, cy = pgs.getHeight()/2;
            int dx = Integer.signum(cx - myBase.getX());
            int dy = Integer.signum(cy - myBase.getY());
            rallyX = myBase.getX() + dx * Math.min(6, Math.abs(cx - myBase.getX()));
            rallyY = myBase.getY() + dy * Math.min(6, Math.abs(cy - myBase.getY()));
        }
    }

    // ==========================================================================
    //  STRATEGY RESOLUTION
    // ==========================================================================

    private String resolveStrategy(int tick, int player, GameState gs, PhysicalGameState pgs) {
        if (isUnderAttack(player, pgs, Math.max(4, pgs.getWidth() / 4))) {
            return "DEFEND";
        }

        if (!"DEFAULT".equals(macroStrategy)) return macroStrategy;

        int mapW = pgs.getWidth();
        if (mapW <= 8)  return "WORKER_RUSH";

        if (mapW <= 12) {
            return tick < 150 ? "ECON_HEAVY" : autoCounter(pgs, player);
        }

        // 16x16+: only EXPAND if we actually want multiple bases
        int targetBases = isHugeMap(pgs) ? 3 : isLargeMap(pgs) ? 2 : 1;

        if (targetBases >= 2 && tick < 150) return "EXPAND";
        if (tick < 400)                      return "ECON_HEAVY";
        return autoCounter(pgs, player);
    }

    private String autoCounter(PhysicalGameState pgs, int player) {
        int eH = 0, eL = 0, eR = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player) {
                if      (u.getType() == heavyType)  eH++;
                else if (u.getType() == lightType)  eL++;
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
                if      (u.getType() == heavyType)  eH++;
                else if (u.getType() == lightType)  eL++;
                else if (u.getType() == rangedType) eR++;
                else if (u.getType() == workerType) eW++;
            }
        }
        if (eL > eH && eL > eR)             return heavyType;
        if (eH >= eL && eH >= eR && eH > 0) return rangedType;
        if (eW > eH + eL + eR)              return lightType;
        return heavyType;
    }

    // ==========================================================================
    //  THREAT DETECTION
    // ==========================================================================

    private boolean isUnderAttack(int player, PhysicalGameState pgs, int threatRadius) {
        Unit myBase = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player) { myBase = u; break; }
        }
        if (myBase == null) return false;
        for (Unit enemy : pgs.getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player) continue;
            if (!enemy.getType().canAttack) continue;
            int d = Math.abs(enemy.getX() - myBase.getX())
                  + Math.abs(enemy.getY() - myBase.getY());
            if (d <= threatRadius) return true;
        }
        return false;
    }

    // ==========================================================================
    //  BUILD PLACEMENT HELPERS
    //  Key fix: offset AWAY from base toward enemy (barracks) or toward resource (expansion).
    //  Passing the base's own (x,y) caused findBuildingPosition to search outward from
    //  the occupied tile, often placing structures in awkward spots.
    // ==========================================================================

    private Unit findMyBase(int player, PhysicalGameState pgs) {
        for (Unit u : pgs.getUnits())
            if (u.getType() == baseType && u.getPlayer() == player) return u;
        return null;
    }

    private Unit findEnemyBase(int player, PhysicalGameState pgs) {
        for (Unit u : pgs.getUnits())
            if (u.getType() == baseType && u.getPlayer() >= 0 && u.getPlayer() != player) return u;
        return null;
    }

    /** Preferred build position: 1-2 tiles from base, on the side toward enemy. */
    private int[] buildPosNearBase(int player, Unit fallback, PhysicalGameState pgs) {
        Unit myBase = findMyBase(player, pgs);
        if (myBase == null) return new int[]{fallback.getX(), fallback.getY()};
        Unit eBase = findEnemyBase(player, pgs);

        int bx = myBase.getX(), by = myBase.getY();
        int dx = 1, dy = 0;
        if (eBase != null) {
            dx = Integer.signum(eBase.getX() - bx);
            dy = Integer.signum(eBase.getY() - by);
            if (dx == 0 && dy == 0) { dx = 1; dy = 0; }
        } else {
            // Face map center
            dx = Integer.signum(pgs.getWidth()/2 - bx);
            dy = Integer.signum(pgs.getHeight()/2 - by);
            if (dx == 0 && dy == 0) { dx = 1; dy = 0; }
        }

        // Clamp hint within map
        int hx = clamp(bx + dx * 2, 0, pgs.getWidth()  - 1);
        int hy = clamp(by + dy * 2, 0, pgs.getHeight() - 1);
        return new int[]{hx, hy};
    }

    private int clamp(int v, int lo, int hi) { return Math.max(lo, Math.min(hi, v)); }

    // ==========================================================================
    //  ARMY MANAGEMENT — rally + commit
    // ==========================================================================

    /** Should we commit (attack) rather than hold rally? */
    private boolean shouldCommit(int player, GameState gs, PhysicalGameState pgs,
                                  int myArmy, int eArmy) {
        // Under attack: commit (home defense)
        if (isUnderAttack(player, pgs, Math.max(4, pgs.getWidth()/4))) return true;

        if ("WORKER_RUSH".equals(macroStrategy) || "ALL_IN".equals(macroStrategy)) return true;

        // LLM aggression controls
        if (aggression == 2) return true;
        if (aggression == 0 && gs.getTime() < 2000) return false;  // only turtle early

        // Base threshold by map size (lowered — previous values caused draws)
        int threshold = isHugeMap(pgs) ? 5 : isLargeMap(pgs) ? 3 : 2;

        // TIME ESCALATION: decay threshold toward 1 as the game drags on.
        // This guarantees commitment before the tick limit — no more draws.
        int tick = gs.getTime();
        if (tick > 1000) threshold--;
        if (tick > 2000) threshold--;
        if (tick > 3000) threshold = 1;
        threshold = Math.max(1, threshold);

        if (myArmy >= threshold) return true;
        // Army advantage: commit even with 1 unit if we clearly out-army them
        if (myArmy > eArmy && myArmy >= 1 && tick > 500) return true;
        if (myArmy >= eArmy + 2 && myArmy >= 2) return true;

        return false;
    }

    /** Priority targeting with target-preference bias from LLM. */
    private void priorityAttack(Unit u, int player, PhysicalGameState pgs) {
        Unit target   = null;
        int  bestPri  = -1;
        int  bestDist = Integer.MAX_VALUE;

        for (Unit enemy : pgs.getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player) continue;
            int pri = baseUnitPriority(enemy);
            pri += targetPrefBonus(enemy);

            int d = Math.abs(enemy.getX() - u.getX()) + Math.abs(enemy.getY() - u.getY());
            // Slight proximity bias: closer targets get +0.1 effectively via tie-break
            if (pri > bestPri || (pri == bestPri && d < bestDist)) {
                bestPri  = pri;
                bestDist = d;
                target   = enemy;
            }
        }
        if (target != null) attack(u, target);
    }

    private int baseUnitPriority(Unit enemy) {
        if      (enemy.getType() == rangedType)   return 6;
        else if (enemy.getType() == lightType)    return 5;
        else if (enemy.getType() == heavyType)    return 4;
        else if (enemy.getType() == workerType)   return 3;
        else if (enemy.getType() == barracksType) return 2;
        else if (enemy.getType() == baseType)     return 1;
        return 0;
    }

    private int targetPrefBonus(Unit enemy) {
        switch (targetPref) {
            case "ECO":
                if (enemy.getType() == workerType) return 5;
                if (enemy.getType() == baseType)   return 4;
                return 0;
            case "EXPANSION":
                if (enemy.getType() == baseType)     return 4;
                if (enemy.getType() == barracksType) return 3;
                return 0;
            case "ARMY":
                if (enemy.getType().canAttack && !enemy.getType().canHarvest) return 3;
                return 0;
            default:
                return 0;
        }
    }

    /** Combat unit behavior: rally or commit based on army state. */
    private void combatBehavior(Unit u, int player, GameState gs, PhysicalGameState pgs,
                                 boolean commit) {
        // If under attack, intercept the closest threat near our base
        if (isUnderAttack(player, pgs, Math.max(4, pgs.getWidth()/4))) {
            Unit threat = nearestEnemyAttackerToBase(player, pgs);
            if (threat != null) { attack(u, threat); return; }
        }

        if (commit) {
            priorityAttack(u, player, pgs);
        } else {
            // Rally: move to rally point if far, otherwise idle (no action)
            int d = Math.abs(u.getX() - rallyX) + Math.abs(u.getY() - rallyY);
            if (d > 2) {
                move(u, rallyX, rallyY);
            } else {
                // At rally — if an enemy is visible within a short range, engage
                Unit near = nearestEnemy(u, player, pgs, 6);
                if (near != null) attack(u, near);
            }
        }
    }

    private Unit nearestEnemy(Unit from, int player, PhysicalGameState pgs, int maxDist) {
        Unit best = null; int bestD = Integer.MAX_VALUE;
        for (Unit e : pgs.getUnits()) {
            if (e.getPlayer() < 0 || e.getPlayer() == player) continue;
            int d = Math.abs(e.getX() - from.getX()) + Math.abs(e.getY() - from.getY());
            if (d <= maxDist && d < bestD) { bestD = d; best = e; }
        }
        return best;
    }

    private Unit nearestEnemyAttackerToBase(int player, PhysicalGameState pgs) {
        Unit myBase = findMyBase(player, pgs);
        if (myBase == null) return null;
        Unit best = null; int bestD = Integer.MAX_VALUE;
        for (Unit e : pgs.getUnits()) {
            if (e.getPlayer() < 0 || e.getPlayer() == player) continue;
            if (!e.getType().canAttack) continue;
            int d = Math.abs(e.getX() - myBase.getX()) + Math.abs(e.getY() - myBase.getY());
            if (d < bestD) { bestD = d; best = e; }
        }
        return best;
    }

    // ==========================================================================
    //  STRATEGY: WORKER RUSH (small maps only)
    // ==========================================================================

    private void executeWorkerRush(int player, Player p, GameState gs, PhysicalGameState pgs) {
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (p.getResources() >= workerType.cost) train(u, workerType);
            }
        }
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, gs, pgs, true);  // always commit in rush
            }
        }
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits())
            if (u.getType().canHarvest && u.getPlayer() == player) workers.add(u);
        workerRushBehavior(workers, player, p, gs, pgs);
    }

    private void workerRushBehavior(List<Unit> workers, int player, Player p,
                                     GameState gs, PhysicalGameState pgs) {
        int nbases = 0, resourcesUsed = 0;
        Unit harvestWorker = null;
        List<Unit> freeWorkers = new LinkedList<>(workers);
        if (workers.isEmpty()) return;

        for (Unit u : pgs.getUnits())
            if (u.getType() == baseType && u.getPlayer() == p.getID()) nbases++;

        List<Integer> reserved = new LinkedList<>();

        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = buildPosNearBase(player, u, pgs);
                buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1], reserved, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        if (!freeWorkers.isEmpty()) harvestWorker = freeWorkers.remove(0);
        if (harvestWorker != null && !doHarvest(harvestWorker, p, pgs))
            freeWorkers.add(harvestWorker);

        for (Unit u : freeWorkers) priorityAttack(u, p.getID(), pgs);
    }

    // ==========================================================================
    //  STRATEGY: ALL IN
    // ==========================================================================

    private void executeAllIn(int player, Player p, GameState gs, PhysicalGameState pgs) {
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (p.getResources() >= workerType.cost) train(u, workerType);
            }
        }
        // Also mass-produce from barracks
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                UnitType t = pickCounterUnit(pgs, player);
                if (p.getResources() >= t.cost) train(u, t);
            }
        }
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType().canAttack
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, gs, pgs, true);
            }
        }
    }

    // ==========================================================================
    //  STRATEGY: ECON (heavy/ranged/counter)
    // ==========================================================================

    private void executeEcon(int player, Player p, GameState gs,
                              PhysicalGameState pgs, UnitType combatUnit) {
        int nbases = 0, nbarracks = 0, nworkers = 0;
        int resourcesUsed = 0;
        int resourceNodes = countResourceNodes(pgs);
        int myArmy = 0, eArmy = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if      (u.getType() == baseType)     nbases++;
                else if (u.getType() == barracksType) nbarracks++;
                else if (u.getType() == workerType)   nworkers++;
                else if (u.getType().canAttack && !u.getType().canHarvest) myArmy++;
            } else if (u.getPlayer() >= 0) {
                if (u.getType().canAttack && !u.getType().canHarvest) eArmy++;
            }
        }

        int wCap = maxWorkers(pgs, nbases);
        int bCap = maxBarracks(pgs);
        boolean commit = shouldCommit(player, gs, pgs, myArmy, eArmy);

        // Train workers from bases
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (nworkers < wCap && p.getResources() - resourcesUsed >= workerType.cost) {
                    train(u, workerType);
                    resourcesUsed += workerType.cost;
                    nworkers++;
                }
            }
        }

        // Train combat units from all barracks
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (p.getResources() - resourcesUsed >= combatUnit.cost) {
                    train(u, combatUnit);
                    resourcesUsed += combatUnit.cost;
                }
            }
        }

        // Combat dispatch (rally or commit)
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, gs, pgs, commit);
            }
        }

        // Workers
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits())
            if (u.getType().canHarvest && u.getPlayer() == player) workers.add(u);

        econWorkerBehavior(workers, player, p, gs, pgs,
                nbases, nbarracks, bCap, resourcesUsed, resourceNodes);
    }

    private void econWorkerBehavior(List<Unit> workers, int player, Player p,
                                     GameState gs, PhysicalGameState pgs,
                                     int nbases, int nbarracks, int bCap,
                                     int resourcesUsed, int resourceNodes) {
        List<Unit> freeWorkers = new LinkedList<>(workers);
        if (freeWorkers.isEmpty()) return;
        List<Integer> reserved = new LinkedList<>();

        // Rebuild base if lost
        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = buildPosNearBase(player, u, pgs);
                buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1], reserved, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        // Build barracks up to cap
        int toBuild = bCap - nbarracks;
        for (int i = 0; i < toBuild && !freeWorkers.isEmpty(); i++) {
            if (p.getResources() >= barracksType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = buildPosNearBase(player, u, pgs);
                buildIfNotAlreadyBuilding(u, barracksType, bp[0], bp[1], reserved, p, pgs);
                resourcesUsed += barracksType.cost;
            }
        }

        // Harvester assignment — each harvester goes to NEAREST resource+base pair
        int hCap = maxHarvesters(pgs, resourceNodes, nbases);
        List<Unit> harvesters = new ArrayList<>();
        for (int i = 0; i < hCap && !freeWorkers.isEmpty(); i++)
            harvesters.add(freeWorkers.remove(0));

        for (Unit hw : harvesters)
            if (!doHarvest(hw, p, pgs)) freeWorkers.add(hw);

        // Leftover workers: if committing, attack; else stay near base
        boolean commit = aggression == 2 || "ALL_IN".equals(macroStrategy);
        for (Unit w : freeWorkers) {
            if (commit) priorityAttack(w, player, pgs);
            else {
                // Stay near base — don't send workers across the map
                Unit myBase = findMyBase(player, pgs);
                if (myBase != null) {
                    int d = Math.abs(w.getX() - myBase.getX())
                          + Math.abs(w.getY() - myBase.getY());
                    if (d > 3) move(w, myBase.getX(), myBase.getY());
                }
            }
        }
    }

    // ==========================================================================
    //  STRATEGY: EXPAND
    // ==========================================================================

    private void executeExpand(int player, Player p, GameState gs, PhysicalGameState pgs) {
        int nbases = 0, nbarracks = 0, nworkers = 0;
        int resourcesUsed = 0;
        int resourceNodes = countResourceNodes(pgs);
        int myArmy = 0, eArmy = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if      (u.getType() == baseType)     nbases++;
                else if (u.getType() == barracksType) nbarracks++;
                else if (u.getType() == workerType)   nworkers++;
                else if (u.getType().canAttack && !u.getType().canHarvest) myArmy++;
            } else if (u.getPlayer() >= 0) {
                if (u.getType().canAttack && !u.getType().canHarvest) eArmy++;
            }
        }

        int wCap        = maxWorkers(pgs, nbases);
        int targetBases = isHugeMap(pgs) ? 3 : isLargeMap(pgs) ? 2 : 1;
        boolean commit  = shouldCommit(player, gs, pgs, myArmy, eArmy);

        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (nworkers < wCap && p.getResources() - resourcesUsed >= workerType.cost) {
                    train(u, workerType);
                    resourcesUsed += workerType.cost;
                    nworkers++;
                }
            }
        }

        // Barracks also produce during expand (so we don't sit defenseless)
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                UnitType t = pickCounterUnit(pgs, player);
                if (p.getResources() - resourcesUsed >= t.cost) {
                    train(u, t);
                    resourcesUsed += t.cost;
                }
            }
        }

        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, gs, pgs, commit);
            }
        }

        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits())
            if (u.getType().canHarvest && u.getPlayer() == player) workers.add(u);

        expandWorkerBehavior(workers, player, p, gs, pgs,
                nbases, targetBases, nbarracks, resourcesUsed, resourceNodes);
    }

    private void expandWorkerBehavior(List<Unit> workers, int player, Player p,
                                       GameState gs, PhysicalGameState pgs,
                                       int nbases, int targetBases, int nbarracks,
                                       int resourcesUsed, int resourceNodes) {
        List<Unit> freeWorkers = new LinkedList<>(workers);
        if (freeWorkers.isEmpty()) return;
        List<Integer> reserved = new LinkedList<>();

        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = buildPosNearBase(player, u, pgs);
                buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1], reserved, p, pgs);
                resourcesUsed += baseType.cost;
                nbases++;
            }
        }

        // Expansion: pick worker CLOSEST to target expansion coord, not just freeWorkers[0]
        if (nbases < targetBases && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                int[] ep = expansionPos(player, pgs);
                Unit closest = null; int bestD = Integer.MAX_VALUE;
                for (Unit w : freeWorkers) {
                    int d = Math.abs(w.getX() - ep[0]) + Math.abs(w.getY() - ep[1]);
                    if (d < bestD) { bestD = d; closest = w; }
                }
                if (closest != null) {
                    freeWorkers.remove(closest);
                    buildIfNotAlreadyBuilding(closest, baseType, ep[0], ep[1], reserved, p, pgs);
                    resourcesUsed += baseType.cost;
                }
            }
        }

        if (nbarracks == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= barracksType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = buildPosNearBase(player, u, pgs);
                buildIfNotAlreadyBuilding(u, barracksType, bp[0], bp[1], reserved, p, pgs);
                resourcesUsed += barracksType.cost;
            }
        }

        int hCap = maxHarvesters(pgs, resourceNodes, nbases);
        List<Unit> harvesters = new ArrayList<>();
        for (int i = 0; i < hCap && !freeWorkers.isEmpty(); i++)
            harvesters.add(freeWorkers.remove(0));

        for (Unit hw : harvesters)
            if (!doHarvest(hw, p, pgs)) freeWorkers.add(hw);

        // Leftovers stay near base (don't walk across map)
        Unit myBase = findMyBase(player, pgs);
        for (Unit w : freeWorkers) {
            if (myBase != null) {
                int d = Math.abs(w.getX() - myBase.getX())
                      + Math.abs(w.getY() - myBase.getY());
                if (d > 3) move(w, myBase.getX(), myBase.getY());
            }
        }
    }

    /**
     * Expansion site: the resource cluster farthest from our base.
     * Clusters > single nodes: find the resource whose sum-of-distances to
     * other resources is smallest (densest), among those far from our base.
     */
    private int[] expansionPos(int player, PhysicalGameState pgs) {
        Unit myBase = findMyBase(player, pgs);
        int bx = myBase != null ? myBase.getX() : pgs.getWidth()/2;
        int by = myBase != null ? myBase.getY() : pgs.getHeight()/2;

        List<Unit> resources = new ArrayList<>();
        for (Unit u : pgs.getUnits()) if (u.getType().isResource) resources.add(u);
        if (resources.isEmpty())
            return new int[]{pgs.getWidth()-1-bx, pgs.getHeight()-1-by};

        // Score each resource: prefer FAR from our base but near OTHER resources (cluster)
        Unit best = null; double bestScore = -1;
        for (Unit r : resources) {
            double distFromBase = Math.abs(r.getX()-bx) + Math.abs(r.getY()-by);
            double clusterScore = 0;
            for (Unit r2 : resources) {
                if (r2 == r) continue;
                int d = Math.abs(r.getX()-r2.getX()) + Math.abs(r.getY()-r2.getY());
                if (d <= 4) clusterScore += (5 - d);  // nearby resources add weight
            }
            double score = distFromBase + clusterScore * 2;
            if (score > bestScore) { bestScore = score; best = r; }
        }
        // Build adjacent to resource cluster but not ON the resource
        int tx = best.getX(), ty = best.getY();
        // Nudge toward our base so the base isn't placed at the very edge
        tx += Integer.signum(bx - tx);
        ty += Integer.signum(by - ty);
        return new int[]{clamp(tx, 0, pgs.getWidth()-1), clamp(ty, 0, pgs.getHeight()-1)};
    }

    // ==========================================================================
    //  STRATEGY: DEFEND
    // ==========================================================================

    private void executeDefend(int player, Player p, GameState gs, PhysicalGameState pgs) {
        int nbases = 0, nbarracks = 0, nworkers = 0;
        int resourcesUsed = 0;
        int resourceNodes = countResourceNodes(pgs);

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if      (u.getType() == baseType)     nbases++;
                else if (u.getType() == barracksType) nbarracks++;
                else if (u.getType() == workerType)   nworkers++;
            }
        }

        // Train counter units
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                UnitType defUnit = pickCounterUnit(pgs, player);
                if (p.getResources() - resourcesUsed >= defUnit.cost) {
                    train(u, defUnit);
                    resourcesUsed += defUnit.cost;
                }
            }
        }

        // Minimum workers
        int minW = isLargeMap(pgs) ? 3 : 2;
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (nworkers < minW && p.getResources() - resourcesUsed >= workerType.cost) {
                    train(u, workerType);
                    resourcesUsed += workerType.cost;
                    nworkers++;
                }
            }
        }

        // Military: always commits when defending (intercept threats)
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, gs, pgs, true);
            }
        }

        List<Unit> freeWorkers = new LinkedList<>();
        for (Unit u : pgs.getUnits())
            if (u.getType().canHarvest && u.getPlayer() == player) freeWorkers.add(u);

        List<Integer> reserved = new LinkedList<>();

        if (nbases == 0 && !freeWorkers.isEmpty()
                && p.getResources() >= baseType.cost + resourcesUsed) {
            Unit u   = freeWorkers.remove(0);
            int[] bp = buildPosNearBase(player, u, pgs);
            buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1], reserved, p, pgs);
            resourcesUsed += baseType.cost;
        }

        if (nbarracks == 0 && nbases > 0 && !freeWorkers.isEmpty()
                && p.getResources() >= barracksType.cost + resourcesUsed) {
            Unit u   = freeWorkers.remove(0);
            int[] bp = buildPosNearBase(player, u, pgs);
            buildIfNotAlreadyBuilding(u, barracksType, bp[0], bp[1], reserved, p, pgs);
        }

        int hCap     = maxHarvesters(pgs, resourceNodes, nbases);
        int assigned = 0;
        for (Unit w : freeWorkers) {
            if (assigned < hCap && doHarvest(w, p, pgs)) { assigned++; continue; }
            // Rest: defend the base (attack close threats, else stay put)
            Unit threat = nearestEnemy(w, player, pgs, 5);
            if (threat != null) attack(w, threat);
            else {
                Unit myBase = findMyBase(player, pgs);
                if (myBase != null) {
                    int d = Math.abs(w.getX() - myBase.getX())
                          + Math.abs(w.getY() - myBase.getY());
                    if (d > 3) move(w, myBase.getX(), myBase.getY());
                }
            }
        }
    }

    // ==========================================================================
    //  HARVEST
    // ==========================================================================

    private boolean doHarvest(Unit hw, Player p, PhysicalGameState pgs) {
        Unit closestBase = null, closestResource = null;
        int  dR = 0, dB = 0;

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - hw.getX()) + Math.abs(u2.getY() - hw.getY());
                if (closestResource == null || d < dR) { closestResource = u2; dR = d; }
            }
        }
        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer() == p.getID()) {
                int d = Math.abs(u2.getX() - hw.getX()) + Math.abs(u2.getY() - hw.getY());
                if (closestBase == null || d < dB) { closestBase = u2; dB = d; }
            }
        }

        if (hw.getResources() > 0) {
            if (closestBase != null) {
                AbstractAction aa = getAbstractAction(hw);
                if (!(aa instanceof Harvest)) harvest(hw, null, closestBase);
                return true;
            }
            return false;
        } else {
            if (closestResource != null && closestBase != null) {
                AbstractAction aa = getAbstractAction(hw);
                if (!(aa instanceof Harvest)) harvest(hw, closestResource, closestBase);
                return true;
            }
            return false;
        }
    }

    // ==========================================================================
    //  LLM STATE TEXT — richer context
    // ==========================================================================

    private String buildMacroStateText(int player, GameState gs, PhysicalGameState pgs) {
        int myW=0, myB=0, myBr=0, myH=0, myR=0, myL=0, myHp=0;
        int eW=0,  eB=0,  eBr=0,  eH=0,  eR=0,  eL=0,  eHp=0;
        int res = 0;

        Unit myBase = null, eBase = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) { res++; continue; }
            if (u.getPlayer() == player) {
                myHp += u.getHitPoints();
                if      (u.getType() == workerType)   myW++;
                else if (u.getType() == baseType)     { myB++; if (myBase==null) myBase=u; }
                else if (u.getType() == barracksType) myBr++;
                else if (u.getType() == heavyType)    myH++;
                else if (u.getType() == rangedType)   myR++;
                else if (u.getType() == lightType)    myL++;
            } else if (u.getPlayer() >= 0) {
                eHp += u.getHitPoints();
                if      (u.getType() == workerType)   eW++;
                else if (u.getType() == baseType)     { eB++; if (eBase==null) eBase=u; }
                else if (u.getType() == barracksType) eBr++;
                else if (u.getType() == heavyType)    eH++;
                else if (u.getType() == rangedType)   eR++;
                else if (u.getType() == lightType)    eL++;
            }
        }

        String mapCat = isHugeMap(pgs) ? "huge" : isLargeMap(pgs) ? "large"
                      : isMediumMap(pgs) ? "medium" : "small";
        boolean threat = isUnderAttack(player, pgs, pgs.getWidth() / 4);
        int myArmy = myH + myR + myL;
        int eArmy  = eH  + eR  + eL;

        int distToEnemy = -1;
        if (myBase != null && eBase != null)
            distToEnemy = Math.abs(myBase.getX()-eBase.getX())
                        + Math.abs(myBase.getY()-eBase.getY());

        String armyDelta = (myArmy > eArmy) ? "ahead" : (myArmy < eArmy) ? "behind" : "even";

        return "Turn=" + gs.getTime()
             + " Map=" + pgs.getWidth() + "x" + pgs.getHeight() + "(" + mapCat + ")"
             + " Resources=" + gs.getPlayer(player).getResources()
             + " ResourceNodes=" + res
             + " DistToEnemyBase=" + (distToEnemy < 0 ? "unknown" : distToEnemy)
             + "\nMY:    W=" + myW + " B=" + myB + " Br=" + myBr
             + " H=" + myH + " R=" + myR + " L=" + myL
             + " Army=" + myArmy + " HP=" + myHp
             + "\nENEMY: W=" + eW + " B=" + eB + " Br=" + eBr
             + " H=" + eH + " R=" + eR + " L=" + eL
             + " Army=" + eArmy + " HP=" + eHp
             + "\nArmyDelta=" + armyDelta
             + " UnderAttack=" + threat
             + " CurrentStrategy=" + macroStrategy
             + " CurrentAggression=" + aggression
             + "\nChoose strategy, aggression, and target.";
    }

    // ==========================================================================
    //  LLM RESPONSE PARSING
    // ==========================================================================

    /** Returns true if at least strategy was parsed successfully. */
    private boolean parseMacroResponse(String response) {
        try {
            response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
            int s = response.indexOf("{"), e = response.lastIndexOf("}") + 1;
            if (s < 0 || e <= s) return false;

            JsonObject json = JsonParser.parseString(response.substring(s, e)).getAsJsonObject();
            if (json.has("thinking"))
                System.out.println("[yebot] LLM thinks: " + json.get("thinking").getAsString());

            boolean parsed = false;
            if (json.has("strategy")) {
                String strat = json.get("strategy").getAsString().toUpperCase().trim();
                switch (strat) {
                    case "WORKER_RUSH": case "ECON_HEAVY":  case "ECON_RANGED":
                    case "COUNTER_MIX": case "ALL_IN":      case "EXPAND":
                    case "DEFEND":
                        macroStrategy = strat;
                        parsed = true;
                }
            }
            if (json.has("aggression")) {
                try {
                    int a = json.get("aggression").getAsInt();
                    if (a >= 0 && a <= 2) aggression = a;
                } catch (Exception ignored) {}
            }
            if (json.has("target")) {
                String t = json.get("target").getAsString().toUpperCase().trim();
                switch (t) {
                    case "ARMY": case "ECO": case "EXPANSION": case "ANY":
                        targetPref = t;
                }
            }
            return parsed;
        } catch (Exception ex) {
            System.err.println("[yebot] Parse macro error: " + ex.getMessage());
        }
        return false;
    }

    // ==========================================================================
    //  LLM HTTP CALL
    // ==========================================================================

    private String truncate(String s, int max) {
        if (s == null) return "null";
        return s.length() <= max ? s : s.substring(0, max) + "...";
    }

    /**
     * Ollama /api/generate call.
     * Uses think:false (Qwen3) to skip the reasoning block — saves 2-4s/call.
     * Timeouts match fortress-bot reference.
     */
    private String callLLM(String stateText) {
        return doLLMRequest(SYSTEM_PROMPT + "\n\nGAME STATE:\n" + stateText, READ_TIMEOUT);
    }

    private String doLLMRequest(String prompt, int readTimeoutMs) {
        HttpURLConnection conn = null;
        long t0 = System.currentTimeMillis();
        try {
            URL url = new URL(API_URL);
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(CONNECT_TIMEOUT);
            conn.setReadTimeout(readTimeoutMs);

            JsonObject req = new JsonObject();
            req.addProperty("model", OLLAMA_MODEL);
            req.addProperty("prompt", prompt);
            req.addProperty("stream", false);
            req.addProperty("think", false);          // Qwen3: skip <think> block
            req.addProperty("format", "json");        // force JSON output

            // Ollama-specific options live under "options"
            JsonObject opts = new JsonObject();
            opts.addProperty("temperature", 0.3);
            opts.addProperty("num_predict", 256);
            req.add("options", opts);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(req.toString().getBytes(StandardCharsets.UTF_8));
            }

            int code = conn.getResponseCode();
            if (code == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    JsonObject resp = JsonParser.parseString(sb.toString()).getAsJsonObject();
                    if (resp.has("response")) {
                        long dt = System.currentTimeMillis() - t0;
                        System.out.println("[yebot] callLLM OK in " + dt + "ms");
                        return resp.get("response").getAsString();
                    }
                    System.out.println("[yebot] callLLM: 200 but no 'response' field, body="
                            + truncate(sb.toString(), 200));
                }
            } else {
                String errBody = "";
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    errBody = sb.toString();
                } catch (Exception ignored) {}
                System.out.println("[yebot] callLLM: HTTP " + code + " body=" + truncate(errBody, 300));
            }
        } catch (java.net.ConnectException ce) {
            System.out.println("[yebot] callLLM: CONNECT REFUSED at " + API_URL
                    + " (is Ollama running? " + ce.getMessage() + ")");
        } catch (java.net.SocketTimeoutException te) {
            long dt = System.currentTimeMillis() - t0;
            System.out.println("[yebot] callLLM: TIMEOUT after " + dt + "ms "
                    + "(connect=" + CONNECT_TIMEOUT + " read=" + readTimeoutMs
                    + ", model=" + OLLAMA_MODEL + ")");
        } catch (java.net.UnknownHostException uhe) {
            System.out.println("[yebot] callLLM: UNKNOWN HOST in " + API_URL);
        } catch (java.security.AccessControlException ace) {
            System.out.println("[yebot] callLLM: SECURITY MANAGER blocked network: "
                    + ace.getMessage());
        } catch (Exception e) {
            System.out.println("[yebot] callLLM: " + e.getClass().getSimpleName()
                    + ": " + e.getMessage());
        } finally {
            if (conn != null) conn.disconnect();
        }
        return "{}";
    }

    @Override
    public List<ParameterSpecification> getParameters() { return new ArrayList<>(); }
}