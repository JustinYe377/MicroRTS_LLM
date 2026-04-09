/*
 * yebot — Macro-LLM + Hard-coded Micro  (v2 — Large Map Edition)
 *
 * Architecture:
 *   MICRO (Java, every tick):
 *     - Small maps (<=8): worker rush
 *     - Medium/Large maps: eco -> expand -> army
 *     - Adaptive worker / harvester / barracks caps scaled to map size
 *     - Priority targeting: ranged > light > heavy > worker > barracks > base
 *     - Buildings placed near own base (not scattered at worker position)
 *     - Emergency DEFEND override when enemy enters base perimeter
 *     - Strategy decay: auto-revert to counter logic if LLM stalls
 *
 *   MACRO (LLM, synchronous every N ticks -- scaled with map size):
 *     - Reads richer state (map category, army HP, threat status, resource nodes)
 *     - Picks one of: WORKER_RUSH, ECON_HEAVY, ECON_RANGED, COUNTER_MIX,
 *                     ALL_IN, EXPAND, DEFEND
 *     - If LLM fails / slow, Java picks a sensible default automatically
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

    // ===========================================================================================
    //  CONFIG
    // ===========================================================================================
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "qwen3:8b";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int LLM_TIMEOUT    = 5000;
    private static final int LLM_INTERVAL   = 200;  // base interval; scaled per map size
    private static final int STRATEGY_DECAY = 800;  // ticks until LLM strategy reverts to auto

    // ===========================================================================================
    //  UNIT TYPES
    // ===========================================================================================
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ===========================================================================================
    //  MACRO STRATEGY (LLM-controlled)
    // ===========================================================================================
    private String macroStrategy   = "DEFAULT";
    private int    lastLLMTick     = -LLM_INTERVAL;
    private int    strategySetTick = 0;

    // ===========================================================================================
    //  LLM SYSTEM PROMPT -- extended with EXPAND and DEFEND
    // ===========================================================================================
    private static final String SYSTEM_PROMPT =
        "You are a MicroRTS macro strategist. Given the game state, choose ONE strategy.\n"
      + "UNITS: Worker(HP=1,dmg=1,cost=1) Light(HP=4,dmg=2,cost=2) "
      + "Heavy(HP=8,dmg=4,cost=3) Ranged(HP=3,dmg=1,range=3,cost=2)\n"
      + "COUNTER LOGIC: Light beats Worker. Heavy beats Light. Ranged beats Heavy. Workers swarm Ranged.\n"
      + "STRATEGIES:\n"
      + "- WORKER_RUSH: Send all workers to attack. Best on small maps or when far ahead in workers.\n"
      + "- ECON_HEAVY: Build barracks, produce Heavies. Good vs Light-heavy enemy.\n"
      + "- ECON_RANGED: Build barracks, produce Ranged. Good vs Heavy-heavy enemy.\n"
      + "- COUNTER_MIX: Produce whatever counters current enemy composition.\n"
      + "- ALL_IN: Stop eco, send everything to attack. Use when you have a clear army advantage.\n"
      + "- EXPAND: Build a second base and scale workers/harvesters. Best on large/huge maps early game.\n"
      + "- DEFEND: Hold position, keep workers harvesting, wait for counter-attack. Use when under attack.\n"
      + "OUTPUT FORMAT (JSON only): {\"thinking\":\"brief reason\",\"strategy\":\"STRATEGY_NAME\"}\n";

    // ===========================================================================================
    //  CONSTRUCTORS
    // ===========================================================================================

    public yebot(UnitTypeTable a_utt) { this(a_utt, new AStarPathFinding()); }

    public yebot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        macroStrategy   = "DEFAULT";
        lastLLMTick     = -LLM_INTERVAL;
        strategySetTick = 0;
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

    // ===========================================================================================
    //  MAIN LOOP
    // ===========================================================================================

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs  = gs.getPhysicalGameState();
        Player            p    = gs.getPlayer(player);
        int               tick = gs.getTime();

        // -- Every N ticks (scaled to map size): call LLM --------------------------------
        int interval = adaptiveLLMInterval(pgs);
        if (tick - lastLLMTick >= interval) {
            lastLLMTick = tick;
            try {
                String stateText = buildMacroStateText(player, gs, pgs);
                String response  = callLLM(stateText);
                String parsed    = parseMacroStrategy(response);
                if (parsed != null) {
                    macroStrategy   = parsed;
                    strategySetTick = tick;
                    System.out.println("[yebot] t=" + tick + " LLM -> " + parsed);
                }
            } catch (Exception e) {
                System.err.println("[yebot] LLM error: " + e.getMessage());
            }
        }

        // -- Strategy decay: revert to auto after STRATEGY_DECAY ticks ------------------
        if (!"DEFAULT".equals(macroStrategy) && (tick - strategySetTick > STRATEGY_DECAY)) {
            macroStrategy = "DEFAULT";
            System.out.println("[yebot] t=" + tick + " strategy decayed -> DEFAULT");
        }

        // -- Resolve strategy ------------------------------------------------------------
        String strategy = resolveStrategy(tick, player, gs, pgs);

        // -- Execute ---------------------------------------------------------------------
        switch (strategy) {
            case "WORKER_RUSH": executeWorkerRush(player, p, gs, pgs);                          break;
            case "ALL_IN":      executeAllIn(player, p, gs, pgs);                               break;
            case "ECON_HEAVY":  executeEcon(player, p, gs, pgs, heavyType);                     break;
            case "ECON_RANGED": executeEcon(player, p, gs, pgs, rangedType);                    break;
            case "COUNTER_MIX": executeEcon(player, p, gs, pgs, pickCounterUnit(pgs, player));  break;
            case "EXPAND":      executeExpand(player, p, gs, pgs);                              break;
            case "DEFEND":      executeDefend(player, p, gs, pgs);                              break;
            default:            executeEcon(player, p, gs, pgs, heavyType);                     break;
        }

        return translateActions(player, gs);
    }

    // ===========================================================================================
    //  MAP SIZE HELPERS
    // ===========================================================================================

    private int     mapArea(PhysicalGameState pgs)     { return pgs.getWidth() * pgs.getHeight(); }
    private boolean isMediumMap(PhysicalGameState pgs)  { return mapArea(pgs) > 144; }  // > 12x12
    private boolean isLargeMap(PhysicalGameState pgs)   { return mapArea(pgs) > 256; }  // > 16x16
    private boolean isHugeMap(PhysicalGameState pgs)    { return mapArea(pgs) > 576; }  // > 24x24

    /** Larger maps get more frequent LLM advice (more strategic complexity). */
    private int adaptiveLLMInterval(PhysicalGameState pgs) {
        if (isHugeMap(pgs))  return 150;
        if (isLargeMap(pgs)) return 175;
        return LLM_INTERVAL;
    }

    /** Max workers to maintain, scaling with base count and map size. */
    private int maxWorkers(PhysicalGameState pgs, int nbases) {
        int perBase = isHugeMap(pgs) ? 6 : isLargeMap(pgs) ? 5 : isMediumMap(pgs) ? 4 : 3;
        return perBase * Math.max(1, nbases);
    }

    /** Max barracks to build, scaling with map size. */
    private int maxBarracks(PhysicalGameState pgs) {
        if (isHugeMap(pgs))  return 3;
        if (isLargeMap(pgs)) return 2;
        return 1;
    }

    /** Max harvesters, capped by available resource nodes and map size. */
    private int maxHarvesters(PhysicalGameState pgs, int resourceNodes) {
        int cap = isHugeMap(pgs) ? 5 : isLargeMap(pgs) ? 4 : isMediumMap(pgs) ? 3 : 2;
        return Math.min(cap, Math.max(1, resourceNodes));
    }

    private int countResourceNodes(PhysicalGameState pgs) {
        int count = 0;
        for (Unit u : pgs.getUnits()) if (u.getType().isResource) count++;
        return count;
    }

    // ===========================================================================================
    //  RESOLVE STRATEGY
    // ===========================================================================================

    private String resolveStrategy(int tick, int player, GameState gs,
                                    PhysicalGameState pgs) {
        // Emergency override: enemy at our base gates -> DEFEND
        if (isUnderAttack(player, pgs, Math.max(4, pgs.getWidth() / 4))) {
            return "DEFEND";
        }

        if (!"DEFAULT".equals(macroStrategy)) return macroStrategy;

        // Default fallbacks scaled to map size
        int mapW = pgs.getWidth();
        if (mapW <= 8) return "WORKER_RUSH";

        if (mapW <= 12) {
            return tick < 150 ? "ECON_HEAVY" : autoCounter(pgs, player);
        }

        // Large/huge maps: expand first, then build army
        if (tick < 100)  return "EXPAND";
        if (tick < 400)  return "ECON_HEAVY";
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
        if (eL > eH && eL > eR)              return heavyType;
        if (eH >= eL && eH >= eR && eH > 0) return rangedType;
        if (eW > eH + eL + eR)              return lightType;
        return heavyType;
    }

    // ===========================================================================================
    //  EMERGENCY DETECTION
    // ===========================================================================================

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

    // ===========================================================================================
    //  BUILD POSITION HELPER
    //  Pass base coordinates as hint so buildIfNotAlreadyBuilding places
    //  structures near the base, not scattered wherever the worker stands.
    // ===========================================================================================

    private int[] nearestBasePos(int player, Unit fallback, PhysicalGameState pgs) {
        int best = Integer.MAX_VALUE;
        int[] pos = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player) {
                int d = Math.abs(u.getX() - fallback.getX())
                      + Math.abs(u.getY() - fallback.getY());
                if (d < best) { best = d; pos = new int[]{u.getX(), u.getY()}; }
            }
        }
        return pos != null ? pos : new int[]{fallback.getX(), fallback.getY()};
    }

    // ===========================================================================================
    //  PRIORITY TARGETING
    //  Attack highest-threat enemy first; ties resolved by proximity.
    // ===========================================================================================

    private void priorityAttack(Unit u, int player, PhysicalGameState pgs) {
        Unit target   = null;
        int  bestPri  = -1;
        int  bestDist = Integer.MAX_VALUE;

        for (Unit enemy : pgs.getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player) continue;
            int pri;
            if      (enemy.getType() == rangedType)   pri = 6;
            else if (enemy.getType() == lightType)    pri = 5;
            else if (enemy.getType() == heavyType)    pri = 4;
            else if (enemy.getType() == workerType)   pri = 3;
            else if (enemy.getType() == barracksType) pri = 2;
            else if (enemy.getType() == baseType)     pri = 1;
            else                                      pri = 0;

            int d = Math.abs(enemy.getX() - u.getX()) + Math.abs(enemy.getY() - u.getY());
            if (pri > bestPri || (pri == bestPri && d < bestDist)) {
                bestPri  = pri;
                bestDist = d;
                target   = enemy;
            }
        }
        if (target != null) attack(u, target);
    }

    /** Route all combat units through priority targeting. */
    private void combatBehavior(Unit u, int player, PhysicalGameState pgs) {
        priorityAttack(u, player, pgs);
    }

    // ===========================================================================================
    //  STRATEGY: WORKER RUSH
    // ===========================================================================================

    private void executeWorkerRush(int player, Player p, GameState gs,
                                    PhysicalGameState pgs) {
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
                combatBehavior(u, player, pgs);
            }
        }
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits())
            if (u.getType().canHarvest && u.getPlayer() == player) workers.add(u);
        workerRushBehavior(workers, player, p, gs, pgs);
    }

    /**
     * Exact same logic as built-in WorkerRush: build base if lost,
     * keep 1 harvester (checked via getAbstractAction), rest attack.
     */
    private void workerRushBehavior(List<Unit> workers, int player, Player p,
                                     GameState gs, PhysicalGameState pgs) {
        int nbases = 0, resourcesUsed = 0;
        Unit harvestWorker = null;
        List<Unit> freeWorkers = new LinkedList<>(workers);
        if (workers.isEmpty()) return;

        for (Unit u : pgs.getUnits())
            if (u.getType() == baseType && u.getPlayer() == p.getID()) nbases++;

        List<Integer> reservedPositions = new LinkedList<>();

        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = nearestBasePos(player, u, pgs);
                buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1],
                        reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        if (!freeWorkers.isEmpty()) harvestWorker = freeWorkers.remove(0);
        if (harvestWorker != null && !doHarvest(harvestWorker, p, pgs))
            freeWorkers.add(harvestWorker);

        for (Unit u : freeWorkers) priorityAttack(u, p.getID(), pgs);
    }

    // ===========================================================================================
    //  STRATEGY: ALL IN
    // ===========================================================================================

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
                combatBehavior(u, player, pgs);
            }
        }
    }

    // ===========================================================================================
    //  STRATEGY: ECON BUILD (heavy, ranged, or counter)
    //  Adaptive caps: more workers / harvesters / barracks on larger maps.
    // ===========================================================================================

    private void executeEcon(int player, Player p, GameState gs,
                              PhysicalGameState pgs, UnitType combatUnit) {
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

        int wCap = maxWorkers(pgs, nbases);
        int bCap = maxBarracks(pgs);

        // Train workers across all bases up to adaptive cap
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

        // Combat units: priority attack dispatch
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, pgs);
            }
        }

        // Workers
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits())
            if (u.getType().canHarvest && u.getPlayer() == player) workers.add(u);

        econWorkerBehavior(workers, player, p, gs, pgs, nbarracks, bCap,
                resourcesUsed, resourceNodes);
    }

    private void econWorkerBehavior(List<Unit> workers, int player, Player p,
                                     GameState gs, PhysicalGameState pgs,
                                     int nbarracks, int bCap,
                                     int resourcesUsed, int resourceNodes) {
        int nbases = 0;
        List<Unit> freeWorkers = new LinkedList<>(workers);
        if (freeWorkers.isEmpty()) return;

        for (Unit u : pgs.getUnits())
            if (u.getType() == baseType && u.getPlayer() == p.getID()) nbases++;

        List<Integer> reservedPositions = new LinkedList<>();

        // Rebuild base if lost
        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = nearestBasePos(player, u, pgs);
                buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1],
                        reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        // Build barracks up to adaptive cap, placed near base
        int toBuild = bCap - nbarracks;
        for (int i = 0; i < toBuild && !freeWorkers.isEmpty(); i++) {
            if (p.getResources() >= barracksType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = nearestBasePos(player, u, pgs);
                buildIfNotAlreadyBuilding(u, barracksType, bp[0], bp[1],
                        reservedPositions, p, pgs);
                resourcesUsed += barracksType.cost;
            }
        }

        // Assign adaptive harvesters
        int hCap = maxHarvesters(pgs, resourceNodes);
        List<Unit> harvesters = new ArrayList<>();
        for (int i = 0; i < hCap && !freeWorkers.isEmpty(); i++)
            harvesters.add(freeWorkers.remove(0));

        for (Unit hw : harvesters)
            if (!doHarvest(hw, p, pgs)) freeWorkers.add(hw);

        for (Unit w : freeWorkers) priorityAttack(w, player, pgs);
    }

    // ===========================================================================================
    //  STRATEGY: EXPAND
    //  Build a second base near the richest resource cluster; maximise eco first.
    // ===========================================================================================

    private void executeExpand(int player, Player p, GameState gs,
                                PhysicalGameState pgs) {
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

        int wCap        = maxWorkers(pgs, nbases);
        int targetBases = isLargeMap(pgs) ? 2 : 1;

        // Train workers from existing bases
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

        // Combat units still press forward
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, pgs);
            }
        }

        // Workers
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
        List<Integer> reservedPositions = new LinkedList<>();

        // Build first base if missing
        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = nearestBasePos(player, u, pgs);
                buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1],
                        reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
                nbases++;
            }
        }

        // Expand: second base near farthest resources from our base
        if (nbases < targetBases && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] ep = expansionPos(player, pgs, reservedPositions);
                buildIfNotAlreadyBuilding(u, baseType, ep[0], ep[1],
                        reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        // Build barracks if none
        if (nbarracks == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= barracksType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = nearestBasePos(player, u, pgs);
                buildIfNotAlreadyBuilding(u, barracksType, bp[0], bp[1],
                        reservedPositions, p, pgs);
                resourcesUsed += barracksType.cost;
            }
        }

        // Max harvesters
        int hCap = maxHarvesters(pgs, resourceNodes);
        List<Unit> harvesters = new ArrayList<>();
        for (int i = 0; i < hCap && !freeWorkers.isEmpty(); i++)
            harvesters.add(freeWorkers.remove(0));

        for (Unit hw : harvesters)
            if (!doHarvest(hw, p, pgs)) freeWorkers.add(hw);

        for (Unit w : freeWorkers) priorityAttack(w, player, pgs);
    }

    /**
     * Choose expansion coordinates: near the resource node farthest from our base.
     * AbstractionLayerAI.findBuildingPosition will locate the actual free cell.
     */
    private int[] expansionPos(int player, PhysicalGameState pgs,
                                List<Integer> reserved) {
        int bx = pgs.getWidth() / 2, by = pgs.getHeight() / 2;
        for (Unit u : pgs.getUnits())
            if (u.getType() == baseType && u.getPlayer() == player)
                { bx = u.getX(); by = u.getY(); break; }

        int targetX = pgs.getWidth()  - 1 - bx;
        int targetY = pgs.getHeight() - 1 - by;
        int bestD   = -1;
        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                int d = Math.abs(u.getX() - bx) + Math.abs(u.getY() - by);
                if (d > bestD) { bestD = d; targetX = u.getX(); targetY = u.getY(); }
            }
        }
        return new int[]{targetX, targetY};
    }

    // ===========================================================================================
    //  STRATEGY: DEFEND
    //  Workers harvest; military defends; build barracks if absent.
    // ===========================================================================================

    private void executeDefend(int player, Player p, GameState gs,
                                PhysicalGameState pgs) {
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

        // Train counter units from barracks
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

        // Maintain minimum 2 workers so eco doesn't die
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (nworkers < 2 && p.getResources() - resourcesUsed >= workerType.cost) {
                    train(u, workerType);
                    resourcesUsed += workerType.cost;
                }
            }
        }

        // Military defends
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                combatBehavior(u, player, pgs);
            }
        }

        // Workers: build missing structures, then harvest
        List<Unit> freeWorkers = new LinkedList<>();
        for (Unit u : pgs.getUnits())
            if (u.getType().canHarvest && u.getPlayer() == player) freeWorkers.add(u);

        List<Integer> reservedPositions = new LinkedList<>();

        if (nbases == 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= baseType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = nearestBasePos(player, u, pgs);
                buildIfNotAlreadyBuilding(u, baseType, bp[0], bp[1],
                        reservedPositions, p, pgs);
                resourcesUsed += baseType.cost;
            }
        }

        if (nbarracks == 0 && nbases > 0 && !freeWorkers.isEmpty()) {
            if (p.getResources() >= barracksType.cost + resourcesUsed) {
                Unit u   = freeWorkers.remove(0);
                int[] bp = nearestBasePos(player, u, pgs);
                buildIfNotAlreadyBuilding(u, barracksType, bp[0], bp[1],
                        reservedPositions, p, pgs);
            }
        }

        int hCap     = maxHarvesters(pgs, resourceNodes);
        int assigned = 0;
        for (Unit w : freeWorkers) {
            if (assigned < hCap && doHarvest(w, p, pgs)) assigned++;
            else priorityAttack(w, player, pgs);
        }
    }

    // ===========================================================================================
    //  HARVEST -- never cancels an in-progress Harvest action
    // ===========================================================================================

    private boolean doHarvest(Unit hw, Player p, PhysicalGameState pgs) {
        Unit closestBase     = null;
        Unit closestResource = null;
        int  closestDist     = 0;

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType().isResource) {
                int d = Math.abs(u2.getX() - hw.getX()) + Math.abs(u2.getY() - hw.getY());
                if (closestResource == null || d < closestDist) {
                    closestResource = u2; closestDist = d;
                }
            }
        }
        closestDist = 0;
        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType().isStockpile && u2.getPlayer() == p.getID()) {
                int d = Math.abs(u2.getX() - hw.getX()) + Math.abs(u2.getY() - hw.getY());
                if (closestBase == null || d < closestDist) {
                    closestBase = u2; closestDist = d;
                }
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

    // ===========================================================================================
    //  LLM STATE TEXT -- richer context for better decisions
    // ===========================================================================================

    private String buildMacroStateText(int player, GameState gs,
                                        PhysicalGameState pgs) {
        int myW = 0, myB = 0, myBr = 0, myH = 0, myR = 0, myL = 0, myHp = 0;
        int eW  = 0, eB  = 0, eBr  = 0, eH  = 0, eR  = 0, eL  = 0, eHp  = 0;
        int res = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) { res++; continue; }
            if (u.getPlayer() == player) {
                myHp += u.getHitPoints();
                if      (u.getType() == workerType)   myW++;
                else if (u.getType() == baseType)     myB++;
                else if (u.getType() == barracksType) myBr++;
                else if (u.getType() == heavyType)    myH++;
                else if (u.getType() == rangedType)   myR++;
                else if (u.getType() == lightType)    myL++;
            } else if (u.getPlayer() >= 0) {
                eHp += u.getHitPoints();
                if      (u.getType() == workerType)   eW++;
                else if (u.getType() == baseType)     eB++;
                else if (u.getType() == barracksType) eBr++;
                else if (u.getType() == heavyType)    eH++;
                else if (u.getType() == rangedType)   eR++;
                else if (u.getType() == lightType)    eL++;
            }
        }

        String  mapCat = isHugeMap(pgs) ? "huge" : isLargeMap(pgs) ? "large"
                       : isMediumMap(pgs) ? "medium" : "small";
        boolean threat = isUnderAttack(player, pgs, pgs.getWidth() / 4);
        int     myArmy = myH + myR + myL;
        int     eArmy  = eH  + eR  + eL;

        return "Turn=" + gs.getTime()
             + " Map=" + pgs.getWidth() + "x" + pgs.getHeight() + "(" + mapCat + ")"
             + " Resources=" + gs.getPlayer(player).getResources()
             + " ResourceNodes=" + res
             + "\nMY:    W=" + myW + " B=" + myB + " Br=" + myBr
             + " H=" + myH + " R=" + myR + " L=" + myL
             + " Army=" + myArmy + " HP=" + myHp
             + "\nENEMY: W=" + eW + " B=" + eB + " Br=" + eBr
             + " H=" + eH + " R=" + eR + " L=" + eL
             + " Army=" + eArmy + " HP=" + eHp
             + "\nUnderAttack=" + threat
             + " CurrentStrategy=" + macroStrategy
             + "\nChoose the best strategy.";
    }

    // ===========================================================================================
    //  LLM -- parse strategy
    // ===========================================================================================

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
                    case "WORKER_RUSH": case "ECON_HEAVY":  case "ECON_RANGED":
                    case "COUNTER_MIX": case "ALL_IN":      case "EXPAND":
                    case "DEFEND":
                        return strat;
                }
            }
        } catch (Exception ex) {
            System.err.println("[yebot] Parse macro error: " + ex.getMessage());
        }
        return null;
    }

    // ===========================================================================================
    //  LLM HTTP CALL  (model / URL / timeout untouched)
    // ===========================================================================================

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
                    JsonObject resp    = JsonParser.parseString(sb.toString()).getAsJsonObject();
                    JsonArray  choices = resp.getAsJsonArray("choices");
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
