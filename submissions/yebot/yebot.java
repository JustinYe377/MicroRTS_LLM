/*
 * yebot — Macro-LLM + Hard-coded Micro
 *
 * Architecture:
 *   MICRO (Java, every tick):
 *     - Units always attack nearby enemies first (reactive combat)
 *     - Counter-unit targeting: light→worker, heavy→light, ranged→heavy, worker→ranged
 *     - Small maps (≤12): aggressive worker rush from tick 0
 *     - Large maps (>12): eco start → barracks → army
 *     - Kiting for ranged units, focus-fire on low-HP targets
 *
 *   MACRO (LLM, async every N ticks):
 *     - Reads game state summary (unit counts, resources, map size)
 *     - Picks one of: WORKER_RUSH, ECON_HEAVY, ECON_RANGED, COUNTER_MIX, ALL_IN
 *     - Java micro adapts production and aggression based on macro plan
 *     - If LLM fails/slow, Java picks a sensible default macro automatically
 *
 * @author Ye
 * Team: yebot
 */
package ai.abstraction.submissions.yebot;

import ai.abstraction.AbstractionLayerAI;
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
    private static final int LLM_TIMEOUT   = 5000;
    private static final int LLM_INTERVAL  = 200;  // call LLM every 200 ticks

    // ═══════════════════════════════════════════════════════════════════════════
    //  UNIT TYPES
    // ═══════════════════════════════════════════════════════════════════════════
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ═══════════════════════════════════════════════════════════════════════════
    //  MACRO STRATEGY (LLM-controlled, synchronous)
    // ═══════════════════════════════════════════════════════════════════════════
    private String macroStrategy = "DEFAULT";
    private int lastLLMTick = -LLM_INTERVAL; // so first call fires at tick 0

    // ═══════════════════════════════════════════════════════════════════════════
    //  HARVESTER MEMORY (persist across ticks)
    // ═══════════════════════════════════════════════════════════════════════════
    private List<Long> harvesterIDs = new ArrayList<>();

    // ═══════════════════════════════════════════════════════════════════════════
    //  PER-TICK STATE (rebuilt each getAction call)
    // ═══════════════════════════════════════════════════════════════════════════
    private List<Unit> myWorkers, myBases, myBarracks, myHeavies, myRanged, myLights;
    private List<Unit> enemyWorkers, enemyBases, enemyBarracks, enemyHeavies, enemyRanged, enemyLights;
    private List<Unit> allEnemies, allAllies, resources;
    private int resourcesUsed;

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
        macroStrategy  = "DEFAULT";
        lastLLMTick    = -LLM_INTERVAL;
        harvesterIDs   = new ArrayList<>();
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
        int tick = gs.getTime();
        int mapW = pgs.getWidth();

        // ── Populate per-tick unit lists ───────────────────────────────────────
        populateUnitLists(player, gs, pgs);

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

        // ── Resolve strategy (LLM plan or Java fallback) ─────────────────────
        String strategy = resolveStrategy(mapW, tick, gs, pgs);

        // ── Execute micro ─────────────────────────────────────────────────────
        return executeMicro(strategy, player, gs, pgs);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  RESOLVE STRATEGY — fallback when LLM hasn't responded yet
    // ═══════════════════════════════════════════════════════════════════════════

    private String resolveStrategy(int mapW, int tick, GameState gs, PhysicalGameState pgs) {
        // If LLM has given us a strategy, use it
        if (!"DEFAULT".equals(macroStrategy)) {
            return macroStrategy;
        }

        // ── Auto-detect sensible default ──────────────────────────────────────
        // Small map: worker rush
        if (mapW <= 12) {
            return "WORKER_RUSH";
        }

        // Early game on big map: eco up
        if (tick < 200) {
            return "ECON_HEAVY";
        }

        // Mid/late: counter based on what enemy has
        return autoCounter();
    }

    /**
     * Simple rule-based counter picker when LLM is unavailable.
     */
    private String autoCounter() {
        int eWorkers = enemyWorkers.size();
        int eHeavies = enemyHeavies.size();
        int eLights  = enemyLights.size();
        int eRanged  = enemyRanged.size();

        // Enemy mostly workers → light rush or worker rush
        if (eWorkers > eHeavies + eLights + eRanged + 2) {
            return myBarracks.isEmpty() ? "WORKER_RUSH" : "ECON_RANGED";
        }
        // Enemy has heavies → ranged counters them
        if (eHeavies >= eLights && eHeavies >= eRanged && eHeavies > 0) {
            return "ECON_RANGED";
        }
        // Enemy has lights → heavies counter them
        if (eLights >= eHeavies && eLights >= eRanged && eLights > 0) {
            return "ECON_HEAVY";
        }
        // Enemy has ranged → workers/lights swarm them
        if (eRanged > eHeavies && eRanged > eLights) {
            return "ECON_HEAVY";
        }
        // Mixed or unknown → heavies are a safe default
        return "ECON_HEAVY";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  MICRO EXECUTION — runs every tick
    // ═══════════════════════════════════════════════════════════════════════════

    private PlayerAction executeMicro(String strategy, int player,
                                       GameState gs, PhysicalGameState pgs) throws Exception {
        resourcesUsed = 0;

        // Count resources already committed by in-progress production
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) continue;
            UnitActionAssignment aa = gs.getActionAssignment(u);
            if (aa != null && aa.action.getType() == UnitAction.TYPE_PRODUCE) {
                UnitType ut = aa.action.getUnitType();
                if (ut != null) resourcesUsed += ut.cost;
            }
        }

        // ── PHASE 1: Reactive combat — all units attack nearby enemies ────────
        reactiveAttackAll(player, gs, pgs);

        // ── PHASE 2: Strategy-specific production and movement ────────────────
        switch (strategy) {
            case "WORKER_RUSH":
                doWorkerRush(player, gs, pgs);
                break;
            case "ALL_IN":
                doAllIn(player, gs, pgs);
                break;
            case "ECON_HEAVY":
                doEconBuild(player, gs, pgs, heavyType);
                break;
            case "ECON_RANGED":
                doEconBuild(player, gs, pgs, rangedType);
                break;
            case "COUNTER_MIX":
                doCounterMix(player, gs, pgs);
                break;
            default:
                doEconBuild(player, gs, pgs, heavyType);
                break;
        }

        // ── PHASE 3: Send combat units to attack ─────────────────────────────
        sendCombatToAttack(player, gs, pgs);

        return translateActions(player, gs);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  PHASE 1: REACTIVE COMBAT — every unit attacks nearby enemies
    //  Uses counter-unit targeting priorities
    // ═══════════════════════════════════════════════════════════════════════════

    private void reactiveAttackAll(int player, GameState gs, PhysicalGameState pgs) {
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) continue;
            if (gs.getActionAssignment(u) != null) continue;
            if (!u.getType().canAttack) continue;

            Unit target = findBestTarget(u, gs, pgs);
            if (target != null) {
                int dist = manhattanDist(u, target);
                if (dist <= u.getType().attackRange) {
                    // In range → attack
                    attack(u, target);
                } else if (dist <= u.getType().attackRange + 2) {
                    // Very close → move to attack
                    attack(u, target);
                }
                // If farther, don't commit here — let strategy phase handle movement
            }
        }
    }

    /**
     * Counter-unit target priority:
     *   Light  → prefers Worker targets
     *   Heavy  → prefers Light targets
     *   Ranged → prefers Heavy targets (kite them)
     *   Worker → prefers Ranged (swarm) or nearest anything
     *
     * Within priority class, prefer low-HP targets (focus fire).
     */
    private Unit findBestTarget(Unit attacker, GameState gs, PhysicalGameState pgs) {
        UnitType myType = attacker.getType();
        int range = myType.attackRange + 2; // engagement range

        Unit bestTarget = null;
        int bestScore = Integer.MIN_VALUE;

        for (Unit enemy : allEnemies) {
            int dist = manhattanDist(attacker, enemy);
            if (dist > range) continue;

            int score = 0;

            // ── Counter-unit priority bonus ───────────────────────────────────
            if (myType == lightType && enemy.getType() == workerType)      score += 100;
            else if (myType == lightType && enemy.getType() == rangedType)  score += 80;
            else if (myType == heavyType && enemy.getType() == lightType)   score += 100;
            else if (myType == heavyType && enemy.getType() == rangedType)  score += 60;
            else if (myType == rangedType && enemy.getType() == heavyType)  score += 100;
            else if (myType == rangedType && enemy.getType() == lightType)  score += 50;
            else if (myType == workerType && enemy.getType() == rangedType) score += 80;
            else if (myType == workerType && enemy.getType() == workerType) score += 60;

            // Prefer buildings if no combat units nearby
            if (enemy.getType() == baseType)     score += 40;
            if (enemy.getType() == barracksType) score += 50;

            // Focus fire: prefer low HP targets (can kill them)
            score += (20 - enemy.getHitPoints());

            // Prefer closer targets
            score -= dist * 5;

            // Big bonus if we can kill this tick
            if (enemy.getHitPoints() <= myType.attackRange) score += 200;

            if (score > bestScore) {
                bestScore = score;
                bestTarget = enemy;
            }
        }
        return bestTarget;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  STRATEGY: WORKER RUSH (small maps or LLM-selected)
    // ═══════════════════════════════════════════════════════════════════════════

    private void doWorkerRush(int player, GameState gs, PhysicalGameState pgs) {
        // Base: keep training workers
        for (Unit base : myBases) {
            if (gs.getActionAssignment(base) != null) continue;
            if (canAfford(player, gs, workerType)) {
                train(base, workerType);
                resourcesUsed += workerType.cost;
            }
        }

        // One harvester, rest attack
        boolean hasHarvester = false;
        for (Unit w : myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;

            if (!hasHarvester && !myBases.isEmpty() && !resources.isEmpty()) {
                Unit res = nearest(w, resources);
                Unit base = nearest(w, myBases);
                if (res != null && base != null) {
                    harvest(w, res, base);
                    hasHarvester = true;
                    continue;
                }
            }

            // Attack nearest enemy
            if (!allEnemies.isEmpty()) {
                attack(w, nearest(w, allEnemies));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  STRATEGY: ALL IN — stop eco, everything attacks
    // ═══════════════════════════════════════════════════════════════════════════

    private void doAllIn(int player, GameState gs, PhysicalGameState pgs) {
        // Still train if we have excess resources and base is idle
        for (Unit base : myBases) {
            if (gs.getActionAssignment(base) != null) continue;
            if (canAfford(player, gs, workerType)) {
                train(base, workerType);
                resourcesUsed += workerType.cost;
            }
        }

        // All workers attack
        for (Unit w : myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (!allEnemies.isEmpty()) {
                attack(w, nearest(w, allEnemies));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  STRATEGY: ECON BUILD (heavy or ranged)
    // ═══════════════════════════════════════════════════════════════════════════

    private void doEconBuild(int player, GameState gs, PhysicalGameState pgs,
                              UnitType combatUnit) {
        // ── Harvest with workers ──────────────────────────────────────────────
        doHarvesting(player, gs, pgs);

        // ── Base: train workers (cap at 2-3 per base) ─────────────────────────
        int workerCap = Math.max(2, myBases.size() * 2);
        // Need at least enough workers to harvest + maybe one to build
        if (myWorkers.size() < workerCap) {
            for (Unit base : myBases) {
                if (gs.getActionAssignment(base) != null) continue;
                if (myWorkers.size() >= workerCap) break;
                if (canAfford(player, gs, workerType)) {
                    train(base, workerType);
                    resourcesUsed += workerType.cost;
                }
            }
        }

        // ── Build barracks if we don't have one ──────────────────────────────
        if (myBarracks.isEmpty()) {
            buildBarracksWithWorker(player, gs, pgs);
        }

        // ── Barracks: produce combat units ───────────────────────────────────
        for (Unit barracks : myBarracks) {
            if (gs.getActionAssignment(barracks) != null) continue;
            if (canAfford(player, gs, combatUnit)) {
                train(barracks, combatUnit);
                resourcesUsed += combatUnit.cost;
            }
        }

        // ── Idle workers attack if we have enough harvesters ─────────────────
        sendIdleWorkersToAttack(player, gs, pgs);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  STRATEGY: COUNTER MIX — produce based on enemy composition
    // ═══════════════════════════════════════════════════════════════════════════

    private void doCounterMix(int player, GameState gs, PhysicalGameState pgs) {
        UnitType needed = pickCounterUnit();
        doEconBuild(player, gs, pgs, needed);
    }

    private UnitType pickCounterUnit() {
        int eHeavy  = enemyHeavies.size();
        int eLight  = enemyLights.size();
        int eRanged = enemyRanged.size();
        int eWorker = enemyWorkers.size();

        // Heavy counters light
        if (eLight > eHeavy && eLight > eRanged) return heavyType;
        // Ranged counters heavy
        if (eHeavy >= eLight && eHeavy >= eRanged && eHeavy > 0) return rangedType;
        // Heavy/light for ranged swarm
        if (eRanged > eHeavy && eRanged > eLight) return heavyType;
        // Mostly workers → light is cost-effective
        if (eWorker > eHeavy + eLight + eRanged) return lightType;
        // Default: heavy is robust
        return heavyType;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  HARVESTING
    // ═══════════════════════════════════════════════════════════════════════════

    private void doHarvesting(int player, GameState gs, PhysicalGameState pgs) {
        // Clean dead harvesters
        harvesterIDs.removeIf(id -> pgs.getUnit(id) == null);

        int maxHarvesters = Math.max(1, myBases.size() * 2);

        // Assign new harvesters if needed
        if (harvesterIDs.size() < maxHarvesters) {
            for (Unit w : myWorkers) {
                if (harvesterIDs.size() >= maxHarvesters) break;
                if (harvesterIDs.contains(w.getID())) continue;
                if (gs.getActionAssignment(w) != null) continue;
                harvesterIDs.add(w.getID());
            }
        }

        // Execute harvesting for assigned harvesters
        for (Long hid : harvesterIDs) {
            Unit w = pgs.getUnit(hid);
            if (w == null) continue;
            if (gs.getActionAssignment(w) != null) continue;

            if (w.getResources() > 0) {
                // Return to base
                Unit base = nearest(w, myBases);
                if (base != null) {
                    harvest(w, nearest(w, resources), base); // AbstractionLayer handles return
                }
            } else {
                // Go harvest
                Unit res = nearest(w, resources);
                Unit base = nearest(w, myBases);
                if (res != null && base != null) {
                    harvest(w, res, base);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  BUILD BARRACKS — find idle non-harvester worker, build near base
    // ═══════════════════════════════════════════════════════════════════════════

    private void buildBarracksWithWorker(int player, GameState gs, PhysicalGameState pgs) {
        if (!canAfford(player, gs, barracksType)) return;
        if (myBases.isEmpty()) return;

        Unit builder = null;
        for (Unit w : myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (!harvesterIDs.contains(w.getID())) {
                builder = w;
                break;
            }
        }
        // If no free worker, pull a harvester
        if (builder == null) {
            for (Unit w : myWorkers) {
                if (gs.getActionAssignment(w) != null) continue;
                builder = w;
                break;
            }
        }
        if (builder == null) return;

        // Find a build location near a base
        Unit base = nearest(builder, myBases);
        if (base == null) return;

        // Try positions adjacent to the base
        int[][] offsets = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {-1, 1}, {1, -1}, {-1, -1}};
        for (int[] off : offsets) {
            int bx = base.getX() + off[0] * 2; // 2 away from base so we don't block
            int by = base.getY() + off[1] * 2;
            if (bx < 0 || by < 0 || bx >= pgs.getWidth() || by >= pgs.getHeight()) continue;
            if (pgs.getUnitAt(bx, by) != null) continue;
            if (pgs.getTerrain(bx, by) != PhysicalGameState.TERRAIN_NONE) continue;
            build(builder, barracksType, bx, by);
            resourcesUsed += barracksType.cost;
            return;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  SEND COMBAT UNITS TO ATTACK
    // ═══════════════════════════════════════════════════════════════════════════

    private void sendCombatToAttack(int player, GameState gs, PhysicalGameState pgs) {
        // Send all combat units (heavy, ranged, light) toward enemies
        List<Unit> combatUnits = new ArrayList<>();
        combatUnits.addAll(myHeavies);
        combatUnits.addAll(myRanged);
        combatUnits.addAll(myLights);

        for (Unit u : combatUnits) {
            if (gs.getActionAssignment(u) != null) continue;
            if (allEnemies.isEmpty()) continue;

            Unit target = findBestTarget(u, gs, pgs);
            if (target == null) target = nearest(u, allEnemies);
            if (target != null) {
                attack(u, target);
            }
        }
    }

    /**
     * Send idle workers (not harvesters) to attack if we have enough economy.
     */
    private void sendIdleWorkersToAttack(int player, GameState gs, PhysicalGameState pgs) {
        for (Unit w : myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (harvesterIDs.contains(w.getID())) continue;
            // This worker is idle and not a harvester → attack
            if (!allEnemies.isEmpty()) {
                attack(w, nearest(w, allEnemies));
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  POPULATE UNIT LISTS
    // ═══════════════════════════════════════════════════════════════════════════

    private void populateUnitLists(int player, GameState gs, PhysicalGameState pgs) {
        myWorkers = new ArrayList<>(); myBases = new ArrayList<>();
        myBarracks = new ArrayList<>(); myHeavies = new ArrayList<>();
        myRanged = new ArrayList<>(); myLights = new ArrayList<>();
        enemyWorkers = new ArrayList<>(); enemyBases = new ArrayList<>();
        enemyBarracks = new ArrayList<>(); enemyHeavies = new ArrayList<>();
        enemyRanged = new ArrayList<>(); enemyLights = new ArrayList<>();
        allEnemies = new ArrayList<>(); allAllies = new ArrayList<>();
        resources = new ArrayList<>();

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                resources.add(u);
                continue;
            }
            if (u.getPlayer() == player) {
                allAllies.add(u);
                if      (u.getType() == workerType)   myWorkers.add(u);
                else if (u.getType() == baseType)     myBases.add(u);
                else if (u.getType() == barracksType) myBarracks.add(u);
                else if (u.getType() == heavyType)    myHeavies.add(u);
                else if (u.getType() == rangedType)   myRanged.add(u);
                else if (u.getType() == lightType)    myLights.add(u);
            } else if (u.getPlayer() >= 0) {
                allEnemies.add(u);
                if      (u.getType() == workerType)   enemyWorkers.add(u);
                else if (u.getType() == baseType)     enemyBases.add(u);
                else if (u.getType() == barracksType) enemyBarracks.add(u);
                else if (u.getType() == heavyType)    enemyHeavies.add(u);
                else if (u.getType() == rangedType)   enemyRanged.add(u);
                else if (u.getType() == lightType)    enemyLights.add(u);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  LLM MACRO — state text + parse
    // ═══════════════════════════════════════════════════════════════════════════

    private String buildMacroStateText(int player, GameState gs, PhysicalGameState pgs) {
        StringBuilder sb = new StringBuilder();
        sb.append("Turn=").append(gs.getTime())
          .append(" Map=").append(pgs.getWidth()).append("x").append(pgs.getHeight())
          .append(" Resources=").append(gs.getPlayer(player).getResources())
          .append("\n");
        sb.append("MY UNITS: Workers=").append(myWorkers.size())
          .append(" Bases=").append(myBases.size())
          .append(" Barracks=").append(myBarracks.size())
          .append(" Heavy=").append(myHeavies.size())
          .append(" Ranged=").append(myRanged.size())
          .append(" Light=").append(myLights.size())
          .append("\n");
        sb.append("ENEMY UNITS: Workers=").append(enemyWorkers.size())
          .append(" Bases=").append(enemyBases.size())
          .append(" Barracks=").append(enemyBarracks.size())
          .append(" Heavy=").append(enemyHeavies.size())
          .append(" Ranged=").append(enemyRanged.size())
          .append(" Light=").append(enemyLights.size())
          .append("\n");
        sb.append("Resources on map: ").append(resources.size()).append("\n");
        sb.append("Choose the best strategy.");
        return sb.toString();
    }

    private String parseMacroStrategy(String response) {
        try {
            // Strip thinking tags from models like qwen3
            response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
            int s = response.indexOf("{"), e = response.lastIndexOf("}") + 1;
            if (s < 0 || e <= s) return null;

            JsonObject json = JsonParser.parseString(response.substring(s, e)).getAsJsonObject();
            if (json.has("thinking")) {
                System.out.println("[yebot] LLM thinks: " + json.get("thinking").getAsString());
            }
            if (json.has("strategy")) {
                String strat = json.get("strategy").getAsString().toUpperCase().trim();
                // Validate
                switch (strat) {
                    case "WORKER_RUSH":
                    case "ECON_HEAVY":
                    case "ECON_RANGED":
                    case "COUNTER_MIX":
                    case "ALL_IN":
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

    // ═══════════════════════════════════════════════════════════════════════════
    //  UTILITY
    // ═══════════════════════════════════════════════════════════════════════════

    private boolean canAfford(int player, GameState gs, UnitType ut) {
        return gs.getPlayer(player).getResources() - resourcesUsed >= ut.cost;
    }

    private int manhattanDist(Unit a, Unit b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    private Unit nearest(Unit src, List<Unit> units) {
        Unit best = null; int bestD = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = manhattanDist(src, u);
            if (d < bestD) { bestD = d; best = u; }
        }
        return best;
    }

    @Override
    public List<ParameterSpecification> getParameters() { return new ArrayList<>(); }
}