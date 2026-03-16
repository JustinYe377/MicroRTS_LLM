/*
 * yerule — Phase-Based Strategy Agent with Async LLM Hints
 *
 * Design philosophy:
 *   Hard-coded rules handle EXECUTION (fast, deterministic, no latency).
 *   LLM handles STRATEGY (slow, async, consulted rarely).
 *   The LLM never blocks the game loop — it runs in a background thread
 *   and deposits a strategy hint that the rule engine reads.
 *
 * Three game phases:
 *
 *   OPENING  (ticks 0–threshold)
 *     Small map (≤144 tiles): Worker rush — flood enemy base with workers.
 *       - 1 worker always harvesting to keep economy alive
 *       - All other workers attack immediately
 *       - Base spams workers non-stop
 *       - No barracks, no ranged units — pure numbers
 *     Large map: Economy first — 2 workers harvest, build barracks ASAP.
 *
 *   MIDGAME  (have barracks, army < ATTACK_THRESHOLD)
 *     - Keep 1–2 workers harvesting
 *     - Barracks trains unit type from LLM hint (default: ranged)
 *     - Army holds near base until strong enough
 *     - LLM consulted for unit composition advice
 *
 *   LATEGAME (army ≥ ATTACK_THRESHOLD or tick > LATE_TICK)
 *     - Workers keep harvesting
 *     - Full army pushes enemy base
 *     - LLM consulted for timing and target priority
 *
 * Speed design:
 *   - No LLM calls in the game loop — all LLM calls are async background
 *   - Pure Java rule execution — microseconds per tick
 *   - LLM called at most once per LLM_INTERVAL ticks
 *   - translateActions() used for AbstractionLayerAI compatibility
 *
 * @author Ye
 * Team: yerule
 */
package ai.abstraction.submissions.yerule;

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
import java.util.concurrent.*;

public class yerule extends AbstractionLayerAI {

    // ─── API Config ────────────────────────────────────────────────────────────
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "llama4:latest";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int LLM_TIMEOUT    = 10000; // 10s max per LLM call
    private static final int LLM_INTERVAL   = 100;   // ticks between LLM consultations

    // ─── Phase Thresholds ─────────────────────────────────────────────────────
    private static final int SMALL_MAP_TILES    = 144;  // ≤ 12x12 = rush map
    private static final int ATTACK_THRESHOLD   = 4;    // army units before full push
    private static final int LATE_TICK          = 600;  // force lategame at this tick
    private static final int MAX_HARVESTERS     = 2;    // workers dedicated to economy

    // ─── Unit Types ────────────────────────────────────────────────────────────
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ─── Game Phase ───────────────────────────────────────────────────────────
    private enum Phase { OPENING, MIDGAME, LATEGAME }

    // ─── Async LLM State ──────────────────────────────────────────────────────
    private final ExecutorService llmExecutor = Executors.newSingleThreadExecutor();
    private volatile String strategyHint = "";   // last hint from LLM
    private volatile boolean llmRunning  = false;
    private int lastLLMTick = -LLM_INTERVAL;

    // ─── LLM Prompts ──────────────────────────────────────────────────────────

    private static final String MIDGAME_SYSTEM = """
You advise a MicroRTS agent in MIDGAME. Pick the best military unit to train.
Units: ranged (cost 2, range 3, fragile), heavy (cost 3, tanky, slow), light (cost 2, fast, medium HP).
Reply JSON only: {"unit": "ranged"|"heavy"|"light", "reason": "one sentence"}
""";

    private static final String LATEGAME_SYSTEM = """
You advise a MicroRTS agent in LATEGAME. Recommend attack strategy.
Options: "rush_base" (attack enemy base directly), "kill_workers" (deny their economy first),
"kill_army" (destroy military before pushing base).
Reply JSON only: {"target": "rush_base"|"kill_workers"|"kill_army", "reason": "one sentence"}
""";

    // ══════════════════════════════════════════════════════════════════════════
    //  CONSTRUCTORS / RESET
    // ══════════════════════════════════════════════════════════════════════════

    public yerule(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    public yerule(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        strategyHint  = "";
        llmRunning    = false;
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
    public AI clone() { return new yerule(utt, pf); }

    // ══════════════════════════════════════════════════════════════════════════
    //  MAIN ENTRY POINT — always returns instantly
    // ══════════════════════════════════════════════════════════════════════════

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int tick = gs.getTime();
        boolean smallMap = pgs.getWidth() * pgs.getHeight() <= SMALL_MAP_TILES;

        // ── Gather game state snapshot ────────────────────────────────────────
        GameSnapshot snap = new GameSnapshot(player, gs, pgs);

        // ── Determine phase ───────────────────────────────────────────────────
        Phase phase = determinePhase(snap, tick, smallMap);

        // ── Fire async LLM if due (non-blocking) ──────────────────────────────
        if (!llmRunning && tick - lastLLMTick >= LLM_INTERVAL) {
            lastLLMTick = tick;
            fireAsyncLLM(phase, snap);
        }

        // ── Execute phase logic (pure Java, instant) ──────────────────────────
        switch (phase) {
            case OPENING:
                if (smallMap) runOpeningRush(snap, gs, pgs);
                else          runOpeningEconomy(snap, gs, pgs);
                break;
            case MIDGAME:
                runMidgame(snap, gs, pgs);
                break;
            case LATEGAME:
                runLategame(snap, gs, pgs);
                break;
        }

        return translateActions(player, gs);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  PHASE DETERMINATION
    // ══════════════════════════════════════════════════════════════════════════

    private Phase determinePhase(GameSnapshot snap, int tick, boolean smallMap) {
        // Small map: stay in OPENING (rush) until we win or lose
        if (smallMap) return Phase.OPENING;

        // No barracks yet → still opening
        if (snap.myBarracks.isEmpty() && snap.myFutureBarracks == 0) return Phase.OPENING;

        // Large army or late tick → lategame
        if (snap.myArmy.size() >= ATTACK_THRESHOLD || tick >= LATE_TICK) return Phase.LATEGAME;

        return Phase.MIDGAME;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  OPENING — SMALL MAP: WORKER RUSH
    //
    //  Strategy: identical philosophy to CRush but our own implementation.
    //  - Exactly 1 worker harvests (the one closest to a resource+base pair)
    //  - All other workers attack the nearest enemy unit
    //  - Base trains a new worker every time it has ≥1 resource
    //  - No barracks, no ranged — pure worker numbers win on tiny maps
    // ══════════════════════════════════════════════════════════════════════════

    private void runOpeningRush(GameSnapshot snap, GameState gs, PhysicalGameState pgs) {
        // ── Designate harvester ────────────────────────────────────────────────
        // Pick the single worker that is best positioned to harvest efficiently.
        // "Best" = shortest combined distance to nearest resource AND base.
        Unit harvester = null;
        if (!snap.myBases.isEmpty() && !snap.resources.isEmpty()) {
            Unit base = snap.myBases.get(0);
            Unit res  = nearestUnit(base, snap.resources);
            if (res != null) {
                int bestScore = Integer.MAX_VALUE;
                for (Unit w : snap.myWorkers) {
                    int score = dist(w, res) + dist(w, base);
                    if (score < bestScore) { bestScore = score; harvester = w; }
                }
            }
        }

        // ── Harvester logic ────────────────────────────────────────────────────
        if (harvester != null) {
            doHarvest(harvester, snap, gs, pgs);
        }

        // ── All other workers attack ───────────────────────────────────────────
        for (Unit w : snap.myWorkers) {
            if (w == harvester) continue;
            doAttackNearest(w, snap.enemies, gs, pgs);
        }

        // ── Base spams workers ─────────────────────────────────────────────────
        for (Unit base : snap.myBases) {
            if (gs.getActionAssignment(base) != null) continue;
            if (snap.resources(gs) >= workerType.cost) {
                // Spawn toward enemy base for instant attack participation
                doTrain(base, workerType, snap, pgs, gs);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  OPENING — LARGE MAP: ECONOMY FIRST
    //
    //  - 2 workers harvest
    //  - 1 worker builds barracks when resources ≥ 5
    //  - Base trains workers until we have 3
    // ══════════════════════════════════════════════════════════════════════════

    private void runOpeningEconomy(GameSnapshot snap, GameState gs, PhysicalGameState pgs) {
        int res = snap.resources(gs);

        // ── Workers: harvest or build barracks ────────────────────────────────
        int harvesters = 0;
        boolean sentBuilder = false;

        // Assign up to MAX_HARVESTERS workers to harvest, one to build if needed
        for (Unit w : snap.myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;

            boolean needBarracks = snap.myBarracks.isEmpty()
                    && snap.myFutureBarracks == 0
                    && res >= barracksType.cost
                    && !sentBuilder;

            if (needBarracks) {
                // Find a good build position near base
                Pos buildPos = findBarracksPos(snap, pgs);
                if (buildPos != null) {
                    doBuild(w, barracksType, buildPos, gs, pgs);
                    sentBuilder = true;
                    continue;
                }
            }

            if (harvesters < MAX_HARVESTERS) {
                doHarvest(w, snap, gs, pgs);
                harvesters++;
            } else {
                // Extra workers: move toward enemies as light pressure
                doAttackNearest(w, snap.enemies, gs, pgs);
            }
        }

        // ── Base: train workers until threshold ───────────────────────────────
        for (Unit base : snap.myBases) {
            if (gs.getActionAssignment(base) != null) continue;
            if (snap.myWorkers.size() < 3 && res >= workerType.cost) {
                doTrain(base, workerType, snap, pgs, gs);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  MIDGAME
    //
    //  Economy running, building army. LLM advises what unit to train.
    //  Army holds near base until it reaches ATTACK_THRESHOLD.
    // ══════════════════════════════════════════════════════════════════════════

    private void runMidgame(GameSnapshot snap, GameState gs, PhysicalGameState pgs) {
        int res = snap.resources(gs);

        // ── Workers: keep economy running ─────────────────────────────────────
        int harvesters = 0;
        for (Unit w : snap.myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (harvesters < MAX_HARVESTERS) {
                doHarvest(w, snap, gs, pgs);
                harvesters++;
            }
        }

        // ── Base: train more workers if we're short ────────────────────────────
        for (Unit base : snap.myBases) {
            if (gs.getActionAssignment(base) != null) continue;
            if (snap.myWorkers.size() < MAX_HARVESTERS && res >= workerType.cost) {
                doTrain(base, workerType, snap, pgs, gs);
            }
        }

        // ── Barracks: train unit based on LLM hint ────────────────────────────
        UnitType trainType = midgameUnitChoice(snap, res);
        for (Unit barrack : snap.myBarracks) {
            if (gs.getActionAssignment(barrack) != null) continue;
            if (res >= trainType.cost) {
                doTrain(barrack, trainType, snap, pgs, gs);
            }
        }

        // ── Army: hold near base, attack anything that gets close ─────────────
        if (!snap.myBases.isEmpty()) {
            Unit base = snap.myBases.get(0);
            for (Unit u : snap.myArmy) {
                if (gs.getActionAssignment(u) != null) continue;
                // Attack enemies within proximity, otherwise hold
                Unit nearbyEnemy = nearestUnitWithinDist(u, snap.enemies, 4);
                if (nearbyEnemy != null) {
                    doAttack(u, nearbyEnemy, gs, pgs);
                } else {
                    doHoldPosition(u, base, gs, pgs);
                }
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LATEGAME
    //
    //  Full push. LLM advises target priority. Army attacks in waves.
    // ══════════════════════════════════════════════════════════════════════════

    private void runLategame(GameSnapshot snap, GameState gs, PhysicalGameState pgs) {
        int res = snap.resources(gs);

        // ── Workers: keep harvesting for reinforcements ────────────────────────
        int harvesters = 0;
        for (Unit w : snap.myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (harvesters < MAX_HARVESTERS) {
                doHarvest(w, snap, gs, pgs);
                harvesters++;
            }
        }

        // ── Keep training reinforcements ──────────────────────────────────────
        UnitType trainType = midgameUnitChoice(snap, res); // reuse same logic
        for (Unit barrack : snap.myBarracks) {
            if (gs.getActionAssignment(barrack) != null) continue;
            if (res >= trainType.cost) {
                doTrain(barrack, trainType, snap, pgs, gs);
            }
        }

        // ── Determine attack target from LLM hint ─────────────────────────────
        Unit primaryTarget = lategameTarget(snap);

        // ── Army: full push ───────────────────────────────────────────────────
        for (Unit u : snap.myArmy) {
            if (gs.getActionAssignment(u) != null) continue;
            if (primaryTarget != null) {
                doAttack(u, primaryTarget, gs, pgs);
            } else if (!snap.enemies.isEmpty()) {
                doAttackNearest(u, snap.enemies, gs, pgs);
            }
        }

        // ── Workers with combat type also attack in lategame ──────────────────
        for (Unit w : snap.myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (harvesters >= MAX_HARVESTERS && !snap.enemies.isEmpty()) {
                doAttackNearest(w, snap.enemies, gs, pgs);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LLM STRATEGY — ASYNC, NON-BLOCKING
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Fire an LLM consultation in a background thread.
     * The game loop never waits for this — it reads `strategyHint` when ready.
     */
    private void fireAsyncLLM(Phase phase, GameSnapshot snap) {
        if (phase == Phase.OPENING) return; // no LLM needed in opening
        llmRunning = true;

        final String prompt  = buildCompactPrompt(snap, phase);
        final String sysPrompt = phase == Phase.MIDGAME ? MIDGAME_SYSTEM : LATEGAME_SYSTEM;

        llmExecutor.submit(() -> {
            try {
                String response = callLLM(prompt, sysPrompt);
                String hint = parseHint(response, phase);
                if (hint != null && !hint.isEmpty()) {
                    strategyHint = hint;
                    System.out.println("[yerule] LLM hint: " + hint);
                }
            } catch (Exception e) {
                System.err.println("[yerule] LLM failed: " + e.getMessage());
            } finally {
                llmRunning = false;
            }
        });
    }

    /**
     * Compact prompt — ~80 chars, fast to tokenize.
     */
    private String buildCompactPrompt(GameSnapshot snap, Phase phase) {
        return String.format(
            "phase=%s t=%d myArmy=%d myRes=%d enemyArmy=%d enemyBase=%b barracks=%d",
            phase.name(), snap.tick,
            snap.myArmy.size(), snap.cachedRes,
            snap.enemies.size(),
            !snap.enemyBases.isEmpty(),
            snap.myBarracks.size()
        );
    }

    /**
     * Parse LLM response into a single hint string.
     * Returns null on failure — caller keeps using last hint.
     */
    private String parseHint(String response, Phase phase) {
        try {
            response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
            int s = response.indexOf("{"), e = response.lastIndexOf("}") + 1;
            if (s >= 0 && e > s) response = response.substring(s, e);
            JsonObject json = JsonParser.parseString(response).getAsJsonObject();

            if (phase == Phase.MIDGAME && json.has("unit")) {
                return json.get("unit").getAsString().toLowerCase();
            }
            if (phase == Phase.LATEGAME && json.has("target")) {
                return json.get("target").getAsString().toLowerCase();
            }
        } catch (Exception ex) {
            // keep last hint
        }
        return null;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  STRATEGY HELPERS — interpret LLM hints
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Choose unit type to train in midgame based on LLM hint + resource check.
     * Default: ranged (cheapest with range advantage).
     */
    private UnitType midgameUnitChoice(GameSnapshot snap, int res) {
        String hint = strategyHint;

        if ("heavy".equals(hint) && res >= heavyType.cost) return heavyType;
        if ("light".equals(hint) && res >= lightType.cost) return lightType;

        // Default: ranged — cheapest, handles both melee and ranged enemies
        if (res >= rangedType.cost) return rangedType;
        if (res >= lightType.cost)  return lightType;
        return rangedType; // will wait for resources
    }

    /**
     * Choose primary attack target in lategame based on LLM hint.
     * "rush_base"    → enemy base
     * "kill_workers" → enemy workers first
     * "kill_army"    → enemy military first
     * Default: enemy base (fastest win condition)
     */
    private Unit lategameTarget(GameSnapshot snap) {
        String hint = strategyHint;

        if ("kill_workers".equals(hint) && !snap.enemyWorkers.isEmpty()) {
            // Target enemy worker with lowest HP
            return snap.enemyWorkers.stream()
                    .min(Comparator.comparingInt(Unit::getHitPoints))
                    .orElse(null);
        }
        if ("kill_army".equals(hint) && !snap.enemyArmy.isEmpty()) {
            // Target enemy military unit with lowest HP
            return snap.enemyArmy.stream()
                    .min(Comparator.comparingInt(Unit::getHitPoints))
                    .orElse(null);
        }

        // Default: rush enemy base
        if (!snap.enemyBases.isEmpty()) return snap.enemyBases.get(0);
        if (!snap.enemies.isEmpty())    return snap.enemies.get(0);
        return null;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  UNIT ACTION HELPERS
    //  Use AbstractionLayerAI's built-in methods (harvest, attack, train, move)
    //  which translateActions() understands. This is the correct pattern for
    //  AbstractionLayerAI subclasses — avoids raw UnitAction construction.
    // ══════════════════════════════════════════════════════════════════════════

    private void doHarvest(Unit worker, GameSnapshot snap, GameState gs, PhysicalGameState pgs) {
        if (gs.getActionAssignment(worker) != null) return;
        Unit res  = nearestUnit(worker, snap.resources);
        Unit base = snap.myBases.isEmpty() ? null : nearestUnit(worker, snap.myBases);
        if (res != null && base != null) {
            harvest(worker, res, base);
        }
    }

    private void doAttackNearest(Unit unit, List<Unit> targets,
                                   GameState gs, PhysicalGameState pgs) {
        if (gs.getActionAssignment(unit) != null) return;
        if (targets.isEmpty()) return;
        Unit target = nearestUnit(unit, targets);
        if (target != null) attack(unit, target);
    }

    private void doAttack(Unit unit, Unit target, GameState gs, PhysicalGameState pgs) {
        if (gs.getActionAssignment(unit) != null) return;
        if (target == null) return;
        attack(unit, target);
    }

    private void doTrain(Unit building, UnitType type,
                          GameSnapshot snap, PhysicalGameState pgs, GameState gs) {
        if (gs.getActionAssignment(building) != null) return;
        // Find a free adjacent tile facing toward enemies for fast deployment
        int bestDir = UnitAction.DIRECTION_DOWN;
        int bestScore = Integer.MIN_VALUE;
        for (int dir = 0; dir < 4; dir++) {
            int nx = building.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
            int ny = building.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
            if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
            if (pgs.getUnitAt(nx, ny) != null) continue;
            if (pgs.getTerrain(nx, ny) != PhysicalGameState.TERRAIN_NONE) continue;
            int score = 0;
            if (!snap.enemies.isEmpty()) {
                Unit nearest = nearestUnit(nx, ny, snap.enemies);
                if (nearest != null)
                    score = -(Math.abs(nx - nearest.getX()) + Math.abs(ny - nearest.getY()));
            }
            if (score > bestScore) { bestScore = score; bestDir = dir; }
        }
        train(building, type);
    }

    private void doBuild(Unit worker, UnitType buildingType,
                          Pos pos, GameState gs, PhysicalGameState pgs) {
        if (gs.getActionAssignment(worker) != null) return;
        build(worker, buildingType, pos.x, pos.y);
    }

    /**
     * Hold near the base — if more than 3 tiles away, move toward it.
     * This keeps the army grouped for a coordinated push.
     */
    private void doHoldPosition(Unit unit, Unit base, GameState gs, PhysicalGameState pgs) {
        if (gs.getActionAssignment(unit) != null) return;
        int d = dist(unit, base);
        if (d > 3) {
            move(unit, base.getX(), base.getY());
        }
        // else stay idle — translateActions handles TYPE_NONE
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  BUILD POSITION FINDER
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Find a good position to build barracks.
     * Prefers tiles adjacent to our base that are:
     *   - Not a wall or resource
     *   - Not already occupied
     *   - Between our base and the enemy (to serve as a defensive choke)
     */
    private Pos findBarracksPos(GameSnapshot snap, PhysicalGameState pgs) {
        if (snap.myBases.isEmpty()) return null;
        Unit base = snap.myBases.get(0);

        // Try 2-tile radius around base
        for (int radius = 1; radius <= 3; radius++) {
            for (int dx = -radius; dx <= radius; dx++) {
                for (int dy = -radius; dy <= radius; dy++) {
                    if (Math.abs(dx) + Math.abs(dy) != radius) continue; // Manhattan ring
                    int nx = base.getX() + dx;
                    int ny = base.getY() + dy;
                    if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
                    if (pgs.getTerrain(nx, ny) != PhysicalGameState.TERRAIN_NONE) continue;
                    if (pgs.getUnitAt(nx, ny) != null) continue;
                    return new Pos(nx, ny);
                }
            }
        }
        return null;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  GAME SNAPSHOT — pre-computed lists for this tick
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Snapshot of the current game state, built once per tick.
     * Avoids repeated iteration over all units in every helper method.
     */
    private class GameSnapshot {
        final int player, tick;
        final List<Unit> myBases      = new ArrayList<>();
        final List<Unit> myBarracks   = new ArrayList<>();
        final List<Unit> myWorkers    = new ArrayList<>();
        final List<Unit> myArmy       = new ArrayList<>(); // non-worker combat
        final List<Unit> enemyBases   = new ArrayList<>();
        final List<Unit> enemyWorkers = new ArrayList<>();
        final List<Unit> enemyArmy    = new ArrayList<>();
        final List<Unit> enemies      = new ArrayList<>(); // all enemy units
        final List<Unit> resources    = new ArrayList<>();
        int cachedRes;
        int myFutureBarracks = 0; // barracks currently being built

        GameSnapshot(int player, GameState gs, PhysicalGameState pgs) {
            this.player   = player;
            this.tick     = gs.getTime();
            this.cachedRes = gs.getPlayer(player).getResources();

            for (Unit u : pgs.getUnits()) {
                int p = u.getPlayer();
                UnitType t = u.getType();

                if (t.isResource) { resources.add(u); continue; }

                if (p == player) {
                    if (t == baseType)     myBases.add(u);
                    else if (t == barracksType) myBarracks.add(u);
                    else if (t == workerType)   myWorkers.add(u);
                    else if (t.canAttack)        myArmy.add(u);
                } else {
                    enemies.add(u);
                    if (t == baseType)     enemyBases.add(u);
                    else if (t == workerType) enemyWorkers.add(u);
                    else if (t.canAttack)  enemyArmy.add(u);
                }
            }

            // Count barracks currently being produced
            for (Unit u : pgs.getUnits()) {
                if (u.getPlayer() != player) continue;
                UnitActionAssignment uaa = gs.getActionAssignment(u);
                if (uaa != null && uaa.action.getType() == UnitAction.TYPE_PRODUCE
                        && uaa.action.getUnitType() == barracksType) {
                    myFutureBarracks++;
                }
            }
        }

        int resources(GameState gs) { return cachedRes; }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LLM HTTP CALL
    // ══════════════════════════════════════════════════════════════════════════

    private String callLLM(String userPrompt, String systemPrompt) {
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
            sys.addProperty("content", systemPrompt + "\nJSON only. No markdown.");
            msgs.add(sys);
            JsonObject usr = new JsonObject();
            usr.addProperty("role", "user");
            usr.addProperty("content", userPrompt);
            msgs.add(usr);
            req.add("messages", msgs);

            JsonObject fmt = new JsonObject();
            fmt.addProperty("type", "json_object");
            req.add("response_format", fmt);
            req.addProperty("temperature", 0.2);
            req.addProperty("max_tokens", 48); // tiny response needed

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
                    if (choices != null && choices.size() > 0) {
                        return choices.get(0).getAsJsonObject()
                                .getAsJsonObject("message")
                                .get("content").getAsString();
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("[yerule] HTTP error: " + e.getMessage());
        }
        return "{}";
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  UTILITIES
    // ══════════════════════════════════════════════════════════════════════════

    private static class Pos {
        final int x, y;
        Pos(int x, int y) { this.x = x; this.y = y; }
    }

    private int dist(Unit a, Unit b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    private Unit nearestUnit(Unit src, List<Unit> units) {
        return nearestUnit(src.getX(), src.getY(), units);
    }

    private Unit nearestUnit(int x, int y, List<Unit> units) {
        Unit best = null;
        int bestD = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = Math.abs(x - u.getX()) + Math.abs(y - u.getY());
            if (d < bestD) { bestD = d; best = u; }
        }
        return best;
    }

    /** Returns nearest unit within maxDist, or null if none close enough */
    private Unit nearestUnitWithinDist(Unit src, List<Unit> units, int maxDist) {
        Unit best = nearestUnit(src, units);
        if (best == null) return null;
        return dist(src, best) <= maxDist ? best : null;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}