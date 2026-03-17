/*
 * yebot — PureLLM with Chain of Summarization (CoS)
 *
 * Inspired by TextStarCraft II (NeurIPS 2024) Chain of Summarization method.
 * Adapted for MicroRTS's faster pace and simpler action space.
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │                   CHAIN OF SUMMARIZATION                        │
 * │                                                                 │
 * │  Every tick (instant, rule-based):                              │
 * │    Raw game state → L1 Frame Summary                            │
 * │    (structured text snapshot: units, threats, resources)        │
 * │                      ↓                                          │
 * │    Stored in Frame Queue (last K=5 frames)                      │
 * │                                                                 │
 * │  Every LLM_INTERVAL ticks (async, background):                 │
 * │    Frame Queue → L2 Multi-Frame Summary (LLM call)              │
 * │    (trend analysis: "enemy army growing", "economy stalling")   │
 * │                      ↓                                          │
 * │    L2 output → Action Extractor                                 │
 * │    (maps LLM decisions to VALID pre-computed action vocabulary) │
 * │                      ↓                                          │
 * │    Action Queue filled for next K ticks                         │
 * └─────────────────────────────────────────────────────────────────┘
 *
 * Key difference from raw PureLLM:
 *   Raw PureLLM: LLM outputs raw coordinates → often invalid/hallucinated
 *   CoS PureLLM: LLM picks from pre-validated action vocabulary → always valid
 *
 * The action vocabulary is built fresh each tick from the actual game state,
 * so every action the LLM can choose is guaranteed to be executable.
 *
 * This keeps it PureLLM: the LLM still decides WHAT to do for every unit.
 * The rule layer only ensures the HOW is physically possible.
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
import java.util.concurrent.*;
import java.util.stream.*;

public class yebot extends AbstractionLayerAI {

    // ─── API Config ────────────────────────────────────────────────────────────
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "llama4:latest";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int LLM_TIMEOUT  = 20000;

    // ─── CoS Config ───────────────────────────────────────────────────────────
    private static final int FRAME_QUEUE_SIZE = 8;  // K: more history = richer L2 context
    private static final int LLM_INTERVAL     = 200; // ~7 LLM calls per 1500-tick game
    private static final int OPENING_TICKS    = 200; // first 200 ticks = opening rush phase
    // No TTL — action queue is always valid, refreshed when LLM finishes.

    // ─── Unit Types ────────────────────────────────────────────────────────────
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ─── CoS State ────────────────────────────────────────────────────────────
    // Frame Queue: circular buffer of L1 summaries (rule-based, instant)
    private final Deque<String> frameQueue = new ArrayDeque<>();

    // Action Queue: pre-validated actions from last LLM response. Always applied.
    // Key = unit ID, Value = the validated UnitAction to execute
    private volatile Map<Long, UnitAction> actionQueue = new HashMap<>();

    // Async L2 state
    private final ExecutorService llmThread = Executors.newSingleThreadExecutor();
    private volatile boolean llmRunning = false;
    private int lastLLMTick = -LLM_INTERVAL;
    private boolean openingFired = false;

    // ─── L2 System Prompt ─────────────────────────────────────────────────────
    private static final String L2_SYSTEM_PROMPT = """
You are a MicroRTS strategic advisor. You receive a sequence of game state snapshots
and must issue commands for each idle unit.

=== UNIT REFERENCE ===
Worker  HP=1  dmg=1  range=1  — harvests resources, builds barracks (cost=1)
Light   HP=4  dmg=2  range=1  — fast melee fighter (cost=2)
Heavy   HP=8  dmg=4  range=1  — slow tank (cost=3)
Ranged  HP=3  dmg=1  range=3  — attacks from distance (cost=2)
Base    HP=10          — produces workers; PROTECT THIS (cost=10)
Barracks HP=5          — produces military units (cost=5)

=== YOUR ACTION VOCABULARY ===
You MUST only use action IDs from the AVAILABLE ACTIONS list below.
Do NOT invent new action IDs. Every ID in the list is guaranteed valid.

=== OUTPUT FORMAT (JSON only) ===
{
  "analysis": "1-2 sentences: what is the trend across recent frames?",
  "strategy": "1 sentence: what is the priority this turn?",
  "assignments": [
    {"unit_id": "<ID from available actions>", "action_id": "<ACTION_ID>"}
  ]
}

Each unit_id and action_id MUST exactly match an entry in AVAILABLE ACTIONS.
Assign at most one action per unit. Unassigned units will idle.
""";

    // ══════════════════════════════════════════════════════════════════════════
    //  CONSTRUCTORS / RESET
    // ══════════════════════════════════════════════════════════════════════════

    public yebot(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    public yebot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        frameQueue.clear();
        actionQueue    = new HashMap<>();
        llmRunning     = false;
        lastLLMTick    = -LLM_INTERVAL;
        openingFired   = false;
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

    // ══════════════════════════════════════════════════════════════════════════
    //  MAIN ENTRY POINT
    // ══════════════════════════════════════════════════════════════════════════

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int tick = gs.getTime();

        // ── L1: Build frame summary (instant, rule-based) ─────────────────────
        String l1 = buildL1Summary(player, gs, pgs);

        // Add to frame queue — keep last FRAME_QUEUE_SIZE frames
        frameQueue.addLast(l1);
        while (frameQueue.size() > FRAME_QUEUE_SIZE) frameQueue.pollFirst();

        // ── Build action vocabulary only when about to call LLM ─────────────
        // Vocab building runs A* per unit — only do this work when needed.
        // Between LLM calls, just apply the cached actionQueue.
        // -- OPENING RUSH: fire at tick 0, no LLM delay --
        if (!openingFired && !llmRunning) {
            openingFired = true;
            llmRunning   = true;
            ActionVocabulary ov = buildActionVocabulary(player, gs, pgs);
            final ActionVocabulary ovc = ov;
            final int t0 = tick;
            llmThread.submit(() -> {
                try {
                    Map<Long, UnitAction> result = callOpeningRushLLM(ovc);
                    if (!result.isEmpty()) {
                        actionQueue = result;
                        System.out.println("[yebot] Opening set t=" + t0
                                + " (" + result.size() + " units)");
                    }
                } catch (Exception e) {
                    System.err.println("[yebot] Opening err: " + e.getMessage());
                } finally {
                    llmRunning = false;
                }
            });
        }

        // -- L2 strategic call after opening phase --
        boolean needVocab = !llmRunning && tick - lastLLMTick >= LLM_INTERVAL
                && tick >= OPENING_TICKS;
        ActionVocabulary vocab = needVocab ? buildActionVocabulary(player, gs, pgs) : null;

        // ── Fire async L2 if interval elapsed and vocab is non-empty ──────────
        if (needVocab && vocab != null && !vocab.isEmpty()) {
            lastLLMTick = tick;
            llmRunning  = true;
            final String frameHistory = buildFrameHistory();
            final ActionVocabulary vocabCopy = vocab;
            final int tickCopy = tick;
            final int playerCopy = player;
            llmThread.submit(() -> {
                try {
                    Map<Long, UnitAction> result = callL2LLM(
                            frameHistory, vocabCopy, playerCopy);
                    if (!result.isEmpty()) {
                        actionQueue = result;
                        System.out.println("[yebot] Queue updated at t=" + tickCopy
                                + " (" + result.size() + " assignments)");
                    }
                } catch (Exception e) {
                    System.err.println("[yebot] L2 error: " + e.getMessage());
                } finally {
                    llmRunning = false;
                }
            });
        }

        // ── Apply action queue to currently idle units ─────────────────────────
        // Always apply last known decisions — stale > nothing.
        // Dead/missing units are safely ignored (not in pgs.getUnits()).
        PlayerAction pa = new PlayerAction();
        Map<Long, UnitAction> queue = actionQueue;
        if (!queue.isEmpty()) {
            for (Unit u : pgs.getUnits()) {
                if (u.getPlayer() != player) continue;
                if (gs.getActionAssignment(u) != null) continue;
                UnitAction ua = queue.get(u.getID());
                if (ua != null && ua.getType() != UnitAction.TYPE_NONE
                        && gs.isUnitActionAllowed(u, ua)) {
                    pa.addUnitAction(u, ua);
                }
            }
        }

        pa.fillWithNones(gs, player, 1);
        return pa;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  L1 SUMMARY — rule-based, instant, every tick
    //
    //  Equivalent to SC2's L1 (single-frame) summarization.
    //  Compresses raw game state into structured text.
    //  No LLM involved — pure Java computation.
    // ══════════════════════════════════════════════════════════════════════════

    private String buildL1Summary(int player, GameState gs, PhysicalGameState pgs) {
        int enemy = 1 - player;
        int tick  = gs.getTime();
        int myRes = gs.getPlayer(player).getResources();
        int enRes = gs.getPlayer(enemy).getResources();

        int myWorkers = 0, myMilitary = 0, myBarracks = 0, myBaseHP = 0;
        int enWorkers = 0, enMilitary = 0, enBaseHP = 0;
        int resources = 0;
        int myIdleWorkers = 0, myIdleMilitary = 0;

        for (Unit u : pgs.getUnits()) {
            UnitType t = u.getType();
            if (t.isResource) { resources++; continue; }
            boolean idle = gs.getActionAssignment(u) == null;

            if (u.getPlayer() == player) {
                if (t == workerType)     { myWorkers++;  if (idle) myIdleWorkers++; }
                else if (t == barracksType) myBarracks++;
                else if (t == baseType)  myBaseHP = u.getHitPoints();
                else if (t.canAttack)    { myMilitary++; if (idle) myIdleMilitary++; }
            } else {
                if (t == workerType)     enWorkers++;
                else if (t == baseType)  enBaseHP = u.getHitPoints();
                else if (t.canAttack)    enMilitary++;
            }
        }

        // Threat level: how urgent is the situation?
        String threat;
        if (enMilitary >= myMilitary + 2) threat = "HIGH";
        else if (enMilitary > myMilitary) threat = "MEDIUM";
        else threat = "LOW";

        // Economy status
        String economy;
        if (myRes >= 5) economy = "SURPLUS";
        else if (myRes >= 2) economy = "ADEQUATE";
        else economy = "SCARCE";

        return String.format(
            "[T=%d] res=%d(%s) myBase=%dHP myArmy=%d(idle=%d) myWorkers=%d(idle=%d) myBarracks=%d | " +
            "enBase=%dHP enArmy=%d enWorkers=%d enRes=%d | mapRes=%d | threat=%s",
            tick, myRes, economy,
            myBaseHP, myMilitary, myIdleMilitary,
            myWorkers, myIdleWorkers, myBarracks,
            enBaseHP, enMilitary, enWorkers, enRes,
            resources, threat
        );
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  ACTION VOCABULARY — pre-validated legal moves, built every tick
    //
    //  This is the core of CoS for MicroRTS.
    //  Every entry is a (unitID, actionID, UnitAction) triple.
    //  The LLM only picks action IDs from this list — impossible to hallucinate
    //  an invalid coordinate or an illegal action.
    //
    //  Action IDs are human-readable strings like:
    //    "W1_harvest_res2_1"  (worker 1 harvests resource at (2,1))
    //    "B2_train_ranged"    (barracks 2 trains ranged unit)
    //    "R3_attack_E5_6"     (ranged unit 3 attacks enemy at (5,6))
    //    "L4_move_toward_E"   (light unit 4 moves toward nearest enemy)
    // ══════════════════════════════════════════════════════════════════════════

    private static class ActionEntry {
        final String unitId;    // e.g. "W1" (worker 1), "B0" (base 0)
        final String actionId;  // human-readable ID for LLM
        final long   unitUID;   // actual unit ID for game engine
        final UnitAction action; // the actual UnitAction to execute

        ActionEntry(String unitId, String actionId, long unitUID, UnitAction action) {
            this.unitId   = unitId;
            this.actionId = actionId;
            this.unitUID  = unitUID;
            this.action   = action;
        }
    }

    private static class ActionVocabulary {
        final List<ActionEntry> entries = new ArrayList<>();
        // Map actionId → ActionEntry for fast lookup during response parsing
        final Map<String, ActionEntry> byActionId = new HashMap<>();
        // Map unitId → list of possible actions
        final Map<String, List<ActionEntry>> byUnitId = new LinkedHashMap<>();

        void add(ActionEntry e) {
            entries.add(e);
            byActionId.put(e.actionId, e);
            byUnitId.computeIfAbsent(e.unitId, k -> new ArrayList<>()).add(e);
        }

        boolean isEmpty() { return entries.isEmpty(); }

        /** Format vocabulary for LLM prompt */
        String format() {
            StringBuilder sb = new StringBuilder();
            for (Map.Entry<String, List<ActionEntry>> unitEntry : byUnitId.entrySet()) {
                String unitId = unitEntry.getKey();
                sb.append("Unit ").append(unitId).append(":\n");
                for (ActionEntry ae : unitEntry.getValue()) {
                    sb.append("  ").append(ae.actionId).append("\n");
                }
            }
            return sb.toString();
        }
    }

    private ActionVocabulary buildActionVocabulary(int player, GameState gs,
                                                     PhysicalGameState pgs) {
        ActionVocabulary vocab = new ActionVocabulary();

        // Counters for readable unit IDs
        int wCount = 0, lCount = 0, hCount = 0, rCount = 0, bCount = 0, rkCount = 0;

        // Collect enemies and resources for targeting
        List<Unit> enemies   = new ArrayList<>();
        List<Unit> resources = new ArrayList<>();
        Unit myBase = null;

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) { resources.add(u); continue; }
            if (u.getPlayer() == player && u.getType() == baseType) myBase = u;
            if (u.getPlayer() != player && u.getPlayer() != -1) enemies.add(u);
        }

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) continue;
            if (gs.getActionAssignment(u) != null) continue; // busy — skip

            UnitType t = u.getType();
            int myResources = gs.getPlayer(player).getResources();

            if (t == workerType) {
                String uid = "W" + wCount++;
                addWorkerActions(vocab, u, uid, myBase, enemies, resources, gs, pgs, player);

            } else if (t == lightType) {
                String uid = "L" + lCount++;
                addMilitaryActions(vocab, u, uid, enemies, gs, pgs);

            } else if (t == heavyType) {
                String uid = "H" + hCount++;
                addMilitaryActions(vocab, u, uid, enemies, gs, pgs);

            } else if (t == rangedType) {
                String uid = "R" + rCount++;
                addMilitaryActions(vocab, u, uid, enemies, gs, pgs);

            } else if (t == baseType) {
                String uid = "BASE" + bCount++;
                addBaseActions(vocab, u, uid, myResources, gs, pgs);

            } else if (t == barracksType) {
                String uid = "BRX" + rkCount++;
                addBarracksActions(vocab, u, uid, myResources, gs, pgs, enemies);
            }
        }

        return vocab;
    }

    private void addWorkerActions(ActionVocabulary vocab, Unit worker, String uid,
                                   Unit myBase, List<Unit> enemies, List<Unit> resources,
                                   GameState gs, PhysicalGameState pgs, int player) {
        long wid = worker.getID();

        // Harvest actions — one per nearby resource
        if (myBase != null && !resources.isEmpty()) {
            // If carrying resources → return to base
            if (worker.getResources() > 0) {
                UnitAction ua = pf.findPathToAdjacentPosition(worker,
                        myBase.getX() + myBase.getY() * pgs.getWidth(), gs, null);
                if (ua != null) {
                    boolean adj = Math.abs(worker.getX()-myBase.getX())
                                + Math.abs(worker.getY()-myBase.getY()) == 1;
                    UnitAction actual = adj
                            ? new UnitAction(UnitAction.TYPE_RETURN, ua.getDirection())
                            : new UnitAction(UnitAction.TYPE_MOVE, ua.getDirection());
                    if (gs.isUnitActionAllowed(worker, actual))
                        vocab.add(new ActionEntry(uid,
                                uid + "_return_resources_to_base", wid, actual));
                }
            } else {
                // Harvest nearest 2 resources (give LLM choices)
                resources.stream()
                        .sorted(Comparator.comparingInt(r ->
                                Math.abs(worker.getX()-r.getX()) + Math.abs(worker.getY()-r.getY())))
                        .limit(2)
                        .forEach(res -> {
                            UnitAction ua = pf.findPathToAdjacentPosition(worker,
                                    res.getX() + res.getY() * pgs.getWidth(), gs, null);
                            if (ua != null) {
                                boolean adj = Math.abs(worker.getX()-res.getX())
                                            + Math.abs(worker.getY()-res.getY()) == 1;
                                UnitAction actual = adj
                                        ? new UnitAction(UnitAction.TYPE_HARVEST, ua.getDirection())
                                        : new UnitAction(UnitAction.TYPE_MOVE, ua.getDirection());
                                if (gs.isUnitActionAllowed(worker, actual))
                                    vocab.add(new ActionEntry(uid,
                                            uid + "_harvest_at_" + res.getX() + "_" + res.getY(),
                                            wid, actual));
                            }
                        });
            }
        }

        // Build barracks — ONE A* call only, to the single best adjacent tile
        // Never scan a grid of positions — that was causing 25+ A* calls per worker
        int myRes = gs.getPlayer(player).getResources();
        if (myRes >= barracksType.cost) {
            // Find one free adjacent tile — check 4 directions, no A* needed
            for (int dir = 0; dir < 4; dir++) {
                int bx = worker.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
                int by = worker.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
                if (bx < 0 || by < 0 || bx >= pgs.getWidth() || by >= pgs.getHeight()) continue;
                if (pgs.getTerrain(bx, by) != PhysicalGameState.TERRAIN_NONE) continue;
                if (pgs.getUnitAt(bx, by) != null) continue;
                // Worker is already adjacent — no A* needed at all
                UnitAction actual = new UnitAction(UnitAction.TYPE_PRODUCE, dir, barracksType);
                if (gs.isUnitActionAllowed(worker, actual)) {
                    vocab.add(new ActionEntry(uid,
                            uid + "_build_barracks_at_" + bx + "_" + by, wid, actual));
                    break; // one build option is enough
                }
            }
            // If not adjacent to any free tile, offer a move-toward-base action
            // (worker needs to reposition first — one A* call)
            if (!vocab.byUnitId.getOrDefault(uid, new ArrayList<>()).stream()
                    .anyMatch(e -> e.actionId.contains("build"))) {
                if (myBase != null) {
                    UnitAction ua = pf.findPathToAdjacentPosition(worker,
                            myBase.getX() + myBase.getY() * pgs.getWidth(), gs, null);
                    if (ua != null) {
                        UnitAction move = new UnitAction(UnitAction.TYPE_MOVE, ua.getDirection());
                        if (gs.isUnitActionAllowed(worker, move))
                            vocab.add(new ActionEntry(uid, uid + "_move_to_build_position",
                                    wid, move));
                    }
                }
            }
        }

        // Attack nearest enemy (workers can fight)
        if (!enemies.isEmpty()) {
            Unit target = nearestUnit(worker, enemies);
            if (target != null) {
                int d = dist(worker, target);
                UnitAction ua;
                if (d <= worker.getType().attackRange) {
                    ua = new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                            target.getX() + target.getY() * pgs.getWidth());
                } else {
                    UnitAction move = pf.findPathToAdjacentPosition(worker,
                            target.getX() + target.getY() * pgs.getWidth(), gs, null);
                    ua = move != null ? new UnitAction(UnitAction.TYPE_MOVE, move.getDirection()) : null;
                }
                if (ua != null && gs.isUnitActionAllowed(worker, ua))
                    vocab.add(new ActionEntry(uid,
                            uid + "_attack_nearest_enemy_at_" + target.getX() + "_" + target.getY(),
                            wid, ua));
            }
        }
    }

    private void addMilitaryActions(ActionVocabulary vocab, Unit unit, String uid,
                                     List<Unit> enemies, GameState gs, PhysicalGameState pgs) {
        if (enemies.isEmpty()) return;
        long id = unit.getID();

        // Find nearest enemy (no A* — just manhattan distance comparison)
        // Offer 3 distinct target types to give LLM meaningful choices:
        //   1. Nearest enemy (always)
        //   2. Enemy base (if exists and not already nearest)
        //   3. Enemy with lowest HP (focus-fire option)
        Set<Long> addedTargets = new HashSet<>();

        // 1. Nearest enemy — ONE A* call
        Unit nearest = nearestUnit(unit, enemies);
        if (nearest != null) {
            UnitAction ua = buildAttackOrMove(unit, nearest, gs, pgs);
            if (ua != null && gs.isUnitActionAllowed(unit, ua)) {
                String label = targetLabel(nearest);
                vocab.add(new ActionEntry(uid, uid + "_" + label
                        + "_at_" + nearest.getX() + "_" + nearest.getY(), id, ua));
                addedTargets.add(nearest.getID());
            }
        }

        // 2. Enemy base — ONE A* call (different target, more strategic)
        enemies.stream()
                .filter(e -> e.getType() == baseType && !addedTargets.contains(e.getID()))
                .findFirst().ifPresent(base -> {
                    UnitAction ua = buildAttackOrMove(unit, base, gs, pgs);
                    if (ua != null && gs.isUnitActionAllowed(unit, ua)) {
                        vocab.add(new ActionEntry(uid, uid + "_attack_base"
                                + "_at_" + base.getX() + "_" + base.getY(), id, ua));
                        addedTargets.add(base.getID());
                    }
                });

        // 3. Lowest HP enemy — ONE A* call (focus-fire option)
        enemies.stream()
                .filter(e -> !addedTargets.contains(e.getID()))
                .min(Comparator.comparingInt(Unit::getHitPoints))
                .ifPresent(weak -> {
                    UnitAction ua = buildAttackOrMove(unit, weak, gs, pgs);
                    if (ua != null && gs.isUnitActionAllowed(unit, ua)) {
                        vocab.add(new ActionEntry(uid, uid + "_focusfire_weakest"
                                + "_at_" + weak.getX() + "_" + weak.getY(), id, ua));
                    }
                });
    }

    private UnitAction buildAttackOrMove(Unit unit, Unit target,
                                          GameState gs, PhysicalGameState pgs) {
        int d = dist(unit, target);
        if (d <= unit.getType().attackRange) {
            return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                    target.getX() + target.getY() * pgs.getWidth());
        }
        UnitAction move = pf.findPathToAdjacentPosition(unit,
                target.getX() + target.getY() * pgs.getWidth(), gs, null);
        return move != null ? new UnitAction(UnitAction.TYPE_MOVE, move.getDirection()) : null;
    }

    private String targetLabel(Unit target) {
        if (target.getType() == baseType)     return "attack_base";
        if (target.getType().canHarvest)      return "attack_worker";
        return "attack_" + target.getType().name.toLowerCase();
    }

    private void addBaseActions(ActionVocabulary vocab, Unit base, String uid,
                                 int myRes, GameState gs, PhysicalGameState pgs) {
        long id = base.getID();
        if (myRes < workerType.cost) return;

        for (int dir = 0; dir < 4; dir++) {
            int nx = base.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
            int ny = base.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
            if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
            if (pgs.getUnitAt(nx, ny) != null) continue;
            UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE, dir, workerType);
            if (gs.isUnitActionAllowed(base, ua)) {
                vocab.add(new ActionEntry(uid, uid + "_train_worker", id, ua));
                break; // one train action per base
            }
        }
    }

    private void addBarracksActions(ActionVocabulary vocab, Unit barrack, String uid,
                                     int myRes, GameState gs, PhysicalGameState pgs,
                                     List<Unit> enemies) {
        long id = barrack.getID();

        // Find best spawn direction (toward nearest enemy)
        int bestDir = findBestSpawnDir(barrack, pgs, enemies);

        // Offer affordable units
        if (myRes >= rangedType.cost) {
            UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE, bestDir, rangedType);
            if (gs.isUnitActionAllowed(barrack, ua))
                vocab.add(new ActionEntry(uid, uid + "_train_ranged", id, ua));
        }
        if (myRes >= lightType.cost) {
            UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE, bestDir, lightType);
            if (gs.isUnitActionAllowed(barrack, ua))
                vocab.add(new ActionEntry(uid, uid + "_train_light", id, ua));
        }
        if (myRes >= heavyType.cost) {
            UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE, bestDir, heavyType);
            if (gs.isUnitActionAllowed(barrack, ua))
                vocab.add(new ActionEntry(uid, uid + "_train_heavy", id, ua));
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  L2 LLM CALL — multi-frame synthesis + action extraction
    //
    //  Equivalent to SC2's L2 (multi-frame) summarization + action extractor.
    //  Receives K L1 summaries, outputs assignments from vocabulary.
    // ══════════════════════════════════════════════════════════════════════════

    private Map<Long, UnitAction> callL2LLM(String frameHistory,
                                              ActionVocabulary vocab,
                                              int player) {
        // Build L2 prompt: frame history + available action vocabulary
        String userPrompt = buildL2Prompt(frameHistory, vocab);
        String response   = callLLMRaw(userPrompt);

        // Action extractor: parse LLM assignments, map to UnitActions via vocab
        return extractActions(response, vocab);
    }

    private Map<Long, UnitAction> callOpeningRushLLM(ActionVocabulary vocab) {
        return extractActions(callLLMRaw(buildOpeningRushPrompt(vocab)), vocab);
    }

    private String buildOpeningRushPrompt(ActionVocabulary vocab) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are executing a WORKER RUSH opening in MicroRTS (tick 0-200).\n");
        sb.append("Assign actions to ALL units using ONLY the exact action IDs below.\n\n");
        sb.append("=== RUSH ASSIGNMENT RULES ===\n");
        sb.append("BASE units  -> assign train_worker (keep producing workers)\n");
        sb.append("WORKER-0    -> assign its harvest action (ONE harvester only)\n");
        sb.append("WORKER-1+   -> assign attack action to rush enemy (NOT harvest)\n");
        sb.append("MILITARY    -> assign attack action toward nearest enemy\n");
        sb.append("BARRACKS    -> assign train_ranged if available\n\n");
        sb.append("=== STRATEGY EXPLANATION ===\n");
        sb.append("Workers deal 1 damage each. More attacking workers = more DPS.\n");
        sb.append("One harvester keeps the base training new workers every tick.\n");
        sb.append("New workers should attack immediately, not harvest.\n\n");
        sb.append("=== AVAILABLE ACTIONS ===\n");

        int workerCount = 0;
        for (Map.Entry<String, List<ActionEntry>> e : vocab.byUnitId.entrySet()) {
            String uid = e.getKey();
            sb.append("Unit ").append(uid).append(":\n");
            boolean isWorker = uid.startsWith("W");
            boolean isFirstWorker = uid.equals("W0");
            for (ActionEntry ae : e.getValue()) {
                String hint = "";
                if (ae.actionId.contains("train_worker"))      hint = "  <-- ASSIGN THIS for base";
                else if (ae.actionId.contains("harvest") && isFirstWorker)
                                                               hint = "  <-- ASSIGN THIS for W0 only";
                else if (ae.actionId.contains("attack") && isWorker && !isFirstWorker)
                                                               hint = "  <-- ASSIGN THIS to rush enemy";
                else if (ae.actionId.contains("train_ranged")) hint = "  <-- assign for barracks";
                else if (ae.actionId.contains("return"))       hint = "  <-- only if carrying";
                sb.append("  ").append(ae.actionId).append(hint).append("\n");
            }
        }

        sb.append("\n=== OUTPUT FORMAT (JSON only) ===\n");
        sb.append("{\n");
        sb.append("  \"analysis\": \"Worker rush opening: one harvests, rest attack\",\n");
        sb.append("  \"strategy\": \"Flood enemy base with workers now\",\n");
        sb.append("  \"assignments\": [\n");
        sb.append("    {\"unit_id\": \"BASE0\", \"action_id\": \"BASE0_train_worker\"},\n");
        sb.append("    {\"unit_id\": \"W0\", \"action_id\": \"W0_harvest_at_X_Y\"},\n");
        sb.append("    {\"unit_id\": \"W1\", \"action_id\": \"W1_attack_nearest_enemy_at_X_Y\"}\n");
        sb.append("  ]\n}\n");
        sb.append("Replace X_Y with actual coordinates from IDs above. Copy IDs exactly.\n");
        return sb.toString();
    }

    private String buildL2Prompt(String frameHistory, ActionVocabulary vocab) {
        return L2_SYSTEM_PROMPT
            + "\n=== RECENT GAME FRAMES (newest last) ===\n"
            + frameHistory
            + "\n\n=== AVAILABLE ACTIONS ===\n"
            + vocab.format()
            + "\nAssign actions to units. Use exact action IDs from the list above.";
    }

    private String buildFrameHistory() {
        StringBuilder sb = new StringBuilder();
        int i = 1;
        for (String frame : frameQueue) {
            sb.append("Frame ").append(i++).append(": ").append(frame).append("\n");
        }
        return sb.toString();
    }

    // ── Action Extractor ───────────────────────────────────────────────────────

    /**
     * Parse LLM response and map action IDs back to actual UnitActions.
     * This is the "action extractor" component from SC2 CoS.
     * Any action ID not in the vocabulary is silently dropped — no hallucinations.
     */
    private Map<Long, UnitAction> extractActions(String response, ActionVocabulary vocab) {
        Map<Long, UnitAction> result = new HashMap<>();

        try {
            response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
            int s = response.indexOf("{"), e = response.lastIndexOf("}") + 1;
            if (s >= 0 && e > s) response = response.substring(s, e);

            JsonObject json = JsonParser.parseString(response).getAsJsonObject();

            if (json.has("analysis"))
                System.out.println("[yebot] Analysis: " + json.get("analysis").getAsString());
            if (json.has("strategy"))
                System.out.println("[yebot] Strategy: " + json.get("strategy").getAsString());

            JsonArray assignments = json.getAsJsonArray("assignments");
            if (assignments == null) return result;

            Set<String> usedUnits = new HashSet<>();

            for (JsonElement el : assignments) {
                if (!el.isJsonObject()) continue;
                JsonObject assignment = el.getAsJsonObject();

                String unitId   = assignment.has("unit_id")   ? assignment.get("unit_id").getAsString()   : null;
                String actionId = assignment.has("action_id") ? assignment.get("action_id").getAsString() : null;

                if (unitId == null || actionId == null) continue;
                if (usedUnits.contains(unitId)) continue; // one action per unit

                // Look up in vocabulary — if not found, silently skip (no hallucinations)
                ActionEntry entry = vocab.byActionId.get(actionId);
                if (entry == null) {
                    System.out.println("[yebot] Unknown action ID: " + actionId + " (skipped)");
                    continue;
                }

                result.put(entry.unitUID, entry.action);
                usedUnits.add(unitId);
            }

        } catch (Exception ex) {
            System.err.println("[yebot] Extraction error: " + ex.getMessage());
        }

        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LLM HTTP CALL
    // ══════════════════════════════════════════════════════════════════════════

    private String callLLMRaw(String fullPrompt) {
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
            JsonObject usr = new JsonObject();
            usr.addProperty("role", "user");
            usr.addProperty("content", fullPrompt);
            msgs.add(usr);
            req.add("messages", msgs);

            JsonObject fmt = new JsonObject();
            fmt.addProperty("type", "json_object");
            req.add("response_format", fmt);
            req.addProperty("temperature", 0.2);
            req.addProperty("max_tokens", 512);

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
            } else {
                System.err.println("[yebot] API error " + conn.getResponseCode());
            }
        } catch (Exception e) {
            System.err.println("[yebot] HTTP error: " + e.getMessage());
        }
        return "{\"analysis\":\"\",\"strategy\":\"\",\"assignments\":[]}";
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  UTILITIES
    // ══════════════════════════════════════════════════════════════════════════

    private int findBestSpawnDir(Unit building, PhysicalGameState pgs, List<Unit> enemies) {
        int bestDir = UnitAction.DIRECTION_DOWN;
        int bestScore = Integer.MIN_VALUE;
        for (int dir = 0; dir < 4; dir++) {
            int nx = building.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
            int ny = building.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
            if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
            if (pgs.getUnitAt(nx, ny) != null) continue;
            if (pgs.getTerrain(nx, ny) != PhysicalGameState.TERRAIN_NONE) continue;
            int score = 0;
            if (!enemies.isEmpty()) {
                Unit nearest = nearestUnit(nx, ny, enemies);
                if (nearest != null)
                    score = -(Math.abs(nx - nearest.getX()) + Math.abs(ny - nearest.getY()));
            }
            if (score > bestScore) { bestScore = score; bestDir = dir; }
        }
        return bestDir;
    }

    private int dist(Unit a, Unit b) {
        return Math.abs(a.getX()-b.getX()) + Math.abs(a.getY()-b.getY());
    }

    private Unit nearestUnit(Unit src, List<Unit> units) {
        return nearestUnit(src.getX(), src.getY(), units);
    }

    private Unit nearestUnit(int x, int y, List<Unit> units) {
        Unit best = null; int bestD = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = Math.abs(x-u.getX()) + Math.abs(y-u.getY());
            if (d < bestD) { bestD = d; best = u; }
        }
        return best;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}