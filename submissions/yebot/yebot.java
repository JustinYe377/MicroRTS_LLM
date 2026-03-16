/*
 * LLM-Guided MCTS Agent for MicroRTS — v3 (Speed + Defense)
 *
 * Speed improvements over v2:
 *   1. Async MCTS — game thread never blocks on LLM
 *   2. Heuristic policy replaces policy LLM — zero LLM calls during expansion
 *   3. LLM used ONLY for final eval pass (1 call per MCTS search)
 *   4. Prompt cache — identical states reuse cached LLM responses
 *   5. Compact eval prompt — ~100 chars instead of ~800
 *   6. Tuned constants — fewer iterations, shallower rollouts
 *
 * Worker Rush Defense:
 *   - Detects incoming worker rushes (many enemy workers approaching base early)
 *   - Activates DEFEND mode: workers intercept attackers, base trains emergency workers
 *   - Returns to normal MCTS play once threat is neutralized
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
import java.util.regex.*;

public class yebot extends AbstractionLayerAI {

    // ─── API Config ────────────────────────────────────────────────────────────
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "llama4:latest";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int REQUEST_TIMEOUT = 15000; // reduced from 30s

    // ─── MCTS Config ───────────────────────────────────────────────────────────
    private static final int MCTS_ITERATIONS   = 8;    // was 10 — heuristic policy makes each iteration free
    private static final int SIMULATION_DEPTH  = 8;    // was 15 — faster rollouts
    private static final double UCB_C          = 1.41;
    private static final int ACTION_INTERVAL   = 20;   // was 10 — search less often, async so it doesn't matter

    // ─── Worker Rush Defense Config ────────────────────────────────────────────
    private static final int RUSH_DETECT_RADIUS   = 6;  // enemy workers within this distance = rush
    private static final int RUSH_WORKER_THRESHOLD = 2;  // at least this many rushing workers = rush
    private static final int RUSH_EARLY_TICKS      = 150; // only detect rush in early game

    // ─── Unit Types ────────────────────────────────────────────────────────────
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ─── Async MCTS State ──────────────────────────────────────────────────────
    private final ExecutorService llmExecutor = Executors.newSingleThreadExecutor();
    private volatile PlayerAction pendingAction = null;
    private volatile boolean llmRunning = false;
    private int lastSubmitTick = -100;

    // ─── LLM Response Cache ────────────────────────────────────────────────────
    private final Map<String, String> llmCache = Collections.synchronizedMap(
        new LinkedHashMap<String, String>() {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, String> e) {
                return size() > 30;
            }
        }
    );

    // ─── Unit ID map for filtering ─────────────────────────────────────────────
    private volatile Map<Long, UnitAction> lastCandidateUnitMap = new HashMap<>();

    // ─── Eval-only prompt (compact) ────────────────────────────────────────────
    private static final String EVAL_SYSTEM_PROMPT = """
You are a MicroRTS position evaluator. Score the ALLY player's position 0-100.
100=ally wins, 0=ally loses, 50=even.
Weigh: unit count, total HP, resources, base health, production buildings.
OUTPUT JSON ONLY: {"score": 65, "reason": "one sentence"}
""";

    // ══════════════════════════════════════════════════════════════════════════
    //  MCTS NODE
    // ══════════════════════════════════════════════════════════════════════════

    private class MCTSNode {
        MCTSNode parent;
        PlayerAction actionTaken;
        GameState state;
        int player;
        List<MCTSNode> children = new ArrayList<>();
        double totalValue = 0.0;
        int visitCount = 0;
        boolean expanded = false;

        MCTSNode(MCTSNode parent, PlayerAction action, GameState state, int player) {
            this.parent = parent;
            this.actionTaken = action;
            this.state = state;
            this.player = player;
        }

        double ucb1() {
            if (visitCount == 0) return Double.MAX_VALUE;
            double exploit = totalValue / visitCount;
            double explore = UCB_C * Math.sqrt(Math.log(parent.visitCount) / visitCount);
            return exploit + explore;
        }

        double avgValue() {
            return visitCount == 0 ? 0 : totalValue / visitCount;
        }
    }

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
        pendingAction = null;
        llmRunning = false;
        lastSubmitTick = -100;
        llmCache.clear();
        lastCandidateUnitMap = new HashMap<>();
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType   = utt.getUnitType("Worker");
        lightType    = utt.getUnitType("Light");
        heavyType    = utt.getUnitType("Heavy");
        rangedType   = utt.getUnitType("Ranged");
        baseType     = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
    }

    @Override
    public AI clone() {
        return new yebot(utt, pf);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  MAIN ENTRY POINT — never blocks
    // ══════════════════════════════════════════════════════════════════════════

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        int tick = gs.getTime();
        PhysicalGameState pgs = gs.getPhysicalGameState();

        // ── PRIORITY 1: Worker Rush Defense ───────────────────────────────────
        // Runs every tick, purely heuristic, instant — no LLM involved
        if (isWorkerRush(player, gs, pgs)) {
            System.out.println("[yebot] RUSH DETECTED at tick " + tick + " — activating defense");
            PlayerAction defense = buildWorkerRushDefense(player, gs, pgs);
            defense.fillWithNones(gs, player, 1);
            return defense;
        }

        // ── PRIORITY 2: Fire async MCTS if not already running ────────────────
        if (!llmRunning && tick - lastSubmitTick >= ACTION_INTERVAL) {
            lastSubmitTick = tick;
            llmRunning = true;
            final GameState gsCopy = gs.clone();
            final int playerCopy = player;
            llmExecutor.submit(() -> {
                try {
                    PlayerAction result = runMCTS(playerCopy, gsCopy);
                    // Store unit map for filtering in game thread
                    pendingAction = result;
                } catch (Exception e) {
                    System.err.println("[MCTS] Async search failed: " + e.getMessage());
                } finally {
                    llmRunning = false;
                }
            });
        }

        // ── PRIORITY 3: Apply last MCTS result if available ───────────────────
        if (pendingAction != null) {
            PlayerAction result = filterValidAction(player, gs);
            result.fillWithNones(gs, player, 1);
            return result;
        }

        // ── PRIORITY 4: Heuristic fallback on first ticks before MCTS finishes ─
        PlayerAction fallback = buildHeuristicAction(player, gs, pgs);
        fallback.fillWithNones(gs, player, 1);
        return fallback;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  WORKER RUSH DETECTION + DEFENSE
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Detect a worker rush: enemy has RUSH_WORKER_THRESHOLD+ workers
     * within RUSH_DETECT_RADIUS of our base in the early game.
     */
    private boolean isWorkerRush(int player, GameState gs, PhysicalGameState pgs) {
        if (gs.getTime() > RUSH_EARLY_TICKS) return false;

        // Find our base
        Unit myBase = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType() == baseType) {
                myBase = u;
                break;
            }
        }
        if (myBase == null) return false;

        // Count enemy workers approaching our base
        int rushingWorkers = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player || u.getPlayer() == -1) continue;
            if (u.getType() != workerType) continue;
            int dist = Math.abs(u.getX() - myBase.getX()) + Math.abs(u.getY() - myBase.getY());
            if (dist <= RUSH_DETECT_RADIUS) rushingWorkers++;
        }

        return rushingWorkers >= RUSH_WORKER_THRESHOLD;
    }

    /**
     * Worker Rush Defense strategy:
     *
     * The key insight about worker rushes:
     *   - Enemy workers have 1 HP and 1 damage — same as yours
     *   - It's a pure numbers game — more workers attacking = win
     *   - Defense: intercept with ALL idle workers, train more immediately
     *   - Workers should attack the WEAKEST enemy first (kill confirmation)
     *   - Base should spam workers if affordable (1 resource cost)
     *   - Do NOT waste time harvesting during a rush
     */
    private PlayerAction buildWorkerRushDefense(int player, GameState gs, PhysicalGameState pgs) {
        PlayerAction pa = new PlayerAction();
        Set<Long> assigned = new HashSet<>();

        Unit myBase = null;
        List<Unit> myWorkers = new ArrayList<>();
        List<Unit> enemyWorkers = new ArrayList<>();
        List<Unit> enemyCombat = new ArrayList<>(); // non-worker enemies

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == baseType) myBase = u;
                if (u.getType() == workerType) myWorkers.add(u);
            } else if (u.getPlayer() != -1) {
                if (u.getType() == workerType) enemyWorkers.add(u);
                else if (u.getType().canAttack) enemyCombat.add(u);
            }
        }

        // Sort enemy workers by HP ascending — attack weakest first for kill confirmation
        enemyWorkers.sort(Comparator.comparingInt(Unit::getHitPoints));

        // All enemies to fight (workers first, then other combat units)
        List<Unit> allEnemyThreats = new ArrayList<>(enemyWorkers);
        allEnemyThreats.addAll(enemyCombat);

        // ── Step 1: All idle workers intercept enemy workers ──────────────────
        for (Unit worker : myWorkers) {
            if (gs.getActionAssignment(worker) != null) continue; // busy
            if (allEnemyThreats.isEmpty()) break;

            // Find closest enemy threat
            Unit target = null;
            int bestDist = Integer.MAX_VALUE;
            for (Unit enemy : allEnemyThreats) {
                int dist = Math.abs(worker.getX() - enemy.getX())
                         + Math.abs(worker.getY() - enemy.getY());
                if (dist < bestDist) {
                    bestDist = dist;
                    target = enemy;
                }
            }
            if (target == null) continue;

            // If adjacent/in range — attack immediately
            if (bestDist <= worker.getType().attackRange) {
                UnitAction attack = new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                        target.getX() + target.getY() * pgs.getWidth());
                if (gs.isUnitActionAllowed(worker, attack)) {
                    pa.addUnitAction(worker, attack);
                    assigned.add(worker.getID());
                    // Remove target if we expect to kill it (1 damage vs 1 HP)
                    if (target.getHitPoints() <= worker.getType().maxDamage) {
                        allEnemyThreats.remove(target);
                    }
                    continue;
                }
            }

            // Otherwise — move toward target
            UnitAction move = pf.findPathToAdjacentPosition(worker,
                    target.getX() + target.getY() * pgs.getWidth(), gs, null);
            if (move != null && gs.isUnitActionAllowed(worker, move)) {
                pa.addUnitAction(worker, move);
                assigned.add(worker.getID());
            }
        }

        // ── Step 2: Base trains emergency worker if affordable ─────────────────
        if (myBase != null && gs.getActionAssignment(myBase) == null) {
            if (gs.getPlayer(player).getResources() >= workerType.cost) {
                // Spawn toward the rush direction for faster intercept
                int bestDir = findBestSpawnDir(myBase, pgs, allEnemyThreats);
                UnitAction train = new UnitAction(UnitAction.TYPE_PRODUCE, bestDir, workerType);
                if (gs.isUnitActionAllowed(myBase, train)) {
                    pa.addUnitAction(myBase, train);
                }
            }
        }

        return pa;
    }

    /**
     * Find the best direction to spawn a unit from a building,
     * preferring directions that face toward the enemy threat.
     */
    private int findBestSpawnDir(Unit building, PhysicalGameState pgs, List<Unit> threats) {
        int bestDir = UnitAction.DIRECTION_DOWN;
        int bestScore = Integer.MIN_VALUE;

        for (int dir = 0; dir < 4; dir++) {
            int nx = building.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
            int ny = building.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
            if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
            if (pgs.getUnitAt(nx, ny) != null) continue;
            if (pgs.getTerrain(nx, ny) != PhysicalGameState.TERRAIN_NONE) continue;

            // Score: negative distance to closest threat (closer = better)
            int score = 0;
            if (!threats.isEmpty()) {
                int minDist = Integer.MAX_VALUE;
                for (Unit t : threats) {
                    int dist = Math.abs(nx - t.getX()) + Math.abs(ny - t.getY());
                    minDist = Math.min(minDist, dist);
                }
                score = -minDist;
            }
            if (score > bestScore) {
                bestScore = score;
                bestDir = dir;
            }
        }
        return bestDir;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  HEURISTIC POLICY (replaces policy LLM — instant, no HTTP)
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Generate a sensible PlayerAction without calling the LLM.
     * Used both as MCTS expansion policy and as game-loop fallback.
     *
     * Priority order per unit:
     *   Military: attack nearest enemy > move toward nearest enemy
     *   Worker:   return resources > harvest > move to resource
     *   Base:     train worker if below threshold
     *   Barracks: train ranged (cheap, flexible) if affordable
     */
    private PlayerAction buildHeuristicAction(int player, GameState gs, PhysicalGameState pgs) {
        PlayerAction pa = new PlayerAction();

        List<Unit> myUnits = new ArrayList<>();
        List<Unit> enemies = new ArrayList<>();
        List<Unit> resources = new ArrayList<>();
        Unit myBase = null;
        int myWorkerCount = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                myUnits.add(u);
                if (u.getType() == workerType) myWorkerCount++;
                if (u.getType() == baseType) myBase = u;
            } else if (u.getPlayer() == -1) {
                resources.add(u);
            } else {
                enemies.add(u);
            }
        }

        final int finalWorkerCount = myWorkerCount;
        final Unit finalBase = myBase;

        for (Unit unit : myUnits) {
            if (gs.getActionAssignment(unit) != null) continue; // already has action

            UnitAction ua = null;

            // ── Military units ──────────────────────────────────────────────
            if (unit.getType() == lightType || unit.getType() == heavyType
                    || unit.getType() == rangedType) {
                ua = militaryAction(unit, enemies, gs, pgs);
            }

            // ── Workers ─────────────────────────────────────────────────────
            else if (unit.getType() == workerType) {
                ua = workerAction(unit, enemies, resources, finalBase, gs, pgs);
            }

            // ── Base ────────────────────────────────────────────────────────
            else if (unit.getType() == baseType) {
                // Train worker if we have fewer than 2
                if (finalWorkerCount < 2
                        && gs.getPlayer(player).getResources() >= workerType.cost) {
                    ua = trainUnit(unit, workerType, enemies, pgs);
                }
            }

            // ── Barracks ────────────────────────────────────────────────────
            else if (unit.getType() == barracksType) {
                // Train ranged as default — cheap and good range
                if (gs.getPlayer(player).getResources() >= rangedType.cost) {
                    ua = trainUnit(unit, rangedType, enemies, pgs);
                } else if (gs.getPlayer(player).getResources() >= lightType.cost) {
                    ua = trainUnit(unit, lightType, enemies, pgs);
                }
            }

            if (ua != null && gs.isUnitActionAllowed(unit, ua)) {
                pa.addUnitAction(unit, ua);
            }
        }

        return pa;
    }

    private UnitAction militaryAction(Unit unit, List<Unit> enemies,
                                       GameState gs, PhysicalGameState pgs) {
        if (enemies.isEmpty()) return null;

        // Find closest enemy
        Unit target = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit e : enemies) {
            int dist = Math.abs(unit.getX() - e.getX()) + Math.abs(unit.getY() - e.getY());
            if (dist < bestDist) { bestDist = dist; target = e; }
        }
        if (target == null) return null;

        // Attack if in range
        if (bestDist <= unit.getType().attackRange) {
            return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                    target.getX() + target.getY() * pgs.getWidth());
        }

        // Move toward target
        UnitAction move = pf.findPathToAdjacentPosition(unit,
                target.getX() + target.getY() * pgs.getWidth(), gs, null);
        return move;
    }

    private UnitAction workerAction(Unit worker, List<Unit> enemies, List<Unit> resources,
                                     Unit base, GameState gs, PhysicalGameState pgs) {
        // Return resources if carrying
        if (worker.getResources() > 0 && base != null) {
            int dist = Math.abs(worker.getX() - base.getX())
                     + Math.abs(worker.getY() - base.getY());
            if (dist == 1) {
                int dir = dirTo(worker.getX(), worker.getY(), base.getX(), base.getY());
                return new UnitAction(UnitAction.TYPE_RETURN, dir);
            }
            UnitAction move = pf.findPathToAdjacentPosition(worker,
                    base.getX() + base.getY() * pgs.getWidth(), gs, null);
            return move;
        }

        // Harvest nearest resource
        if (!resources.isEmpty()) {
            Unit res = nearestUnit(worker, resources);
            if (res != null) {
                int dist = Math.abs(worker.getX() - res.getX())
                         + Math.abs(worker.getY() - res.getY());
                if (dist == 1) {
                    int dir = dirTo(worker.getX(), worker.getY(), res.getX(), res.getY());
                    return new UnitAction(UnitAction.TYPE_HARVEST, dir);
                }
                UnitAction move = pf.findPathToAdjacentPosition(worker,
                        res.getX() + res.getY() * pgs.getWidth(), gs, null);
                return move;
            }
        }

        // Attack if no resources and enemies nearby
        if (!enemies.isEmpty()) {
            return militaryAction(worker, enemies, gs, pgs);
        }

        return null;
    }

    private UnitAction trainUnit(Unit building, UnitType trainType,
                                  List<Unit> enemies, PhysicalGameState pgs) {
        // Pick spawn direction facing toward nearest enemy
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
                Unit nearest = nearestUnit(building.getX(), building.getY(), enemies);
                if (nearest != null) {
                    score = -(Math.abs(nx - nearest.getX()) + Math.abs(ny - nearest.getY()));
                }
            }
            if (score > bestScore) { bestScore = score; bestDir = dir; }
        }

        return new UnitAction(UnitAction.TYPE_PRODUCE, bestDir, trainType);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  MCTS CORE
    // ══════════════════════════════════════════════════════════════════════════

    private PlayerAction runMCTS(int player, GameState gs) throws Exception {
        MCTSNode root = new MCTSNode(null, null, gs.clone(), player);

        for (int i = 0; i < MCTS_ITERATIONS; i++) {
            MCTSNode selected = select(root);

            // Expand using heuristic policy — instant, no LLM
            if (!selected.expanded) {
                expandWithHeuristic(selected);
            }

            MCTSNode toSimulate = selected;
            if (!selected.children.isEmpty()) {
                toSimulate = selected.children.stream()
                        .filter(c -> c.visitCount == 0)
                        .findFirst()
                        .orElse(selected.children.get(
                                new Random().nextInt(selected.children.size())));
            }

            GameState simState = simulate(toSimulate.state.clone(), player);

            // LLM eval only on LAST iteration — one call per full MCTS search
            double value;
            if (i == MCTS_ITERATIONS - 1) {
                value = evaluateWithLLM(player, simState);
            } else {
                value = evaluateWithHeuristic(player, simState);
            }

            backpropagate(toSimulate, value);
        }

        return bestAction(root);
    }

    // ── Selection ──────────────────────────────────────────────────────────────

    private MCTSNode select(MCTSNode node) {
        while (!node.children.isEmpty()) {
            Optional<MCTSNode> unvisited = node.children.stream()
                    .filter(c -> c.visitCount == 0).findFirst();
            if (unvisited.isPresent()) return unvisited.get();

            node = node.children.stream()
                    .max(Comparator.comparingDouble(MCTSNode::ucb1))
                    .orElse(node);
            if (!node.expanded) break;
        }
        return node;
    }

    // ── Expansion (heuristic, no LLM) ─────────────────────────────────────────

    private void expandWithHeuristic(MCTSNode node) {
        node.expanded = true;
        try {
            PhysicalGameState pgs = node.state.getPhysicalGameState();

            // Generate 3 candidate actions:
            // 1. Full heuristic action (all units)
            // 2. Attack-focused (military units only)
            // 3. Economy-focused (workers + train)
            List<PlayerAction> candidates = new ArrayList<>();
            candidates.add(buildHeuristicAction(node.player, node.state, pgs));
            candidates.add(buildAttackFocusedAction(node.player, node.state, pgs));
            candidates.add(buildEconomyFocusedAction(node.player, node.state, pgs));

            for (PlayerAction action : candidates) {
                if (action == null || action.isEmpty()) continue;
                try {
                    GameState childState = node.state.clone();
                    childState.issueSafe(action);
                    for (int t = 0; t < 3; t++) {
                        if (childState.cycle()) break;
                    }
                    node.children.add(new MCTSNode(node, action, childState, node.player));
                } catch (Exception e) {
                    // skip
                }
            }

            if (node.children.isEmpty()) {
                node.children.add(new MCTSNode(node, new PlayerAction(),
                        node.state.clone(), node.player));
            }
        } catch (Exception e) {
            System.err.println("[MCTS] Expansion error: " + e.getMessage());
            node.children.add(new MCTSNode(node, new PlayerAction(),
                    node.state.clone(), node.player));
        }
    }

    /** Attack-focused candidate: all military units advance, buildings train */
    private PlayerAction buildAttackFocusedAction(int player, GameState gs, PhysicalGameState pgs) {
        PlayerAction pa = new PlayerAction();
        List<Unit> enemies = new ArrayList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player && u.getPlayer() != -1) enemies.add(u);
        }
        for (Unit unit : pgs.getUnits()) {
            if (unit.getPlayer() != player) continue;
            if (gs.getActionAssignment(unit) != null) continue;
            UnitAction ua = null;
            if (unit.getType().canAttack && unit.getType() != baseType
                    && unit.getType() != barracksType) {
                ua = militaryAction(unit, enemies, gs, pgs);
            } else if (unit.getType() == barracksType
                    && gs.getPlayer(player).getResources() >= heavyType.cost) {
                ua = trainUnit(unit, heavyType, enemies, pgs);
            } else if (unit.getType() == baseType
                    && gs.getPlayer(player).getResources() >= workerType.cost) {
                ua = trainUnit(unit, workerType, enemies, pgs);
            }
            if (ua != null && gs.isUnitActionAllowed(unit, ua)) pa.addUnitAction(unit, ua);
        }
        return pa;
    }

    /** Economy-focused candidate: workers harvest, base trains workers */
    private PlayerAction buildEconomyFocusedAction(int player, GameState gs, PhysicalGameState pgs) {
        PlayerAction pa = new PlayerAction();
        List<Unit> resources = new ArrayList<>();
        List<Unit> enemies = new ArrayList<>();
        Unit base = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == -1) resources.add(u);
            else if (u.getPlayer() == player && u.getType() == baseType) base = u;
            else if (u.getPlayer() != player) enemies.add(u);
        }
        for (Unit unit : pgs.getUnits()) {
            if (unit.getPlayer() != player) continue;
            if (gs.getActionAssignment(unit) != null) continue;
            UnitAction ua = null;
            if (unit.getType() == workerType) {
                ua = workerAction(unit, enemies, resources, base, gs, pgs);
            } else if (unit.getType() == baseType
                    && gs.getPlayer(player).getResources() >= workerType.cost) {
                ua = trainUnit(unit, workerType, enemies, pgs);
            }
            if (ua != null && gs.isUnitActionAllowed(unit, ua)) pa.addUnitAction(unit, ua);
        }
        return pa;
    }

    // ── Simulation ─────────────────────────────────────────────────────────────

    private GameState simulate(GameState state, int player) {
        try {
            for (int t = 0; t < SIMULATION_DEPTH; t++) {
                if (state.cycle()) break;
            }
        } catch (Exception e) {
            // partial simulation ok
        }
        return state;
    }

    // ── Evaluation ─────────────────────────────────────────────────────────────

    /**
     * LLM evaluation with compact prompt + cache.
     * Only called once per full MCTS search (on last iteration).
     */
    private double evaluateWithLLM(int player, GameState gs) {
        try {
            String prompt = buildCompactEvalPrompt(player, gs);
            String cacheKey = Integer.toHexString(prompt.hashCode());

            String response;
            if (llmCache.containsKey(cacheKey)) {
                response = llmCache.get(cacheKey);
                System.out.println("[MCTS] Eval cache hit");
            } else {
                response = callLLM(prompt, EVAL_SYSTEM_PROMPT);
                llmCache.put(cacheKey, response);
            }

            JsonObject json = parseJsonResponse(response);
            if (json != null && json.has("score")) {
                double score = json.get("score").getAsDouble();
                return score / 100.0;
            }
        } catch (Exception e) {
            System.err.println("[MCTS] LLM eval failed: " + e.getMessage());
        }
        return evaluateWithHeuristic(player, gs);
    }

    /**
     * Compact eval prompt — ~100 chars vs ~800 chars of full state.
     * Enough for the LLM to score the position meaningfully.
     */
    private String buildCompactEvalPrompt(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemy = 1 - player;

        int aUnits = 0, eUnits = 0, aHP = 0, eHP = 0;
        boolean aBase = false, eBase = false, aBarracks = false;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                aUnits++; aHP += u.getHitPoints();
                if (u.getType() == baseType) aBase = true;
                if (u.getType() == barracksType) aBarracks = true;
            } else if (u.getPlayer() != -1) {
                eUnits++; eHP += u.getHitPoints();
                if (u.getType() == baseType) eBase = true;
            }
        }

        return String.format("t=%d A:[%du %dhp %dr base=%b brx=%b] E:[%du %dhp %dr base=%b]",
                gs.getTime(),
                aUnits, aHP, gs.getPlayer(player).getResources(), aBase, aBarracks,
                eUnits, eHP, gs.getPlayer(enemy).getResources(), eBase);
    }

    /**
     * Fast heuristic evaluation — no LLM, runs in microseconds.
     */
    private double evaluateWithHeuristic(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemy = 1 - player;

        double allyScore = 0, enemyScore = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == -1) continue;
            double val = getUnitValue(u);
            if (u.getPlayer() == player) allyScore += val;
            else enemyScore += val;
        }

        allyScore  += gs.getPlayer(player).getResources() * 0.5;
        enemyScore += gs.getPlayer(enemy).getResources()  * 0.5;

        double total = allyScore + enemyScore;
        if (total == 0) return 0.5;
        return allyScore / total;
    }

    private double getUnitValue(Unit u) {
        double hpFraction = (double) u.getHitPoints() / u.getType().hp;
        switch (u.getType().name) {
            case "Base":     return 10.0 * hpFraction;
            case "Barracks": return 5.0  * hpFraction;
            case "Heavy":    return 4.0  * hpFraction;
            case "Light":    return 2.5  * hpFraction;
            case "Ranged":   return 2.5  * hpFraction;
            case "Worker":   return 1.0  * hpFraction;
            default:         return 1.0  * hpFraction;
        }
    }

    // ── Backpropagation ────────────────────────────────────────────────────────

    private void backpropagate(MCTSNode node, double value) {
        while (node != null) {
            node.visitCount++;
            node.totalValue += value;
            node = node.parent;
        }
    }

    // ── Best Action ────────────────────────────────────────────────────────────

    private PlayerAction bestAction(MCTSNode root) {
        if (root.children.isEmpty()) return new PlayerAction();

        MCTSNode best = root.children.stream()
                .max(Comparator.comparingDouble(MCTSNode::avgValue))
                .orElse(root.children.get(0));

        System.out.printf("[MCTS] Best: avg=%.3f visits=%d%n",
                best.avgValue(), best.visitCount);

        // Store unit map for game thread to use in filterValidAction
        // Re-parse from best action's state
        if (best.actionTaken != null) {
            rebuildUnitMap(best.actionTaken, best.state.getPhysicalGameState(),
                    best.player, best.state);
        }

        return best.actionTaken != null ? best.actionTaken : new PlayerAction();
    }

    /**
     * Rebuild lastCandidateUnitMap from a PlayerAction by matching units
     * in the game state at the time the action was generated.
     * Used so filterValidAction can re-apply the action to the live game state.
     */
    private void rebuildUnitMap(PlayerAction action, PhysicalGameState pgs,
                                  int player, GameState gs) {
        Map<Long, UnitAction> map = new HashMap<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) continue;
            UnitAction ua = action.getAction(u);
            if (ua != null && ua.getType() != UnitAction.TYPE_NONE) {
                map.put(u.getID(), ua);
            }
        }
        lastCandidateUnitMap = map;
    }

    // ── Filter valid action ────────────────────────────────────────────────────

    private PlayerAction filterValidAction(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        PlayerAction valid = new PlayerAction();
        Map<Long, UnitAction> map = lastCandidateUnitMap;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) continue;
            if (gs.getActionAssignment(u) != null) continue;
            UnitAction ua = map.get(u.getID());
            if (ua != null && ua.getType() != UnitAction.TYPE_NONE
                    && gs.isUnitActionAllowed(u, ua)) {
                valid.addUnitAction(u, ua);
            }
        }
        return valid;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LLM API CALL
    // ══════════════════════════════════════════════════════════════════════════

    private String callLLM(String userPrompt, String systemPrompt) {
        try {
            URL url = new URL(API_URL);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(REQUEST_TIMEOUT);
            conn.setReadTimeout(REQUEST_TIMEOUT);

            JsonObject request = new JsonObject();
            request.addProperty("model", OLLAMA_MODEL);

            JsonArray messages = new JsonArray();
            JsonObject sysMsg = new JsonObject();
            sysMsg.addProperty("role", "system");
            sysMsg.addProperty("content", systemPrompt
                    + "\nRespond ONLY with valid JSON. No markdown, no backticks.");
            messages.add(sysMsg);

            JsonObject userMsg = new JsonObject();
            userMsg.addProperty("role", "user");
            userMsg.addProperty("content", userPrompt);
            messages.add(userMsg);

            request.add("messages", messages);
            JsonObject fmt = new JsonObject();
            fmt.addProperty("type", "json_object");
            request.add("response_format", fmt);
            request.addProperty("temperature", 0.2);
            request.addProperty("max_tokens", 64); // eval only needs score + reason

            try (OutputStream os = conn.getOutputStream()) {
                os.write(request.toString().getBytes(StandardCharsets.UTF_8));
            }

            int code = conn.getResponseCode();
            if (code == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder resp = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) resp.append(line);
                    JsonObject jsonResp = JsonParser.parseString(resp.toString()).getAsJsonObject();
                    JsonArray choices = jsonResp.getAsJsonArray("choices");
                    if (choices != null && choices.size() > 0) {
                        return choices.get(0).getAsJsonObject()
                                .getAsJsonObject("message")
                                .get("content").getAsString();
                    }
                }
            } else {
                System.err.println("[MCTS] API error " + code);
            }
        } catch (Exception e) {
            System.err.println("[MCTS] LLM call failed: " + e.getMessage());
        }
        return "{\"score\":50,\"reason\":\"error\"}";
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  UTILITIES
    // ══════════════════════════════════════════════════════════════════════════

    private JsonObject parseJsonResponse(String response) {
        response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
        try {
            return JsonParser.parseString(response).getAsJsonObject();
        } catch (Exception e) {
            int start = response.indexOf("{");
            int end = response.lastIndexOf("}") + 1;
            if (start >= 0 && end > start) {
                try {
                    return JsonParser.parseString(response.substring(start, end)).getAsJsonObject();
                } catch (Exception ignored) {}
            }
        }
        return null;
    }

    private Unit nearestUnit(Unit src, List<Unit> units) {
        return nearestUnit(src.getX(), src.getY(), units);
    }

    private Unit nearestUnit(int x, int y, List<Unit> units) {
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit u : units) {
            int dist = Math.abs(x - u.getX()) + Math.abs(y - u.getY());
            if (dist < bestDist) { bestDist = dist; best = u; }
        }
        return best;
    }

    private int dirTo(int fromX, int fromY, int toX, int toY) {
        int dx = toX - fromX;
        int dy = toY - fromY;
        if (Math.abs(dx) >= Math.abs(dy)) {
            return dx > 0 ? UnitAction.DIRECTION_RIGHT : UnitAction.DIRECTION_LEFT;
        } else {
            return dy > 0 ? UnitAction.DIRECTION_DOWN : UnitAction.DIRECTION_UP;
        }
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}