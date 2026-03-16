/*
 * yebot — LLMInformedMCTS + Worker Rush Defense
 *
 * Architecture :
 *   PRIMARY:  LLMInformedMCTS — the shared tournament library
 *             - 200ms MCTS time budget per tick, runs instantly
 *             - LLM consulted only ~2-3x per 500 ticks for strategic goals
 *             - LLMPolicyProbabilityDistribution biases search without blocking
 *   DEFENSE:  Worker Rush Detector — pure heuristic, runs every tick first
 *             - Intercepts early worker rushes before they reach base
 *             - Focus-fires weakest enemy workers, trains emergency workers
 *   FALLBACK: Heuristic action — used if LLMInformedMCTS returns empty/fails
 *             - Harvest, attack nearest, train units — no LLM, instant
 *
 *   yebot  fallback = lightweight heuristic (our own, no borrowed AI)
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
import ai.mcts.llmguided.LLMInformedMCTS;
import rts.*;
import rts.units.*;

import java.util.*;

public class yebot extends AbstractionLayerAI {

    // ─── Worker Rush Defense Config ────────────────────────────────────────────
    private static final int RUSH_DETECT_RADIUS    = 6;   // manhattan distance to base
    private static final int RUSH_WORKER_THRESHOLD = 2;   // min enemy workers to trigger
    private static final int RUSH_EARLY_TICKS      = 200; // only active in early game

    // ─── Unit Types ────────────────────────────────────────────────────────────
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ─── LLMInformedMCTS primary engine ────────────────────────────────────────
    private final LLMInformedMCTS searchAgent;
    private int lastSearchTick = -9999;
    private static final int SEARCH_INTERVAL = 1; // run every tick (it has its own 200ms budget)

    // ══════════════════════════════════════════════════════════════════════════
    //  CONSTRUCTORS / RESET
    // ══════════════════════════════════════════════════════════════════════════

    public yebot(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    public yebot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);

        // Initialize the shared MCTS engine exactly like AlliBot does
        LLMInformedMCTS tmp;
        try {
            tmp = new LLMInformedMCTS(a_utt);
        } catch (Exception e) {
            tmp = null;
            System.err.println("[yebot] LLMInformedMCTS init failed, using heuristic only: "
                    + e.getMessage());
        }
        searchAgent = tmp;
    }

    @Override
    public void reset() {
        super.reset();
        lastSearchTick = -9999;
        if (searchAgent != null) searchAgent.reset();
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
    //  MAIN ENTRY POINT
    // ══════════════════════════════════════════════════════════════════════════

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        // ── PRIORITY 1: Worker Rush Defense ───────────────────────────────────
        // Pure heuristic — runs every tick, instant, no LLM involved.
        // Fires before LLMInformedMCTS so even if search is slow, defense works.
        if (isWorkerRush(player, gs, pgs)) {
            System.out.println("[yebot] RUSH at t=" + gs.getTime() + " — defending");
            PlayerAction defense = buildWorkerRushDefense(player, gs, pgs);
            defense.fillWithNones(gs, player, 1);
            return defense;
        }

        // ── PRIORITY 2: LLMInformedMCTS ───────────────────────────────────────
        if (searchAgent != null
                && gs.getTime() - lastSearchTick >= SEARCH_INTERVAL) {
            try {
                PlayerAction searchAction = searchAgent.getAction(player, gs);
                lastSearchTick = gs.getTime();

                // Only use search result if it contains real (non-NONE) actions
                if (searchAction != null && searchAction.hasNonNoneActions()) {
                    return searchAction;
                }
            } catch (Exception e) {
                System.err.println("[yebot] LLMInformedMCTS failed at t=" + gs.getTime()
                        + ": " + e.getMessage());
            }
        }

        // ── PRIORITY 3: Heuristic fallback ────────────────────────────────────
        // Our own logic — no borrowed AI, no LLM calls.
        PlayerAction fallback = buildHeuristicAction(player, gs, pgs);
        fallback.fillWithNones(gs, player, 1);
        return fallback;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  WORKER RUSH DETECTION
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Detect a worker rush: 2+ enemy workers within distance 6 of our base
     * in the first 200 ticks.
     */
    private boolean isWorkerRush(int player, GameState gs, PhysicalGameState pgs) {
        if (gs.getTime() > RUSH_EARLY_TICKS) return false;

        Unit myBase = null;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType() == baseType) {
                myBase = u;
                break;
            }
        }
        if (myBase == null) return false;

        int rushingWorkers = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player || u.getPlayer() == -1) continue;
            if (u.getType() != workerType) continue;
            int dist = Math.abs(u.getX() - myBase.getX())
                     + Math.abs(u.getY() - myBase.getY());
            if (dist <= RUSH_DETECT_RADIUS) rushingWorkers++;
        }

        return rushingWorkers >= RUSH_WORKER_THRESHOLD;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  WORKER RUSH DEFENSE
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Worker rush defense logic.
     *
     * Worker rushes are a numbers game — each worker has 1 HP, 1 damage.
     * Winning strategy:
     *   1. Focus-fire weakest enemy workers first (maximize kills per tick)
     *   2. All idle ally workers intercept — don't harvest during a rush
     *   3. Base spams workers toward the rush direction
     *   4. Any combat units also engage
     */
    private PlayerAction buildWorkerRushDefense(int player, GameState gs,
                                                  PhysicalGameState pgs) {
        PlayerAction pa = new PlayerAction();

        Unit myBase = null;
        List<Unit> myWorkers    = new ArrayList<>();
        List<Unit> myCombat     = new ArrayList<>();
        List<Unit> enemyWorkers = new ArrayList<>();
        List<Unit> enemyCombat  = new ArrayList<>();

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == baseType)   myBase = u;
                if (u.getType() == workerType) myWorkers.add(u);
                if (u.getType().canAttack
                        && u.getType() != baseType
                        && u.getType() != barracksType
                        && u.getType() != workerType) myCombat.add(u);
            } else if (u.getPlayer() != -1) {
                if (u.getType() == workerType) enemyWorkers.add(u);
                else if (u.getType().canAttack) enemyCombat.add(u);
            }
        }

        // Sort enemy workers by HP asc — kill weakest first for guaranteed eliminations
        enemyWorkers.sort(Comparator.comparingInt(Unit::getHitPoints));

        List<Unit> threats = new ArrayList<>(enemyWorkers);
        threats.addAll(enemyCombat);

        // ── All my workers intercept ──────────────────────────────────────────
        for (Unit worker : myWorkers) {
            if (gs.getActionAssignment(worker) != null) continue;
            if (threats.isEmpty()) break;

            Unit target = nearestUnit(worker, threats);
            if (target == null) continue;

            int dist = Math.abs(worker.getX() - target.getX())
                     + Math.abs(worker.getY() - target.getY());

            if (dist <= worker.getType().attackRange) {
                UnitAction atk = new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                        target.getX() + target.getY() * pgs.getWidth());
                if (gs.isUnitActionAllowed(worker, atk)) {
                    pa.addUnitAction(worker, atk);
                    // Expect to kill — remove from threat list
                    if (target.getHitPoints() <= worker.getType().maxDamage) {
                        threats.remove(target);
                    }
                    continue;
                }
            }

            UnitAction move = pf.findPathToAdjacentPosition(worker,
                    target.getX() + target.getY() * pgs.getWidth(), gs, null);
            if (move != null && gs.isUnitActionAllowed(worker, move)) {
                pa.addUnitAction(worker, move);
            }
        }

        // ── Combat units also engage ──────────────────────────────────────────
        for (Unit unit : myCombat) {
            if (gs.getActionAssignment(unit) != null) continue;
            if (threats.isEmpty()) break;

            Unit target = nearestUnit(unit, threats);
            if (target == null) continue;

            int dist = Math.abs(unit.getX() - target.getX())
                     + Math.abs(unit.getY() - target.getY());

            if (dist <= unit.getType().attackRange) {
                UnitAction atk = new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                        target.getX() + target.getY() * pgs.getWidth());
                if (gs.isUnitActionAllowed(unit, atk)) {
                    pa.addUnitAction(unit, atk);
                    continue;
                }
            }

            UnitAction move = pf.findPathToAdjacentPosition(unit,
                    target.getX() + target.getY() * pgs.getWidth(), gs, null);
            if (move != null && gs.isUnitActionAllowed(unit, move)) {
                pa.addUnitAction(unit, move);
            }
        }

        // ── Base trains emergency worker toward rush ───────────────────────────
        if (myBase != null && gs.getActionAssignment(myBase) == null
                && gs.getPlayer(player).getResources() >= workerType.cost) {
            int dir = findBestSpawnDir(myBase, pgs, threats);
            UnitAction train = new UnitAction(UnitAction.TYPE_PRODUCE, dir, workerType);
            if (gs.isUnitActionAllowed(myBase, train)) {
                pa.addUnitAction(myBase, train);
            }
        }

        return pa;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  HEURISTIC FALLBACK
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Lightweight heuristic action — no LLM, no borrowed AI.
     * Priority per unit type:
     *   Military  → attack nearest enemy, or move toward them
     *   Worker    → return if carrying → harvest → move to resource
     *   Base      → train worker if < 2 workers
     *   Barracks  → train ranged (cheapest combat unit with range advantage)
     */
    private PlayerAction buildHeuristicAction(int player, GameState gs,
                                               PhysicalGameState pgs) {
        PlayerAction pa = new PlayerAction();

        List<Unit> myUnits   = new ArrayList<>();
        List<Unit> enemies   = new ArrayList<>();
        List<Unit> resources = new ArrayList<>();
        Unit myBase = null;
        int workerCount = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                myUnits.add(u);
                if (u.getType() == workerType) workerCount++;
                if (u.getType() == baseType)   myBase = u;
            } else if (u.getPlayer() == -1) {
                resources.add(u);
            } else {
                enemies.add(u);
            }
        }

        final int wc = workerCount;
        final Unit base = myBase;

        for (Unit unit : myUnits) {
            if (gs.getActionAssignment(unit) != null) continue;

            UnitAction ua = null;

            if (unit.getType() == lightType || unit.getType() == heavyType
                    || unit.getType() == rangedType) {
                ua = heuristicMilitary(unit, enemies, gs, pgs);

            } else if (unit.getType() == workerType) {
                ua = heuristicWorker(unit, enemies, resources, base, gs, pgs);

            } else if (unit.getType() == baseType) {
                if (wc < 2 && gs.getPlayer(player).getResources() >= workerType.cost) {
                    ua = bestTrain(unit, workerType, enemies, pgs);
                }

            } else if (unit.getType() == barracksType) {
                if (gs.getPlayer(player).getResources() >= rangedType.cost) {
                    ua = bestTrain(unit, rangedType, enemies, pgs);
                } else if (gs.getPlayer(player).getResources() >= lightType.cost) {
                    ua = bestTrain(unit, lightType, enemies, pgs);
                }
            }

            if (ua != null && gs.isUnitActionAllowed(unit, ua)) {
                pa.addUnitAction(unit, ua);
            }
        }

        return pa;
    }

    private UnitAction heuristicMilitary(Unit unit, List<Unit> enemies,
                                          GameState gs, PhysicalGameState pgs) {
        if (enemies.isEmpty()) return null;
        Unit target = nearestUnit(unit, enemies);
        if (target == null) return null;

        int dist = Math.abs(unit.getX() - target.getX())
                 + Math.abs(unit.getY() - target.getY());

        if (dist <= unit.getType().attackRange) {
            return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                    target.getX() + target.getY() * pgs.getWidth());
        }
        return pf.findPathToAdjacentPosition(unit,
                target.getX() + target.getY() * pgs.getWidth(), gs, null);
    }

    private UnitAction heuristicWorker(Unit worker, List<Unit> enemies,
                                        List<Unit> resources, Unit base,
                                        GameState gs, PhysicalGameState pgs) {
        // Return resources if carrying
        if (worker.getResources() > 0 && base != null) {
            int dist = Math.abs(worker.getX() - base.getX())
                     + Math.abs(worker.getY() - base.getY());
            if (dist == 1) {
                return new UnitAction(UnitAction.TYPE_RETURN,
                        dirTo(worker.getX(), worker.getY(), base.getX(), base.getY()));
            }
            return pf.findPathToAdjacentPosition(worker,
                    base.getX() + base.getY() * pgs.getWidth(), gs, null);
        }

        // Harvest
        if (!resources.isEmpty()) {
            Unit res = nearestUnit(worker, resources);
            if (res != null) {
                int dist = Math.abs(worker.getX() - res.getX())
                         + Math.abs(worker.getY() - res.getY());
                if (dist == 1) {
                    return new UnitAction(UnitAction.TYPE_HARVEST,
                            dirTo(worker.getX(), worker.getY(), res.getX(), res.getY()));
                }
                return pf.findPathToAdjacentPosition(worker,
                        res.getX() + res.getY() * pgs.getWidth(), gs, null);
            }
        }

        // No resources — go fight
        return heuristicMilitary(worker, enemies, gs, pgs);
    }

    private UnitAction bestTrain(Unit building, UnitType trainType,
                                  List<Unit> enemies, PhysicalGameState pgs) {
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
                if (nearest != null) {
                    score = -(Math.abs(nx - nearest.getX()) + Math.abs(ny - nearest.getY()));
                }
            }
            if (score > bestScore) { bestScore = score; bestDir = dir; }
        }
        return new UnitAction(UnitAction.TYPE_PRODUCE, bestDir, trainType);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  UTILITIES
    // ══════════════════════════════════════════════════════════════════════════

    private int findBestSpawnDir(Unit building, PhysicalGameState pgs,
                                   List<Unit> threats) {
        int bestDir = UnitAction.DIRECTION_DOWN;
        int bestScore = Integer.MIN_VALUE;

        for (int dir = 0; dir < 4; dir++) {
            int nx = building.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
            int ny = building.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
            if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
            if (pgs.getUnitAt(nx, ny) != null) continue;
            if (pgs.getTerrain(nx, ny) != PhysicalGameState.TERRAIN_NONE) continue;

            int score = 0;
            if (!threats.isEmpty()) {
                int minDist = Integer.MAX_VALUE;
                for (Unit t : threats) {
                    int d = Math.abs(nx - t.getX()) + Math.abs(ny - t.getY());
                    minDist = Math.min(minDist, d);
                }
                score = -minDist;
            }
            if (score > bestScore) { bestScore = score; bestDir = dir; }
        }
        return bestDir;
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

    private int dirTo(int fx, int fy, int tx, int ty) {
        int dx = tx - fx, dy = ty - fy;
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