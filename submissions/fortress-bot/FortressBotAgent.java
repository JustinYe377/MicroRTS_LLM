package ai.abstraction.submissions.fortress_bot;

import ai.abstraction.AbstractAction;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.Harvest;
import ai.abstraction.WorkerRushPlusPlus;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

public class FortressBotAgent extends AbstractionLayerAI {

    private enum StrategyMode {
        BOOM_THEN_PUSH,
        RANGED_CONTAIN,
        HEAVY_BREAKTHROUGH,
        WORKER_SWARM,
        SCOUT_AND_RAID,
        FORTRESS_DEFENSE,
        ALL_IN_PUSH
    }

    private static class StrategyDirective {
        StrategyMode mode;
        int holdTicks;
        int targetWorkers;
        int targetBarracks;
        int attackWaveSize;
        int wavePeriod;
        int defendRadius;
        double aggression;
        int raidWorkers;
        boolean expandBase;
        String primaryUnit;
        String rationale;

        static StrategyDirective defaults() {
            StrategyDirective d = new StrategyDirective();
            d.mode = StrategyMode.BOOM_THEN_PUSH;
            d.holdTicks = 110;
            d.targetWorkers = 5;
            d.targetBarracks = 1;
            d.attackWaveSize = 4;
            d.wavePeriod = 120;
            d.defendRadius = 7;
            d.aggression = 0.55;
            d.raidWorkers = 0;
            d.expandBase = false;
            d.primaryUnit = "RANGED";
            d.rationale = "fallback";
            return d;
        }

        StrategyDirective copy() {
            StrategyDirective d = new StrategyDirective();
            d.mode = mode;
            d.holdTicks = holdTicks;
            d.targetWorkers = targetWorkers;
            d.targetBarracks = targetBarracks;
            d.attackWaveSize = attackWaveSize;
            d.wavePeriod = wavePeriod;
            d.defendRadius = defendRadius;
            d.aggression = aggression;
            d.raidWorkers = raidWorkers;
            d.expandBase = expandBase;
            d.primaryUnit = primaryUnit;
            d.rationale = rationale;
            return d;
        }

        void absorb(StrategyDirective incoming) {
            if (incoming == null) {
                return;
            }
            targetWorkers = incoming.targetWorkers;
            targetBarracks = incoming.targetBarracks;
            attackWaveSize = incoming.attackWaveSize;
            wavePeriod = incoming.wavePeriod;
            defendRadius = incoming.defendRadius;
            aggression = incoming.aggression;
            raidWorkers = incoming.raidWorkers;
            expandBase = incoming.expandBase;
            primaryUnit = incoming.primaryUnit;
            rationale = incoming.rationale;
        }
    }

    private static class StateSnapshot {
        int myWorkerCount;
        int myCombatCount;
        int myBaseCount;
        int myBarracksCount;
        int myLightCount;
        int myRangedCount;
        int myHeavyCount;

        int enemyWorkerCount;
        int enemyCombatCount;
        int enemyBaseCount;
        int enemyBarracksCount;
        int enemyLightCount;
        int enemyRangedCount;
        int enemyHeavyCount;

        int myResources;
        int enemyResources;
        int enemyThreatNearBase;
        int enemyClosestToBase;
        int enemyWorkersNearBase;
        int enemyClosestWorkerToBase;
        int mapWidth;
        int mapHeight;
        int resourceNodeCount;

        List<Unit> myBases = new ArrayList<>();
        List<Unit> enemyBases = new ArrayList<>();
        List<Unit> myUnits = new ArrayList<>();
        List<Unit> enemyUnits = new ArrayList<>();

        boolean enemyCollapsed;
    }

    private static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String OLLAMA_MODEL =
            System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    private static final boolean DEBUG =
            Boolean.parseBoolean(System.getenv().getOrDefault("ORACLE_STRAT_DEBUG", "false"));
    private static final int CONSULT_INTERVAL = getEnvInt("OLLAMA_SUBMISSION_INTERVAL", 160);
    private static final int CONNECT_TIMEOUT_MS = getEnvInt("OLLAMA_SUBMISSION_CONNECT_TIMEOUT_MS", 1800);
    private static final int READ_TIMEOUT_MS = getEnvInt("OLLAMA_SUBMISSION_READ_TIMEOUT_MS", 15000);

    private static final int MIN_HOLD_TICKS = 75;
    private static final int OPENING_END_TICK = 360;
    private static final int OPENING_WORKER_TARGET = 6;
    private static final int OPENING_WAVE_SIZE = 3;
    private static final int THREAT_RADIUS = 7;
    private static final int BOOM_EARLY_TICK = 650;
    private static final int BOOM_MIN_HARVESTERS_EARLY = 4;
    private static final int BOOM_MIN_HARVESTERS_LATE = 3;
    private static final int HOME_RESERVE_THREAT_RADIUS = 4;
    private static final int HOME_RESERVE_MAX_CHASE_DISTANCE = 2;
    private static final int EARLY_BARRACKS_SNIPE_END_TICK = 650;
    private static final int LARGE_ADVANTAGE_COMBAT_DELTA = 3;
    private static final int WORKER_RUSH_DEFENSE_END_TICK = 900;
    private static final int WORKER_RUSH_TRIGGER_NEAR_BASE = 2;
    private static final int SMALL_MAP_AREA = 64;
    private static final int LARGE_MAP_AREA = 196;

    private static final Pattern JSON_OBJECT = Pattern.compile("\\{.*\\}", Pattern.DOTALL);

    protected UnitTypeTable utt;
    private UnitType workerType;
    private UnitType baseType;
    private UnitType barracksType;
    private UnitType lightType;
    private UnitType rangedType;
    private UnitType heavyType;

    private StrategyDirective currentDirective = StrategyDirective.defaults();
    private int lastConsultTick = Integer.MIN_VALUE / 4;
    private int modeLockedUntilTick = 0;
    private int lastWaveTick = Integer.MIN_VALUE / 4;
    private AI workerRushMirror;
    private boolean workerRushMirrorLatched;

    public FortressBotAgent(UnitTypeTable aUtt) {
        super(new AStarPathFinding());
        reset(aUtt);
    }

    @Override
    public void reset() {
        super.reset();
        currentDirective = StrategyDirective.defaults();
        lastConsultTick = Integer.MIN_VALUE / 4;
        modeLockedUntilTick = 0;
        lastWaveTick = Integer.MIN_VALUE / 4;
        workerRushMirrorLatched = false;
        if (workerRushMirror != null) {
            workerRushMirror.reset();
        }
    }

    public void reset(UnitTypeTable aUtt) {
        utt = aUtt;
        if (utt != null) {
            workerType = utt.getUnitType("Worker");
            baseType = utt.getUnitType("Base");
            barracksType = utt.getUnitType("Barracks");
            lightType = utt.getUnitType("Light");
            rangedType = utt.getUnitType("Ranged");
            heavyType = utt.getUnitType("Heavy");
            workerRushMirror = new WorkerRushPlusPlus(utt, new AStarPathFinding());
        }
        reset();
    }

    @Override
    public AI clone() {
        FortressBotAgent clone = new FortressBotAgent(utt);
        clone.currentDirective = currentDirective.copy();
        clone.lastConsultTick = lastConsultTick;
        clone.modeLockedUntilTick = modeLockedUntilTick;
        clone.lastWaveTick = lastWaveTick;
        clone.workerRushMirrorLatched = workerRushMirrorLatched;
        if (workerRushMirror != null) {
            clone.workerRushMirror = workerRushMirror.clone();
        }
        return clone;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (!gs.canExecuteAnyAction(player)) {
            return new PlayerAction();
        }

        StateSnapshot snapshot = inspectState(player, gs);
        updateWorkerRushMirrorLatch(gs, snapshot);

        if (shouldDelegateToWorkerRushMirror(gs, snapshot) && workerRushMirror != null) {
            try {
                return workerRushMirror.getAction(player, gs);
            } catch (Throwable e) {
                workerRushMirrorLatched = false;
                debug("T=%d mirror-delegate failed: %s", gs.getTime(), e.getMessage());
            }
        }

        if (shouldRunScriptedOpening(gs, snapshot)) {
            executeScriptedOpening(player, gs, snapshot);
            return translateActions(player, gs);
        }

        int consultInterval = consultIntervalFor(snapshot);
        StrategyDirective proposal;
        if (gs.getTime() == 0 || gs.getTime() - lastConsultTick >= consultInterval) {
            proposal = askOllama(player, gs, snapshot, consultInterval);
            if (proposal == null) {
                proposal = fallbackDirective(snapshot, gs);
            }
            lastConsultTick = gs.getTime();
        } else {
            proposal = currentDirective.copy();
        }

        applyDirectiveWithLock(gs, snapshot, proposal);

        if (DEBUG && gs.getTime() % 120 == 0) {
            debug(
                    "T=%d mode=%s lock=%d myW=%d myC=%d enemyC=%d threat=%d rationale=%s",
                    gs.getTime(),
                    currentDirective.mode,
                    modeLockedUntilTick,
                    snapshot.myWorkerCount,
                    snapshot.myCombatCount,
                    snapshot.enemyCombatCount,
                    snapshot.enemyThreatNearBase,
                    currentDirective.rationale);
        }

        executeDirective(player, gs, snapshot, currentDirective);
        return translateActions(player, gs);
    }

    private boolean shouldRunScriptedOpening(GameState gs, StateSnapshot snapshot) {
        if (gs.getTime() > OPENING_END_TICK || !snapshot.enemyCollapsed) {
            return false;
        }
        if (snapshot.enemyThreatNearBase > 0 || snapshot.enemyWorkersNearBase > 0) {
            return false;
        }
        if (isSmallMap(snapshot) && snapshot.enemyClosestWorkerToBase <= THREAT_RADIUS + 1) {
            return false;
        }
        return true;
    }

    private boolean shouldDelegateToWorkerRushMirror(GameState gs, StateSnapshot snapshot) {
        if (snapshot == null || gs.getTime() > 1490) {
            return false;
        }
        if (isSmallMap(snapshot)) {
            return true;
        }
        return shouldDelegateLargeMapPressure(gs, snapshot);
    }

    private boolean shouldUseAntiWorkerRushAllIn(GameState gs, StateSnapshot snapshot) {
        return false;
    }

    private void updateWorkerRushMirrorLatch(GameState gs, StateSnapshot snapshot) {
        if (snapshot == null || !isSmallMap(snapshot) || workerRushMirrorLatched) {
            return;
        }
        // Strict signature: many enemy workers committed forward early with no military tech.
        boolean earlyAllWorkerCommit = gs.getTime() <= 420
                && snapshot.enemyCombatCount == 0
                && snapshot.enemyBarracksCount == 0
                && snapshot.enemyWorkerCount >= 4
                && snapshot.enemyClosestWorkerToBase <= THREAT_RADIUS + 2;
        boolean immediateSwarm = gs.getTime() <= 300
                && snapshot.enemyBarracksCount == 0
                && snapshot.enemyWorkersNearBase >= WORKER_RUSH_TRIGGER_NEAR_BASE;
        if (earlyAllWorkerCommit || immediateSwarm) {
            workerRushMirrorLatched = true;
        }
    }

    private boolean shouldDelegateLargeMapPressure(GameState gs, StateSnapshot snapshot) {
        if (!isLargeMap(snapshot) || gs.getTime() > 1050) {
            return false;
        }
        boolean enemyCommittedTech = snapshot.enemyBarracksCount > 0 || snapshot.enemyCombatCount > 0;
        boolean noSkirmishersSeen = snapshot.enemyLightCount == 0 && snapshot.enemyRangedCount == 0;
        boolean noImmediateWorkerSwarm = snapshot.enemyWorkersNearBase <= 1 && snapshot.enemyThreatNearBase <= 1;
        return enemyCommittedTech && noSkirmishersSeen && noImmediateWorkerSwarm;
    }

    private void executeScriptedOpening(int player, GameState gs, StateSnapshot snapshot) {
        StrategyDirective opening = StrategyDirective.defaults();
        opening.mode = StrategyMode.BOOM_THEN_PUSH;
        opening.targetWorkers = OPENING_WORKER_TARGET + 1;
        opening.targetBarracks = 1;
        opening.attackWaveSize = OPENING_WAVE_SIZE;
        opening.wavePeriod = 80;
        opening.defendRadius = 5;
        opening.aggression = 0.60;
        opening.raidWorkers = 0;
        opening.expandBase = snapshot.myWorkerCount >= 7 && snapshot.myBaseCount < 2;
        opening.primaryUnit = "RANGED";
        opening.rationale = "scripted-greedy-opening";

        executeDirective(player, gs, snapshot, opening);
    }

    private void applyDirectiveWithLock(GameState gs, StateSnapshot snapshot, StrategyDirective proposal) {
        boolean antiWorkerRushAllIn = shouldUseAntiWorkerRushAllIn(gs, snapshot);
        if (antiWorkerRushAllIn) {
            proposal.mode = StrategyMode.ALL_IN_PUSH;
            proposal.holdTicks = Math.max(proposal.holdTicks, 120);
            proposal.targetWorkers = Math.max(proposal.targetWorkers, Math.min(8, Math.max(4, snapshot.enemyWorkerCount + 1)));
            proposal.targetBarracks = 0;
            proposal.attackWaveSize = 1;
            proposal.wavePeriod = 35;
            proposal.defendRadius = 5;
            proposal.aggression = Math.max(proposal.aggression, 0.95);
            proposal.raidWorkers = 6;
            proposal.expandBase = false;
            proposal.primaryUnit = "LIGHT";
            proposal.rationale = "small-map-anti-worker-rush-all-in";
        }

        boolean workerRushDefense = shouldActivateWorkerRushDefense(gs, snapshot);
        if (workerRushDefense && !antiWorkerRushAllIn) {
            proposal.mode = StrategyMode.FORTRESS_DEFENSE;
            proposal.holdTicks = Math.max(proposal.holdTicks, 120);
            proposal.targetWorkers = Math.max(proposal.targetWorkers, Math.min(8, Math.max(4, snapshot.enemyWorkersNearBase + 2)));
            proposal.targetBarracks = 1;
            proposal.attackWaveSize = Math.max(3, Math.min(5, proposal.attackWaveSize));
            proposal.wavePeriod = Math.min(proposal.wavePeriod, 95);
            proposal.defendRadius = Math.max(proposal.defendRadius, 7);
            proposal.raidWorkers = 0;
            proposal.expandBase = false;
            proposal.primaryUnit = "LIGHT";
            proposal.rationale = "worker-rush-defense";
        }

        boolean forcedAllOutRush = !workerRushDefense && !antiWorkerRushAllIn && shouldForceAllOutRush(gs, snapshot);
        if (forcedAllOutRush) {
            proposal.mode = StrategyMode.ALL_IN_PUSH;
            proposal.holdTicks = Math.max(proposal.holdTicks, 120);
            proposal.raidWorkers = Math.max(proposal.raidWorkers, 4);
            proposal.rationale = "forced-all-out-rush";
        }

        boolean largeMapCloseout = isLargeMap(snapshot)
                && gs.getTime() >= 700
                && snapshot.enemyCombatCount == 0
                && snapshot.myCombatCount >= 1;
        if (largeMapCloseout) {
            proposal.mode = StrategyMode.ALL_IN_PUSH;
            proposal.holdTicks = Math.max(proposal.holdTicks, 120);
            proposal.targetBarracks = Math.max(proposal.targetBarracks, 2);
            proposal.attackWaveSize = 1;
            proposal.wavePeriod = Math.min(proposal.wavePeriod, 60);
            proposal.raidWorkers = Math.max(proposal.raidWorkers, 2);
            proposal.aggression = Math.max(proposal.aggression, 0.85);
            proposal.rationale = "large-map-closeout";
        }

        StrategyMode emergencyMode = emergencyOverride(snapshot);
        if (emergencyMode != null) {
            proposal.mode = emergencyMode;
            proposal.holdTicks = Math.max(proposal.holdTicks, 90);
            proposal.rationale = "emergency-defense";
        }

        boolean locked = gs.getTime() < modeLockedUntilTick;
        boolean wantsSwitch = proposal.mode != currentDirective.mode;
        boolean canSwitch = !locked
                || emergencyMode != null
                || workerRushDefense
                || antiWorkerRushAllIn
                || forcedAllOutRush
                || largeMapCloseout
                || proposal.mode == StrategyMode.FORTRESS_DEFENSE;

        if (wantsSwitch && canSwitch) {
            currentDirective = proposal.copy();
            modeLockedUntilTick = gs.getTime() + Math.max(MIN_HOLD_TICKS, proposal.holdTicks);
            return;
        }

        if (!wantsSwitch) {
            currentDirective = proposal.copy();
            if (gs.getTime() >= modeLockedUntilTick) {
                modeLockedUntilTick = gs.getTime() + Math.max(MIN_HOLD_TICKS, proposal.holdTicks);
            }
            return;
        }

        currentDirective.absorb(proposal);
    }

    private StrategyMode emergencyOverride(StateSnapshot snapshot) {
        if (snapshot.myBaseCount == 0) {
            return StrategyMode.WORKER_SWARM;
        }
        if (isLargeMap(snapshot)
                && snapshot.enemyHeavyCount > 0
                && snapshot.enemyClosestToBase <= 10) {
            return StrategyMode.FORTRESS_DEFENSE;
        }
        if (snapshot.enemyThreatNearBase >= 3
                || (snapshot.enemyClosestToBase <= 5 && snapshot.enemyCombatCount > snapshot.myCombatCount + 1)) {
            return StrategyMode.FORTRESS_DEFENSE;
        }
        return null;
    }

    private void executeDirective(
            int player,
            GameState gs,
            StateSnapshot snapshot,
            StrategyDirective directive) {

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player me = gs.getPlayer(player);
        if (me == null) {
            return;
        }

        List<Unit> myWorkers = new ArrayList<>();
        List<Unit> myCombat = new ArrayList<>();
        List<Unit> myBarracks = new ArrayList<>();
        List<Unit> myBases = new ArrayList<>();

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) {
                continue;
            }
            if (u.getType() == workerType) {
                myWorkers.add(u);
            } else if (u.getType() == baseType) {
                myBases.add(u);
            } else if (u.getType() == barracksType) {
                myBarracks.add(u);
            } else if (u.getType().canAttack) {
                myCombat.add(u);
            }
        }

        Unit anchorBase = !myBases.isEmpty() ? myBases.get(0) : null;
        int resourcesLeft = me.getResources();
        List<Integer> reservedPositions = new LinkedList<>();

        List<Unit> freeWorkers = new ArrayList<>(myWorkers);
        boolean workerRushDefense = shouldActivateWorkerRushDefense(gs, snapshot);
        boolean antiWorkerRushAllIn = shouldUseAntiWorkerRushAllIn(gs, snapshot);

        if (myBases.isEmpty() && !freeWorkers.isEmpty() && resourcesLeft >= baseType.cost) {
            Unit builder = chooseBuilder(freeWorkers, anchorBase);
            if (builder == null) {
                builder = freeWorkers.get(0);
            }
            if (tryBuild(builder, baseType, builder.getX(), builder.getY(), reservedPositions, me, pgs)) {
                resourcesLeft -= baseType.cost;
                freeWorkers.remove(builder);
                anchorBase = builder;
            }
        }

        if (anchorBase == null && !freeWorkers.isEmpty()) {
            anchorBase = freeWorkers.get(0);
        }

        int queuedWorkers = 0;
        boolean delayBarracksForWorkers = false;
        if (myWorkers.size() < 3 && !myBases.isEmpty()) {
            for (Unit base : myBases) {
                if (gs.getActionAssignment(base) != null) {
                    continue;
                }
                if (myWorkers.size() + queuedWorkers >= 3 || resourcesLeft < workerType.cost) {
                    break;
                }
                train(base, workerType);
                resourcesLeft -= workerType.cost;
                queuedWorkers++;
            }
            delayBarracksForWorkers = myWorkers.size() + queuedWorkers < 3;
        }

        int desiredBarracks = clampInt(directive.targetBarracks, 0, 3);
        if (antiWorkerRushAllIn) {
            desiredBarracks = 0;
        }
        if (workerRushDefense) {
            boolean stabilizedDefense = snapshot.enemyWorkersNearBase == 0
                    && snapshot.myWorkerCount >= Math.max(4, snapshot.enemyWorkerCount);
            desiredBarracks = stabilizedDefense ? 1 : 0;
        }
        if (isLargeMap(snapshot) && snapshot.enemyCombatCount > 0) {
            desiredBarracks = Math.max(desiredBarracks, 1);
            if (snapshot.enemyHeavyCount >= 2) {
                desiredBarracks = Math.max(desiredBarracks, 2);
            }
        }
        if (!delayBarracksForWorkers
                && myBarracks.size() < desiredBarracks
                && !freeWorkers.isEmpty()
                && resourcesLeft >= barracksType.cost) {
            Unit builder = chooseBuilder(freeWorkers, anchorBase);
            if (builder != null) {
                int bx = anchorBase != null ? anchorBase.getX() : builder.getX();
                int by = anchorBase != null ? anchorBase.getY() : builder.getY();
                if (tryBuild(builder, barracksType, bx, by, reservedPositions, me, pgs)) {
                    resourcesLeft -= barracksType.cost;
                    freeWorkers.remove(builder);
                }
            }
        }

        boolean mustTechBeforeExpand = desiredBarracks > 0 && myBarracks.size() < 1;
        if (!workerRushDefense
                && directive.expandBase
                && !mustTechBeforeExpand
                && snapshot.enemyCombatCount == 0
                && myBases.size() < 2
                && !freeWorkers.isEmpty()
                && resourcesLeft >= baseType.cost) {
            Unit expandWorker = chooseExpansionWorker(freeWorkers, pgs, snapshot.enemyBases);
            if (expandWorker != null) {
                int[] pos = findExpansionPosition(expandWorker, pgs, snapshot.enemyBases);
                if (pos != null && tryBuild(expandWorker, baseType, pos[0], pos[1], reservedPositions, me, pgs)) {
                    resourcesLeft -= baseType.cost;
                    freeWorkers.remove(expandWorker);
                }
            }
        }

        int workerGoal = clampInt(directive.targetWorkers, 1, 10);
        if (workerRushDefense) {
            workerGoal = Math.max(workerGoal, Math.min(8, Math.max(4, snapshot.enemyWorkersNearBase + 2)));
        }
        for (Unit base : myBases) {
            if (gs.getActionAssignment(base) != null) {
                continue;
            }
            boolean needWorker = myWorkers.size() + queuedWorkers < workerGoal || myWorkers.isEmpty();
            if (needWorker && resourcesLeft >= workerType.cost) {
                train(base, workerType);
                resourcesLeft -= workerType.cost;
                queuedWorkers++;
            }
        }

        UnitType trainingType = selectBarracksUnit(directive, snapshot);
        for (Unit barracks : myBarracks) {
            if (gs.getActionAssignment(barracks) != null) {
                continue;
            }
            if (trainingType != null && resourcesLeft >= trainingType.cost) {
                train(barracks, trainingType);
                resourcesLeft -= trainingType.cost;
            }
        }

        boolean earlyBarracksSnipe = shouldDoEarlyBarracksSnipe(gs, snapshot, myWorkers.size());
        boolean forcedAllOutRush = !workerRushDefense && !antiWorkerRushAllIn && shouldForceAllOutRush(gs, snapshot);
        boolean allowNoHomeReserve =
                (canAllWorkersCommitForFinisher(snapshot) || forcedAllOutRush || antiWorkerRushAllIn)
                        && !earlyBarracksSnipe
                        && !workerRushDefense;
        Unit homeReserveWorker = null;
        if (!allowNoHomeReserve && !freeWorkers.isEmpty()) {
            homeReserveWorker = selectHomeReserveWorker(freeWorkers, snapshot.myBases, player, pgs);
            if (homeReserveWorker != null) {
                freeWorkers.remove(homeReserveWorker);
            }
        }

        if ((forcedAllOutRush || antiWorkerRushAllIn) && !freeWorkers.isEmpty()) {
            for (Unit worker : freeWorkers) {
                Unit target = chooseAttackTarget(worker, player, snapshot.enemyUnits, directive, false);
                if (target != null) {
                    attack(worker, target);
                } else {
                    issueMovementFallback(worker, snapshot, player, pgs, anchorBase, true);
                }
            }
        } else if (earlyBarracksSnipe && !freeWorkers.isEmpty()) {
            for (Unit worker : freeWorkers) {
                Unit target = chooseBarracksFirstTarget(worker, snapshot.enemyUnits);
                if (target != null) {
                    attack(worker, target);
                }
            }
        } else {
            int availableWorkers = freeWorkers.size();
            int defensivePull = snapshot.enemyThreatNearBase > 0 ? Math.min(2, Math.max(0, availableWorkers - 1)) : 0;
            int raidWorkers = clampInt(directive.raidWorkers, 0, availableWorkers);
            boolean largeMapHeavyPressure = isLargeMap(snapshot)
                    && snapshot.enemyHeavyCount > 0
                    && snapshot.enemyClosestToBase <= 10;
            if (directive.mode == StrategyMode.WORKER_SWARM) {
                raidWorkers = Math.max(raidWorkers, Math.max(0, availableWorkers - 2));
            }
            if (directive.mode == StrategyMode.FORTRESS_DEFENSE) {
                raidWorkers = 0;
                defensivePull = Math.max(defensivePull, Math.min(3, availableWorkers));
            }
            if (workerRushDefense) {
                raidWorkers = 0;
                int requiredPull = Math.min(
                        Math.max(0, availableWorkers - 1),
                        Math.max(1, snapshot.enemyWorkersNearBase + 1));
                defensivePull = Math.max(defensivePull, requiredPull);
                if (snapshot.enemyWorkersNearBase >= 2
                        && snapshot.myWorkerCount <= snapshot.enemyWorkersNearBase + 1) {
                    defensivePull = availableWorkers;
                }
            }
            if (largeMapHeavyPressure) {
                defensivePull = Math.max(defensivePull, Math.min(3, availableWorkers));
                raidWorkers = 0;
            }

            int minimumHarvesters = workerRushDefense ? 0 : 1;
            if (directive.mode == StrategyMode.BOOM_THEN_PUSH) {
                int floor = gs.getTime() < BOOM_EARLY_TICK
                        ? Math.max(BOOM_MIN_HARVESTERS_EARLY, workerGoal - 1)
                        : Math.max(BOOM_MIN_HARVESTERS_LATE, workerGoal - 2);
                minimumHarvesters = clampInt(floor, 1, Math.max(1, availableWorkers));
            }

            int raidCap = Math.max(0, availableWorkers - defensivePull - minimumHarvesters);
            raidWorkers = Math.min(raidWorkers, raidCap);

            // BOOM_THEN_PUSH should preserve worker economy to sustain production.
            if (directive.mode == StrategyMode.BOOM_THEN_PUSH
                    && (gs.getTime() < BOOM_EARLY_TICK
                            || myCombat.size() < Math.max(2, directive.attackWaveSize / 2))) {
                raidWorkers = 0;
            }

            int harvesters = Math.max(minimumHarvesters, availableWorkers - raidWorkers - defensivePull);
            if (availableWorkers == 0) {
                harvesters = 0;
            } else {
                int harvestFloor = minimumHarvesters > 0 ? 1 : 0;
                harvesters = clampInt(harvesters, harvestFloor, availableWorkers);
            }
            Collections.sort(freeWorkers, Comparator.comparingInt(w -> distanceToClosestEnemy(w, player, pgs)));

            List<Unit> workerDefenders = new ArrayList<>();
            for (int i = 0; i < defensivePull && !freeWorkers.isEmpty(); i++) {
                workerDefenders.add(freeWorkers.remove(0));
            }

            List<Unit> workerRaiders = new ArrayList<>();
            while (workerRaiders.size() < raidWorkers && !freeWorkers.isEmpty()) {
                workerRaiders.add(freeWorkers.remove(freeWorkers.size() - 1));
            }

            int assignedHarvest = 0;
            for (Unit worker : freeWorkers) {
                if (assignedHarvest < harvesters) {
                    assignHarvest(worker, player, pgs);
                    assignedHarvest++;
                } else {
                    workerRaiders.add(worker);
                }
            }

            for (Unit defender : workerDefenders) {
                Unit threat = workerRushDefense
                        ? chooseWorkerRushDefenderTarget(defender, snapshot)
                        : closestThreatNearBases(
                                defender,
                                snapshot.myBases,
                                snapshot.enemyUnits,
                                (largeMapHeavyPressure ? Math.max(10, directive.defendRadius + 2) : directive.defendRadius + 1));
                if (threat != null) {
                    attack(defender, threat);
                } else {
                    assignHarvest(defender, player, pgs);
                }
            }

            for (Unit raider : workerRaiders) {
                Unit target = chooseAttackTarget(raider, player, snapshot.enemyUnits, directive, true);
                if (target != null) {
                    attack(raider, target);
                }
            }
        }

        if (homeReserveWorker != null) {
            assignHomeReserveTask(homeReserveWorker, player, pgs, snapshot);
        }

        boolean launchWave = shouldLaunchWave(gs, snapshot, directive, myCombat.size());
        if (launchWave) {
            lastWaveTick = gs.getTime();
        }

        for (Unit fighter : myCombat) {
            Unit target;
            if (launchWave || directive.mode == StrategyMode.ALL_IN_PUSH) {
                target = chooseAttackTarget(fighter, player, snapshot.enemyUnits, directive, false);
            } else {
                target = closestThreatNearBases(fighter, snapshot.myBases, snapshot.enemyUnits, directive.defendRadius + 1);
                if (target == null) {
                    if (directive.mode == StrategyMode.RANGED_CONTAIN || directive.mode == StrategyMode.SCOUT_AND_RAID) {
                        target = chooseAttackTarget(fighter, player, snapshot.enemyUnits, directive, false);
                    } else if (anchorBase != null) {
                        int[] rally = defaultRallyPoint(anchorBase, player, pgs, directive.defendRadius);
                        if (rally != null && manhattan(fighter.getX(), fighter.getY(), rally[0], rally[1]) > 1) {
                            move(fighter, rally[0], rally[1]);
                            continue;
                        }
                    }
                }
            }
            if (target != null) {
                attack(fighter, target);
            }
        }

        // Guardrail: leave no unit without a purposeful assignment.
        ensureActiveAssignments(player, gs, snapshot, directive, anchorBase, trainingType);
    }

    private boolean shouldLaunchWave(GameState gs, StateSnapshot snapshot, StrategyDirective directive, int myCombat) {
        if (shouldActivateWorkerRushDefense(gs, snapshot)) {
            return false;
        }
        if (isLargeMap(snapshot) && gs.getTime() >= 700 && myCombat >= 1 && snapshot.enemyCombatCount == 0) {
            return true;
        }
        if (snapshot.enemyBaseCount == 0 && snapshot.enemyCombatCount == 0) {
            return true;
        }
        int minimumWave = Math.max(1, directive.attackWaveSize);
        boolean enoughArmy = myCombat >= minimumWave;
        boolean waveTimerReady = gs.getTime() - lastWaveTick >= Math.max(35, directive.wavePeriod);
        boolean emergency = directive.mode == StrategyMode.ALL_IN_PUSH;
        boolean punishGreed = snapshot.enemyCombatCount <= 1 && myCombat >= Math.max(2, minimumWave - 1);
        boolean forceAdvantagePush = hasLargeAdvantage(snapshot);

        return emergency || (enoughArmy && waveTimerReady) || punishGreed || forceAdvantagePush;
    }

    private UnitType selectBarracksUnit(StrategyDirective directive, StateSnapshot snapshot) {
        String requested = directive.primaryUnit == null ? "RANGED" : directive.primaryUnit.toUpperCase();

        if (snapshot.enemyWorkersNearBase >= WORKER_RUSH_TRIGGER_NEAR_BASE && snapshot.enemyCombatCount <= 1) {
            requested = lightType != null ? "LIGHT" : "RANGED";
        } else if (directive.mode == StrategyMode.WORKER_SWARM) {
            requested = "LIGHT";
        } else if (directive.mode == StrategyMode.HEAVY_BREAKTHROUGH) {
            requested = "HEAVY";
        } else if (isLargeMap(snapshot) && snapshot.enemyHeavyCount >= 1 && snapshot.enemyCombatCount >= snapshot.myCombatCount) {
            requested = "RANGED";
        } else if (snapshot.enemyRangedCount > snapshot.enemyHeavyCount + 1) {
            requested = "HEAVY";
        } else if (snapshot.enemyHeavyCount > snapshot.enemyRangedCount + 1) {
            requested = "RANGED";
        }

        if ("HEAVY".equals(requested) && heavyType != null) {
            return heavyType;
        }
        if ("LIGHT".equals(requested) && lightType != null) {
            return lightType;
        }
        if (rangedType != null) {
            return rangedType;
        }
        if (lightType != null) {
            return lightType;
        }
        return heavyType;
    }

    private StrategyDirective askOllama(int player, GameState gs, StateSnapshot snapshot, int consultInterval) {
        try {
            String prompt = buildPrompt(player, gs, snapshot, consultInterval);
            String response = callOllama(prompt);
            StrategyDirective parsed = parseDirective(response);
            if (parsed == null) {
                return null;
            }
            return sanitizeDirective(parsed, snapshot);
        } catch (Exception e) {
            debug("T=%d LLM call failed: %s", gs.getTime(), e.getMessage());
            return null;
        }
    }

    private StrategyDirective sanitizeDirective(StrategyDirective in, StateSnapshot snapshot) {
        StrategyDirective out = StrategyDirective.defaults();
        if (in.mode != null) {
            out.mode = in.mode;
        }
        out.holdTicks = clampInt(in.holdTicks, MIN_HOLD_TICKS, 240);
        out.targetWorkers = clampInt(in.targetWorkers, 1, 12);
        out.targetBarracks = clampInt(in.targetBarracks, 0, 3);
        out.attackWaveSize = clampInt(in.attackWaveSize, 1, 12);
        out.wavePeriod = clampInt(in.wavePeriod, 35, 280);
        out.defendRadius = clampInt(in.defendRadius, 3, 12);
        out.aggression = clampDouble(in.aggression, 0.0, 1.0);
        out.raidWorkers = clampInt(in.raidWorkers, 0, 6);
        out.expandBase = in.expandBase;
        out.primaryUnit = normalizeUnitName(in.primaryUnit);
        out.rationale = in.rationale == null ? "llm" : in.rationale;

        if (snapshot.myWorkerCount <= 1) {
            out.targetWorkers = Math.max(out.targetWorkers, 3);
            out.raidWorkers = 0;
        }
        if (snapshot.myBarracksCount == 0 && out.targetBarracks == 0) {
            out.targetBarracks = 1;
        }

        return out;
    }

    private StrategyDirective fallbackDirective(StateSnapshot snapshot, GameState gs) {
        StrategyDirective d = StrategyDirective.defaults();
        boolean largeMap = isLargeMap(snapshot);

        if (snapshot.enemyWorkersNearBase >= WORKER_RUSH_TRIGGER_NEAR_BASE
                || snapshot.enemyThreatNearBase >= 2
                || snapshot.enemyCombatCount > snapshot.myCombatCount + 1) {
            d.mode = StrategyMode.FORTRESS_DEFENSE;
            d.targetWorkers = largeMap ? 5 : 4;
            d.targetBarracks = largeMap ? 2 : 1;
            d.attackWaveSize = largeMap ? 3 : 5;
            d.wavePeriod = largeMap ? 90 : 140;
            d.defendRadius = 7;
            d.aggression = largeMap ? 0.45 : 0.25;
            d.raidWorkers = 0;
            d.primaryUnit = snapshot.enemyHeavyCount > snapshot.enemyRangedCount ? "RANGED" : "HEAVY";
            d.rationale = largeMap ? "fallback-defense-large-map" : "fallback-defense";
            return d;
        }

        if (snapshot.myCombatCount >= snapshot.enemyCombatCount + 3 && snapshot.myBarracksCount > 0) {
            d.mode = StrategyMode.ALL_IN_PUSH;
            d.targetWorkers = 4;
            d.targetBarracks = Math.max(1, snapshot.myBarracksCount);
            d.attackWaveSize = 2;
            d.wavePeriod = 50;
            d.defendRadius = 4;
            d.aggression = 0.9;
            d.raidWorkers = 1;
            d.primaryUnit = "RANGED";
            d.rationale = "fallback-finish";
            return d;
        }

        if (gs.getTime() < 450) {
            d.mode = StrategyMode.BOOM_THEN_PUSH;
            d.targetWorkers = largeMap ? (snapshot.myBarracksCount == 0 ? 5 : 7) : 6;
            d.targetBarracks = largeMap ? 2 : 1;
            d.attackWaveSize = largeMap ? 5 : 4;
            d.wavePeriod = largeMap ? 85 : 120;
            d.defendRadius = largeMap ? 7 : 6;
            d.aggression = largeMap ? 0.62 : 0.55;
            d.raidWorkers = largeMap ? 1 : 0;
            d.expandBase = snapshot.myBaseCount < 2 && snapshot.myWorkerCount >= (largeMap ? 5 : 6);
            d.primaryUnit = "RANGED";
            d.rationale = largeMap ? "fallback-boom-large-map" : "fallback-boom";
            return d;
        }

        if (largeMap) {
            d.mode = StrategyMode.SCOUT_AND_RAID;
            d.targetWorkers = snapshot.myBarracksCount == 0 ? 5 : 7;
            d.targetBarracks = snapshot.myBarracksCount == 0 ? 1 : 2;
            d.attackWaveSize = 4;
            d.wavePeriod = 75;
            d.defendRadius = 7;
            d.aggression = 0.78;
            d.raidWorkers = 2;
            d.expandBase = snapshot.myBaseCount < 2 && snapshot.myWorkerCount >= 6;
            d.primaryUnit = snapshot.enemyRangedCount > snapshot.enemyHeavyCount ? "HEAVY" : "RANGED";
            d.rationale = "fallback-large-map-raid";
            return d;
        }

        d.mode = StrategyMode.RANGED_CONTAIN;
        d.targetWorkers = 5;
        d.targetBarracks = 2;
        d.attackWaveSize = 5;
        d.wavePeriod = 95;
        d.defendRadius = 6;
        d.aggression = 0.7;
        d.raidWorkers = snapshot.enemyWorkerCount >= 4 ? 1 : 0;
        d.primaryUnit = snapshot.enemyRangedCount > snapshot.enemyHeavyCount ? "HEAVY" : "RANGED";
        d.rationale = "fallback-contain";
        return d;
    }

    private String buildPrompt(int player, GameState gs, StateSnapshot s, int consultInterval) {
        StringBuilder sb = new StringBuilder();
        sb.append("You are setting GRAND STRATEGY for a MicroRTS bot. Return JSON only.\\n");
        sb.append("Your output is a HIGH-LEVEL strategy, not unit-by-unit commands.\\n");
        sb.append("Choose one strategy mode and numeric knobs for the next ")
                .append(consultInterval)
                .append(" ticks.\\n");
        sb.append("\\n");
        sb.append("State summary:\\n");
        sb.append("- time: ").append(gs.getTime()).append("\\n");
        sb.append("- map: ").append(s.mapWidth).append("x").append(s.mapHeight).append("\\n");
        sb.append("- my_resources: ").append(s.myResources).append("\\n");
        sb.append("- enemy_resources: ").append(s.enemyResources).append("\\n");
        sb.append("- my_workers: ").append(s.myWorkerCount).append("\\n");
        sb.append("- my_bases: ").append(s.myBaseCount).append("\\n");
        sb.append("- my_barracks: ").append(s.myBarracksCount).append("\\n");
        sb.append("- my_light: ").append(s.myLightCount).append("\\n");
        sb.append("- my_ranged: ").append(s.myRangedCount).append("\\n");
        sb.append("- my_heavy: ").append(s.myHeavyCount).append("\\n");
        sb.append("- enemy_workers: ").append(s.enemyWorkerCount).append("\\n");
        sb.append("- enemy_bases: ").append(s.enemyBaseCount).append("\\n");
        sb.append("- enemy_barracks: ").append(s.enemyBarracksCount).append("\\n");
        sb.append("- enemy_light: ").append(s.enemyLightCount).append("\\n");
        sb.append("- enemy_ranged: ").append(s.enemyRangedCount).append("\\n");
        sb.append("- enemy_heavy: ").append(s.enemyHeavyCount).append("\\n");
        sb.append("- enemy_threat_near_base: ").append(s.enemyThreatNearBase).append("\\n");
        sb.append("- enemy_workers_near_base: ").append(s.enemyWorkersNearBase).append("\\n");
        sb.append("- enemy_closest_to_my_base: ").append(s.enemyClosestToBase).append("\\n");
        sb.append("\\n");
        sb.append("Allowed mode values:\\n");
        sb.append("- BOOM_THEN_PUSH: focus economy then timed attacks\\n");
        sb.append("- RANGED_CONTAIN: pressure map with ranged and deny expansion\\n");
        sb.append("- HEAVY_BREAKTHROUGH: fewer stronger frontliners\\n");
        sb.append("- WORKER_SWARM: pull workers for short aggressive timing\\n");
        sb.append("- SCOUT_AND_RAID: selective harassment on workers/barracks\\n");
        sb.append("- FORTRESS_DEFENSE: turtle and stabilize\\n");
        sb.append("- ALL_IN_PUSH: commit all army to finish\\n");
        sb.append("\\n");
        sb.append("JSON schema (all keys required):\\n");
        sb.append("{");
        sb.append("\\\"mode\\\":\\\"...\\\",");
        sb.append("\\\"hold_ticks\\\":75..240,");
        sb.append("\\\"target_workers\\\":1..12,");
        sb.append("\\\"target_barracks\\\":0..3,");
        sb.append("\\\"attack_wave_size\\\":1..12,");
        sb.append("\\\"wave_period\\\":35..280,");
        sb.append("\\\"defend_radius\\\":3..12,");
        sb.append("\\\"aggression\\\":0..1,");
        sb.append("\\\"raid_workers\\\":0..6,");
        sb.append("\\\"expand_base\\\":true|false,");
        sb.append("\\\"primary_unit\\\":\\\"RANGED|HEAVY|LIGHT\\\",");
        sb.append("\\\"rationale\\\":\\\"short text\\\"");
        sb.append("}\\n");

        String suggestedMode = s.enemyThreatNearBase >= 2 ? "FORTRESS_DEFENSE" : "RANGED_CONTAIN";
        sb.append("Example:\\n");
        sb.append("{");
        sb.append("\\\"mode\\\":\\\"").append(suggestedMode).append("\\\",");
        sb.append("\\\"hold_ticks\\\":120,");
        sb.append("\\\"target_workers\\\":5,");
        sb.append("\\\"target_barracks\\\":1,");
        sb.append("\\\"attack_wave_size\\\":4,");
        sb.append("\\\"wave_period\\\":100,");
        sb.append("\\\"defend_radius\\\":6,");
        sb.append("\\\"aggression\\\":0.6,");
        sb.append("\\\"raid_workers\\\":1,");
        sb.append("\\\"expand_base\\\":false,");
        sb.append("\\\"primary_unit\\\":\\\"RANGED\\\",");
        sb.append("\\\"rationale\\\":\\\"keep pressure while safe\\\"");
        sb.append("}");

        return sb.toString();
    }

    private String callOllama(String prompt) throws IOException {
        URL url = new URL(OLLAMA_HOST + "/api/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setConnectTimeout(CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(READ_TIMEOUT_MS);
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");

        JsonObject req = new JsonObject();
        req.addProperty("model", OLLAMA_MODEL);
        req.addProperty("prompt", prompt);
        req.addProperty("stream", false);

        try (OutputStream os = conn.getOutputStream()) {
            os.write(req.toString().getBytes(StandardCharsets.UTF_8));
        }

        int status = conn.getResponseCode();
        byte[] raw;
        if (status >= 200 && status < 300) {
            raw = conn.getInputStream().readAllBytes();
        } else {
            raw = conn.getErrorStream() != null ? conn.getErrorStream().readAllBytes() : new byte[0];
            throw new IOException("Ollama status " + status + ": " + new String(raw, StandardCharsets.UTF_8));
        }

        String envelope = new String(raw, StandardCharsets.UTF_8);
        JsonObject parsed = JsonParser.parseString(envelope).getAsJsonObject();
        if (parsed.has("response")) {
            return parsed.get("response").getAsString();
        }
        return envelope;
    }

    private StrategyDirective parseDirective(String raw) {
        JsonObject obj = parseJsonObject(raw);
        if (obj == null) {
            return null;
        }

        StrategyDirective d = StrategyDirective.defaults();
        d.mode = parseMode(readString(obj, "mode", d.mode.name()));
        d.holdTicks = readInt(obj, "hold_ticks", d.holdTicks);
        d.targetWorkers = readInt(obj, "target_workers", d.targetWorkers);
        d.targetBarracks = readInt(obj, "target_barracks", d.targetBarracks);
        d.attackWaveSize = readInt(obj, "attack_wave_size", d.attackWaveSize);
        d.wavePeriod = readInt(obj, "wave_period", d.wavePeriod);
        d.defendRadius = readInt(obj, "defend_radius", d.defendRadius);
        d.aggression = readDouble(obj, "aggression", d.aggression);
        d.raidWorkers = readInt(obj, "raid_workers", d.raidWorkers);
        d.expandBase = readBoolean(obj, "expand_base", d.expandBase);
        d.primaryUnit = normalizeUnitName(readString(obj, "primary_unit", d.primaryUnit));
        d.rationale = readString(obj, "rationale", "llm");

        return d;
    }

    private JsonObject parseJsonObject(String raw) {
        String text = raw == null ? "" : raw.trim();
        if (text.isEmpty()) {
            return null;
        }

        try {
            JsonElement parsed = JsonParser.parseString(text);
            if (parsed.isJsonObject()) {
                return parsed.getAsJsonObject();
            }
        } catch (Exception ignored) {
        }

        Matcher m = JSON_OBJECT.matcher(text);
        if (!m.find()) {
            return null;
        }

        try {
            JsonElement parsed = JsonParser.parseString(m.group());
            if (parsed.isJsonObject()) {
                return parsed.getAsJsonObject();
            }
        } catch (Exception ignored) {
        }

        return null;
    }

    private StateSnapshot inspectState(int player, GameState gs) {
        StateSnapshot s = new StateSnapshot();
        PhysicalGameState pgs = gs.getPhysicalGameState();
        s.mapWidth = pgs.getWidth();
        s.mapHeight = pgs.getHeight();

        int enemyPlayer = findEnemyPlayer(player, gs);
        Player me = gs.getPlayer(player);
        Player enemy = enemyPlayer >= 0 ? gs.getPlayer(enemyPlayer) : null;
        s.myResources = me != null ? me.getResources() : 0;
        s.enemyResources = enemy != null ? enemy.getResources() : 0;
        s.enemyClosestToBase = Integer.MAX_VALUE;
        s.enemyClosestWorkerToBase = Integer.MAX_VALUE;

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                s.resourceNodeCount++;
            }
            if (u.getPlayer() == player) {
                s.myUnits.add(u);
                if (u.getType() == workerType) {
                    s.myWorkerCount++;
                } else if (u.getType() == baseType) {
                    s.myBaseCount++;
                    s.myBases.add(u);
                } else if (u.getType() == barracksType) {
                    s.myBarracksCount++;
                }
                if (u.getType() == lightType) {
                    s.myLightCount++;
                }
                if (u.getType() == rangedType) {
                    s.myRangedCount++;
                }
                if (u.getType() == heavyType) {
                    s.myHeavyCount++;
                }
                if (u.getType().canAttack && u.getType() != workerType) {
                    s.myCombatCount++;
                }
            } else if (u.getPlayer() >= 0) {
                s.enemyUnits.add(u);
                if (u.getType() == workerType) {
                    s.enemyWorkerCount++;
                } else if (u.getType() == baseType) {
                    s.enemyBaseCount++;
                    s.enemyBases.add(u);
                } else if (u.getType() == barracksType) {
                    s.enemyBarracksCount++;
                }
                if (u.getType() == lightType) {
                    s.enemyLightCount++;
                }
                if (u.getType() == rangedType) {
                    s.enemyRangedCount++;
                }
                if (u.getType() == heavyType) {
                    s.enemyHeavyCount++;
                }
                if (u.getType().canAttack && u.getType() != workerType) {
                    s.enemyCombatCount++;
                }
            }
        }

        for (Unit enemyUnit : s.enemyUnits) {
            int d = distanceToClosest(enemyUnit, s.myBases);
            if (d <= THREAT_RADIUS && (enemyUnit.getType().canAttack || enemyUnit.getType() == workerType)) {
                s.enemyThreatNearBase++;
            }
            if (enemyUnit.getType() == workerType) {
                if (d <= THREAT_RADIUS) {
                    s.enemyWorkersNearBase++;
                }
                if (d < s.enemyClosestWorkerToBase) {
                    s.enemyClosestWorkerToBase = d;
                }
            }
            if (d < s.enemyClosestToBase) {
                s.enemyClosestToBase = d;
            }
        }
        if (s.enemyClosestToBase == Integer.MAX_VALUE) {
            s.enemyClosestToBase = 999;
        }
        if (s.enemyClosestWorkerToBase == Integer.MAX_VALUE) {
            s.enemyClosestWorkerToBase = 999;
        }

        s.enemyCollapsed = s.enemyBaseCount == 0
                || (s.enemyCombatCount == 0 && s.enemyWorkerCount <= 1 && s.enemyResources == 0);
        return s;
    }

    private int findEnemyPlayer(int player, GameState gs) {
        int tentative = player == 0 ? 1 : 0;
        if (gs.getPlayer(tentative) != null) {
            return tentative;
        }
        for (int i = 0; i < 4; i++) {
            if (i != player && gs.getPlayer(i) != null) {
                return i;
            }
        }
        return -1;
    }

    private boolean assignHarvest(Unit worker, int player, PhysicalGameState pgs) {
        Unit closestResource = closestResource(worker, pgs);
        Unit closestBase = closestStockpile(worker, player, pgs);
        if (closestResource == null || closestBase == null) {
            return false;
        }

        AbstractAction aa = getAbstractAction(worker);
        if (aa instanceof Harvest) {
            Harvest h = (Harvest) aa;
            if (h.getTarget() == closestResource && h.getBase() == closestBase) {
                return true;
            }
        }
        harvest(worker, closestResource, closestBase);
        return true;
    }

    private void assignHomeReserveTask(Unit worker, int player, PhysicalGameState pgs, StateSnapshot snapshot) {
        Unit threat = closestThreatNearBases(
                worker,
                snapshot.myBases,
                snapshot.enemyUnits,
                HOME_RESERVE_THREAT_RADIUS);
        if (threat != null
                && distanceToClosest(threat, snapshot.myBases) <= HOME_RESERVE_THREAT_RADIUS
                && manhattan(worker, threat) <= HOME_RESERVE_MAX_CHASE_DISTANCE) {
            attack(worker, threat);
            return;
        }

        if (assignSafeHarvest(worker, player, pgs, snapshot.myBases, snapshot.enemyBases)) {
            return;
        }

        if (!issueMovementFallback(worker, snapshot, player, pgs, null, false)) {
            Unit fallbackThreat = chooseAttackTarget(worker, player, snapshot.enemyUnits, currentDirective, true);
            if (fallbackThreat != null) {
                attack(worker, fallbackThreat);
            }
        }
    }

    private boolean shouldDoEarlyBarracksSnipe(GameState gs, StateSnapshot snapshot, int myWorkers) {
        return gs.getTime() <= EARLY_BARRACKS_SNIPE_END_TICK
                && myWorkers >= 2
                && snapshot.enemyBarracksCount > 0
                && snapshot.enemyWorkerCount < 2
                && snapshot.enemyResources < 2;
    }

    private boolean shouldForceAllOutRush(GameState gs, StateSnapshot snapshot) {
        if (snapshot == null) {
            return false;
        }
        if (shouldActivateWorkerRushDefense(gs, snapshot)) {
            return false;
        }
        if (isLargeMap(snapshot) && gs.getTime() < 850) {
            return false;
        }
        if (snapshot.enemyCombatCount > 0) {
            return false;
        }
        if (snapshot.enemyBarracksCount > 0 && gs.getTime() < 1000) {
            return false;
        }
        if (snapshot.enemyBaseCount > 0 && snapshot.enemyWorkerCount >= 3) {
            return false;
        }

        int myPressureCount = snapshot.myWorkerCount + snapshot.myCombatCount;
        int enemyEconomicCount = snapshot.enemyWorkerCount + snapshot.enemyResources;
        return myPressureCount >= 2 * enemyEconomicCount;
    }

    private boolean shouldActivateWorkerRushDefense(GameState gs, StateSnapshot snapshot) {
        if (snapshot == null || snapshot.myBaseCount <= 0) {
            return false;
        }

        boolean immediateThreat = snapshot.enemyWorkersNearBase >= WORKER_RUSH_TRIGGER_NEAR_BASE;
        boolean earlyPressure = gs.getTime() <= WORKER_RUSH_DEFENSE_END_TICK
                && snapshot.enemyWorkersNearBase >= 1
                && snapshot.enemyClosestWorkerToBase <= 4;
        boolean lowCombatPressure = isSmallMap(snapshot)
                && gs.getTime() <= 450
                && snapshot.enemyCombatCount <= 1
                && snapshot.enemyWorkersNearBase >= 2
                && snapshot.enemyClosestWorkerToBase <= 6;
        boolean proactiveSmallMapScout = gs.getTime() <= 260
                && isSmallMap(snapshot)
                && snapshot.enemyWorkerCount >= 4
                && snapshot.enemyClosestWorkerToBase <= THREAT_RADIUS + 2;
        return immediateThreat || earlyPressure || lowCombatPressure || proactiveSmallMapScout;
    }

    private Unit chooseWorkerRushDefenderTarget(Unit defender, StateSnapshot snapshot) {
        Unit best = null;
        int bestScore = Integer.MAX_VALUE;
        int defendRadius = Math.max(5, currentDirective.defendRadius + 1);
        for (Unit enemy : snapshot.enemyUnits) {
            if (enemy.getType() != workerType) {
                continue;
            }
            int dBase = distanceToClosest(enemy, snapshot.myBases);
            if (dBase > defendRadius) {
                continue;
            }
            int d = manhattan(defender, enemy);
            int score = d * 2 + dBase;
            if (score < bestScore) {
                bestScore = score;
                best = enemy;
            }
        }
        if (best != null) {
            return best;
        }
        return closestThreatNearBases(defender, snapshot.myBases, snapshot.enemyUnits, defendRadius);
    }

    private Unit chooseBarracksFirstTarget(Unit attacker, List<Unit> enemyUnits) {
        Unit closestBarracks = null;
        int bestBarracksDist = Integer.MAX_VALUE;
        Unit closestWorker = null;
        int bestWorkerDist = Integer.MAX_VALUE;
        Unit closestBase = null;
        int bestBaseDist = Integer.MAX_VALUE;
        Unit closestAny = null;
        int bestAnyDist = Integer.MAX_VALUE;

        for (Unit enemy : enemyUnits) {
            int d = manhattan(attacker, enemy);
            if (d < bestAnyDist) {
                bestAnyDist = d;
                closestAny = enemy;
            }
            if (enemy.getType() == barracksType && d < bestBarracksDist) {
                bestBarracksDist = d;
                closestBarracks = enemy;
            }
            if (enemy.getType() == workerType && d < bestWorkerDist) {
                bestWorkerDist = d;
                closestWorker = enemy;
            }
            if (enemy.getType() == baseType && d < bestBaseDist) {
                bestBaseDist = d;
                closestBase = enemy;
            }
        }

        if (closestBarracks != null) {
            return closestBarracks;
        }
        if (closestWorker != null) {
            return closestWorker;
        }
        if (closestBase != null) {
            return closestBase;
        }
        return closestAny;
    }

    private boolean assignSafeHarvest(
            Unit worker,
            int player,
            PhysicalGameState pgs,
            List<Unit> myBases,
            List<Unit> enemyBases) {
        Unit closestBase = closestStockpile(worker, player, pgs);
        if (closestBase == null) {
            return false;
        }

        Unit resource = safestResourceNearBase(worker, myBases, enemyBases, pgs);
        if (resource == null) {
            resource = closestResource(worker, pgs);
        }
        if (resource == null) {
            return false;
        }

        AbstractAction aa = getAbstractAction(worker);
        if (aa instanceof Harvest) {
            Harvest h = (Harvest) aa;
            if (h.getTarget() == resource && h.getBase() == closestBase) {
                return true;
            }
        }
        harvest(worker, resource, closestBase);
        return true;
    }

    private void ensureActiveAssignments(
            int player,
            GameState gs,
            StateSnapshot snapshot,
            StrategyDirective directive,
            Unit anchorBase,
            UnitType trainingType) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player me = gs.getPlayer(player);
        if (me == null) {
            return;
        }

        boolean forceForward = hasLargeAdvantage(snapshot) || directive.mode == StrategyMode.ALL_IN_PUSH;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) {
                continue;
            }
            if (gs.getActionAssignment(u) != null) {
                continue;
            }
            if (getAbstractAction(u) != null) {
                continue;
            }

            if (u.getType() == baseType) {
                int workerCap = clampInt(directive.targetWorkers, 1, 10);
                boolean shouldBankForBarracks = snapshot.myBarracksCount < Math.max(1, directive.targetBarracks)
                        && me.getResources() < barracksType.cost;
                if (snapshot.myWorkerCount < workerCap
                        && !shouldBankForBarracks
                        && me.getResources() >= workerType.cost) {
                    train(u, workerType);
                }
                continue;
            }

            if (u.getType() == barracksType) {
                if (trainingType != null && me.getResources() >= trainingType.cost) {
                    train(u, trainingType);
                }
                continue;
            }

            if (u.getType() == workerType) {
                if (assignSafeHarvest(u, player, pgs, snapshot.myBases, snapshot.enemyBases)) {
                    continue;
                }
                Unit workerTarget = chooseAttackTarget(u, player, snapshot.enemyUnits, directive, true);
                if (workerTarget != null && forceForward) {
                    attack(u, workerTarget);
                    continue;
                }
                issueMovementFallback(u, snapshot, player, pgs, anchorBase, forceForward);
                continue;
            }

            if (u.getType().canAttack) {
                Unit target = chooseAttackTarget(u, player, snapshot.enemyUnits, directive, false);
                if (target != null) {
                    attack(u, target);
                } else {
                    issueMovementFallback(u, snapshot, player, pgs, anchorBase, forceForward);
                }
            }
        }
    }

    private boolean issueMovementFallback(
            Unit unit,
            StateSnapshot snapshot,
            int player,
            PhysicalGameState pgs,
            Unit anchorBase,
            boolean forceForward) {
        int tx = -1;
        int ty = -1;

        Unit enemyTarget = closestUnit(unit, snapshot.enemyUnits);
        if (enemyTarget != null && (forceForward || snapshot.enemyThreatNearBase > 0)) {
            tx = enemyTarget.getX();
            ty = enemyTarget.getY();
        } else if (forceForward && snapshot.enemyBases != null && !snapshot.enemyBases.isEmpty()) {
            Unit enemyBase = closestUnit(unit, snapshot.enemyBases);
            if (enemyBase != null) {
                tx = enemyBase.getX();
                ty = enemyBase.getY();
            }
        } else if (anchorBase != null) {
            int[] rally = defaultRallyPoint(anchorBase, player, pgs, Math.max(4, currentDirective.defendRadius));
            if (rally != null) {
                tx = rally[0];
                ty = rally[1];
            }
        }

        if (tx < 0 || ty < 0) {
            tx = clampInt(unit.getX() + (player == 0 ? 1 : -1), 0, pgs.getWidth() - 1);
            ty = unit.getY();
        }

        if (tx == unit.getX() && ty == unit.getY()) {
            ty = clampInt(unit.getY() + 1, 0, pgs.getHeight() - 1);
            if (tx == unit.getX() && ty == unit.getY()) {
                return false;
            }
        }

        move(unit, tx, ty);
        return true;
    }

    private Unit safestResourceNearBase(
            Unit worker,
            List<Unit> myBases,
            List<Unit> enemyBases,
            PhysicalGameState pgs) {
        if (myBases == null || myBases.isEmpty()) {
            return null;
        }

        Unit best = null;
        int bestScore = Integer.MAX_VALUE;

        for (Unit resource : pgs.getUnits()) {
            if (!resource.getType().isResource) {
                continue;
            }
            int myBaseDist = distanceToClosest(resource, myBases);
            int enemyDist = distanceToClosestCoords(resource.getX(), resource.getY(), enemyBases);
            if (enemyDist == Integer.MAX_VALUE) {
                enemyDist = 99;
            }

            // Skip resources clearly closer to the enemy side.
            if (enemyDist + 1 < myBaseDist) {
                continue;
            }

            int workerDist = manhattan(worker, resource);
            int score = myBaseDist * 8 + workerDist * 2 - enemyDist * 3;
            if (score < bestScore) {
                bestScore = score;
                best = resource;
            }
        }

        return best;
    }

    private Unit closestResource(Unit from, PhysicalGameState pgs) {
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit u : pgs.getUnits()) {
            if (!u.getType().isResource) {
                continue;
            }
            int d = manhattan(from, u);
            if (d < bestDist) {
                best = u;
                bestDist = d;
            }
        }
        return best;
    }

    private Unit closestStockpile(Unit from, int player, PhysicalGameState pgs) {
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && u.getType().isStockpile) {
                int d = manhattan(from, u);
                if (d < bestDist) {
                    best = u;
                    bestDist = d;
                }
            }
        }
        return best;
    }

    private Unit chooseBuilder(List<Unit> freeWorkers, Unit anchorBase) {
        if (freeWorkers.isEmpty()) {
            return null;
        }
        if (anchorBase == null) {
            return freeWorkers.get(0);
        }

        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit w : freeWorkers) {
            int d = manhattan(w, anchorBase);
            if (d < bestDist) {
                bestDist = d;
                best = w;
            }
        }
        return best;
    }

    private boolean canAllWorkersCommitForFinisher(StateSnapshot snapshot) {
        return snapshot.enemyCombatCount == 0
                && snapshot.enemyWorkerCount == 0
                && snapshot.enemyBaseCount > 0;
    }

    private Unit selectHomeReserveWorker(
            List<Unit> freeWorkers,
            List<Unit> myBases,
            int player,
            PhysicalGameState pgs) {
        if (freeWorkers.isEmpty()) {
            return null;
        }

        if (myBases != null && !myBases.isEmpty()) {
            Unit best = null;
            int bestDist = Integer.MAX_VALUE;
            for (Unit worker : freeWorkers) {
                int d = distanceToClosest(worker, myBases);
                if (d < bestDist) {
                    bestDist = d;
                    best = worker;
                }
            }
            return best;
        }

        Unit safest = null;
        int safestEnemyDist = Integer.MIN_VALUE;
        for (Unit worker : freeWorkers) {
            int d = distanceToClosestEnemy(worker, player, pgs);
            if (d > safestEnemyDist) {
                safestEnemyDist = d;
                safest = worker;
            }
        }
        return safest;
    }

    private Unit chooseExpansionWorker(List<Unit> freeWorkers, PhysicalGameState pgs, List<Unit> enemyBases) {
        if (freeWorkers.isEmpty()) {
            return null;
        }

        Unit best = null;
        int bestScore = Integer.MIN_VALUE;
        for (Unit w : freeWorkers) {
            int nearestEnemy = distanceToClosest(w, enemyBases);
            int nearestResource = Integer.MAX_VALUE;
            for (Unit u : pgs.getUnits()) {
                if (u.getType().isResource) {
                    nearestResource = Math.min(nearestResource, manhattan(w, u));
                }
            }
            if (nearestEnemy == Integer.MAX_VALUE) {
                nearestEnemy = 20;
            }
            if (nearestResource == Integer.MAX_VALUE) {
                nearestResource = 10;
            }
            int score = nearestEnemy - nearestResource;
            if (score > bestScore) {
                bestScore = score;
                best = w;
            }
        }

        return best;
    }

    private Unit closestUnit(Unit origin, List<Unit> candidates) {
        if (origin == null || candidates == null || candidates.isEmpty()) {
            return null;
        }
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit c : candidates) {
            int d = manhattan(origin, c);
            if (d < bestDist) {
                bestDist = d;
                best = c;
            }
        }
        return best;
    }

    private int[] findExpansionPosition(Unit worker, PhysicalGameState pgs, List<Unit> enemyBases) {
        Unit bestResource = null;
        int bestScore = Integer.MIN_VALUE;

        for (Unit u : pgs.getUnits()) {
            if (!u.getType().isResource) {
                continue;
            }
            int dWorker = manhattan(worker, u);
            int dEnemy = distanceToClosestCoords(u.getX(), u.getY(), enemyBases);
            if (dEnemy == Integer.MAX_VALUE) {
                dEnemy = 30;
            }
            int score = dEnemy - dWorker;
            if (score > bestScore) {
                bestScore = score;
                bestResource = u;
            }
        }

        if (bestResource == null) {
            return null;
        }

        return new int[] {bestResource.getX(), bestResource.getY()};
    }

    private boolean tryBuild(
            Unit worker,
            UnitType building,
            int desiredX,
            int desiredY,
            List<Integer> reserved,
            Player p,
            PhysicalGameState pgs) {
        if (worker == null || building == null) {
            return false;
        }
        if (getAbstractAction(worker) != null && getAbstractAction(worker).getClass().getSimpleName().equals("Build")) {
            return false;
        }

        int pos = findBuildingPosition(reserved, desiredX, desiredY, p, pgs);
        if (pos < 0) {
            return false;
        }
        build(worker, building, pos % pgs.getWidth(), pos / pgs.getWidth());
        reserved.add(pos);
        return true;
    }

    private Unit closestThreatNearBases(Unit self, List<Unit> myBases, List<Unit> enemyUnits, int radius) {
        Unit closest = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit enemy : enemyUnits) {
            if (!(enemy.getType().canAttack || enemy.getType().canHarvest)) {
                continue;
            }
            int dBase = distanceToClosest(enemy, myBases);
            if (dBase > radius) {
                continue;
            }
            int d = manhattan(self, enemy);
            if (d < bestDist) {
                bestDist = d;
                closest = enemy;
            }
        }
        return closest;
    }

    private Unit chooseAttackTarget(
            Unit attacker,
            int player,
            List<Unit> enemyUnits,
            StrategyDirective directive,
            boolean harassOnly) {
        Unit best = null;
        int bestScore = Integer.MAX_VALUE;

        for (Unit enemy : enemyUnits) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player) {
                continue;
            }

            int d = manhattan(attacker, enemy);
            int priority = basePriority(enemy, directive.mode, harassOnly);
            int score = d * 2 + priority;

            if (score < bestScore) {
                bestScore = score;
                best = enemy;
            }
        }
        return best;
    }

    private int basePriority(Unit enemy, StrategyMode mode, boolean harassOnly) {
        if (enemy.getType().isResource) {
            return 500;
        }

        int value;
        if (enemy.getType() == baseType) {
            value = 8;
        } else if (enemy.getType() == barracksType) {
            value = 6;
        } else if (enemy.getType().canAttack) {
            value = 3;
        } else if (enemy.getType() == workerType) {
            value = 2;
        } else {
            value = 10;
        }

        if (harassOnly) {
            if (enemy.getType() == workerType) {
                value = 1;
            } else if (enemy.getType() == barracksType) {
                value = 3;
            } else if (enemy.getType() == baseType) {
                value = 7;
            }
        }

        switch (mode) {
            case FORTRESS_DEFENSE:
                if (enemy.getType().canAttack) {
                    value -= 1;
                }
                break;
            case ALL_IN_PUSH:
            case HEAVY_BREAKTHROUGH:
                if (enemy.getType() == baseType || enemy.getType() == barracksType) {
                    value -= 3;
                }
                if (mode == StrategyMode.ALL_IN_PUSH) {
                    if (enemy.getType() == baseType) {
                        value = Math.min(value, 0);
                    } else if (enemy.getType() == barracksType) {
                        value = Math.min(value, 1);
                    } else if (enemy.getType() == workerType) {
                        value += 2;
                    }
                }
                break;
            case SCOUT_AND_RAID:
                if (enemy.getType() == workerType || enemy.getType() == barracksType) {
                    value -= 2;
                }
                break;
            case RANGED_CONTAIN:
                if (enemy.getType().canAttack) {
                    value -= 2;
                }
                break;
            case WORKER_SWARM:
                if (enemy.getType() == workerType) {
                    value -= 3;
                }
                break;
            case BOOM_THEN_PUSH:
            default:
                break;
        }

        return value;
    }

    private int[] defaultRallyPoint(Unit base, int player, PhysicalGameState pgs, int defendRadius) {
        if (base == null) {
            return null;
        }

        int x = base.getX();
        int y = base.getY();
        int offset = Math.max(1, defendRadius / 2);
        int targetX = clampInt(x + (player == 0 ? offset : -offset), 0, pgs.getWidth() - 1);
        int targetY = clampInt(y, 0, pgs.getHeight() - 1);
        return new int[] {targetX, targetY};
    }

    private int distanceToClosest(Unit origin, List<Unit> candidates) {
        if (origin == null || candidates == null || candidates.isEmpty()) {
            return Integer.MAX_VALUE;
        }
        int best = Integer.MAX_VALUE;
        for (Unit c : candidates) {
            int d = manhattan(origin, c);
            if (d < best) {
                best = d;
            }
        }
        return best;
    }

    private int distanceToClosestCoords(int x, int y, List<Unit> candidates) {
        if (candidates == null || candidates.isEmpty()) {
            return Integer.MAX_VALUE;
        }
        int best = Integer.MAX_VALUE;
        for (Unit c : candidates) {
            int d = manhattan(x, y, c.getX(), c.getY());
            if (d < best) {
                best = d;
            }
        }
        return best;
    }

    private int distanceToClosestEnemy(Unit from, int player, PhysicalGameState pgs) {
        int best = Integer.MAX_VALUE;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player) {
                best = Math.min(best, manhattan(from, u));
            }
        }
        return best == Integer.MAX_VALUE ? 999 : best;
    }

    private boolean hasLargeAdvantage(StateSnapshot snapshot) {
        if (snapshot == null) {
            return false;
        }
        if (snapshot.myCombatCount >= snapshot.enemyCombatCount + LARGE_ADVANTAGE_COMBAT_DELTA) {
            return true;
        }
        return snapshot.enemyCombatCount == 0
                && snapshot.myCombatCount >= 2
                && snapshot.myWorkerCount >= snapshot.enemyWorkerCount + 2;
    }

    private boolean isSmallMap(StateSnapshot snapshot) {
        return snapshot != null && snapshot.mapWidth * snapshot.mapHeight <= SMALL_MAP_AREA;
    }

    private boolean isLargeMap(StateSnapshot snapshot) {
        return snapshot != null && snapshot.mapWidth * snapshot.mapHeight >= LARGE_MAP_AREA;
    }

    private int consultIntervalFor(StateSnapshot snapshot) {
        int base = Math.max(1, CONSULT_INTERVAL);
        if (isLargeMap(snapshot)) {
            return base + 40;
        }
        return base;
    }

    private int manhattan(Unit a, Unit b) {
        return manhattan(a.getX(), a.getY(), b.getX(), b.getY());
    }

    private int manhattan(int ax, int ay, int bx, int by) {
        return Math.abs(ax - bx) + Math.abs(ay - by);
    }

    private StrategyMode parseMode(String raw) {
        if (raw == null) {
            return StrategyMode.BOOM_THEN_PUSH;
        }
        String k = raw.trim().toUpperCase();
        for (StrategyMode mode : StrategyMode.values()) {
            if (mode.name().equals(k)) {
                return mode;
            }
        }
        return StrategyMode.BOOM_THEN_PUSH;
    }

    private String normalizeUnitName(String s) {
        if (s == null) {
            return "RANGED";
        }
        String up = s.trim().toUpperCase();
        if ("HEAVY".equals(up) || "RANGED".equals(up) || "LIGHT".equals(up)) {
            return up;
        }
        return "RANGED";
    }

    private static int readInt(JsonObject obj, String key, int fallback) {
        if (obj == null || !obj.has(key)) {
            return fallback;
        }
        try {
            return obj.get(key).getAsInt();
        } catch (Exception e) {
            return fallback;
        }
    }

    private static double readDouble(JsonObject obj, String key, double fallback) {
        if (obj == null || !obj.has(key)) {
            return fallback;
        }
        try {
            return obj.get(key).getAsDouble();
        } catch (Exception e) {
            return fallback;
        }
    }

    private static boolean readBoolean(JsonObject obj, String key, boolean fallback) {
        if (obj == null || !obj.has(key)) {
            return fallback;
        }
        try {
            return obj.get(key).getAsBoolean();
        } catch (Exception e) {
            return fallback;
        }
    }

    private static String readString(JsonObject obj, String key, String fallback) {
        if (obj == null || !obj.has(key)) {
            return fallback;
        }
        try {
            return obj.get(key).getAsString();
        } catch (Exception e) {
            return fallback;
        }
    }

    private static int getEnvInt(String key, int fallback) {
        String raw = System.getenv(key);
        if (raw == null || raw.isEmpty()) {
            return fallback;
        }
        try {
            return Integer.parseInt(raw.trim());
        } catch (Exception e) {
            return fallback;
        }
    }

    private static int clampInt(int value, int min, int max) {
        if (value < min) {
            return min;
        }
        if (value > max) {
            return max;
        }
        return value;
    }

    private static double clampDouble(double value, double min, double max) {
        if (value < min) {
            return min;
        }
        if (value > max) {
            return max;
        }
        return value;
    }

    private void debug(String format, Object... args) {
        if (!DEBUG) {
            return;
        }
        System.out.println("[OracleStrategy] " + String.format(format, args));
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}
