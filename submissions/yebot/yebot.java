/*
 * yebot — Worker Rush (no LLM blocking)
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
import rts.*;
import rts.units.*;

import java.util.*;

public class yebot extends AbstractionLayerAI {

    private UnitTypeTable utt;
    private UnitType workerType, baseType;

    public yebot(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    public yebot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        utt        = a_utt;
        workerType = a_utt.getUnitType("Worker");
        baseType   = a_utt.getUnitType("Base");
    }

    @Override public void reset() { super.reset(); }
    @Override public AI clone()   { return new yebot(utt, pf); }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();

        List<Unit> myWorkers = new ArrayList<>();
        List<Unit> enemies   = new ArrayList<>();
        List<Unit> resources = new ArrayList<>();
        Unit myBase = null;

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) { resources.add(u); continue; }
            if (u.getPlayer() == player) {
                if (u.getType() == baseType)   myBase = u;
                if (u.getType() == workerType) myWorkers.add(u);
            } else {
                enemies.add(u);
            }
        }

        // Base: train worker every time resources allow
        if (myBase != null && gs.getActionAssignment(myBase) == null
                && gs.getPlayer(player).getResources() >= workerType.cost) {
            for (int dir = 0; dir < 4; dir++) {
                int nx = myBase.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
                int ny = myBase.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
                if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
                if (pgs.getUnitAt(nx, ny) != null) continue;
                UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE, dir, workerType);
                if (gs.isUnitActionAllowed(myBase, ua)) { train(myBase, workerType); break; }
            }
        }

        // Workers: one harvests, rest attack
        boolean hasHarvester = false;
        for (Unit w : myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (!hasHarvester && myBase != null && !resources.isEmpty()) {
                Unit res = nearest(w, resources);
                harvest(w, res, myBase);
                hasHarvester = true;
            } else if (!enemies.isEmpty()) {
                attack(w, nearest(w, enemies));
            }
        }

        return translateActions(player, gs);
    }

    private Unit nearest(Unit src, List<Unit> units) {
        Unit best = null; int bestD = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = Math.abs(src.getX()-u.getX()) + Math.abs(src.getY()-u.getY());
            if (d < bestD) { bestD = d; best = u; }
        }
        return best;
    }

    @Override
    public List<ParameterSpecification> getParameters() { return new ArrayList<>(); }
}