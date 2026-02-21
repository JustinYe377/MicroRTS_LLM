/*
 * Example submission: a simple WorkerRush agent.
 *
 * This demonstrates the submission format without requiring an LLM.
 * Strategy: train workers, keep one harvesting, send the rest to attack.
 */
package ai.abstraction.submissions.example_team;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.AbstractAction;
import ai.abstraction.Harvest;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.*;

public class WorkerRushLLMAgent extends AbstractionLayerAI {

    protected UnitTypeTable utt;
    UnitType workerType;
    UnitType baseType;

    public WorkerRushLLMAgent(UnitTypeTable a_utt) {
        super(new AStarPathFinding());
        reset(a_utt);
    }

    public void reset() {
        super.reset();
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        if (utt != null) {
            workerType = utt.getUnitType("Worker");
            baseType = utt.getUnitType("Base");
        }
    }

    public AI clone() {
        return new WorkerRushLLMAgent(utt);
    }

    public PlayerAction getAction(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        // Train workers from bases
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType &&
                u.getPlayer() == player &&
                gs.getActionAssignment(u) == null) {
                if (p.getResources() >= workerType.cost) {
                    train(u, workerType);
                }
            }
        }

        // Collect workers
        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canHarvest && u.getPlayer() == player) {
                workers.add(u);
            }
        }

        // One worker harvests, rest attack
        if (!workers.isEmpty()) {
            // Check if we need a base
            int nbases = 0;
            for (Unit u : pgs.getUnits()) {
                if (u.getType() == baseType && u.getPlayer() == player) nbases++;
            }

            List<Unit> freeWorkers = new LinkedList<>(workers);

            // Build base if none exists
            if (nbases == 0 && !freeWorkers.isEmpty()) {
                if (p.getResources() >= baseType.cost) {
                    Unit u = freeWorkers.remove(0);
                    List<Integer> reserved = new LinkedList<>();
                    buildIfNotAlreadyBuilding(u, baseType, u.getX(), u.getY(), reserved, p, pgs);
                }
            }

            // First free worker harvests
            if (!freeWorkers.isEmpty()) {
                Unit harvester = freeWorkers.remove(0);
                Unit closestResource = null;
                Unit closestBase = null;
                int minDist = Integer.MAX_VALUE;

                for (Unit u : pgs.getUnits()) {
                    if (u.getType().isResource) {
                        int d = Math.abs(u.getX() - harvester.getX()) + Math.abs(u.getY() - harvester.getY());
                        if (d < minDist) {
                            closestResource = u;
                            minDist = d;
                        }
                    }
                }
                minDist = Integer.MAX_VALUE;
                for (Unit u : pgs.getUnits()) {
                    if (u.getType().isStockpile && u.getPlayer() == player) {
                        int d = Math.abs(u.getX() - harvester.getX()) + Math.abs(u.getY() - harvester.getY());
                        if (d < minDist) {
                            closestBase = u;
                            minDist = d;
                        }
                    }
                }

                if (closestResource != null && closestBase != null) {
                    harvest(harvester, closestResource, closestBase);
                } else {
                    freeWorkers.add(harvester);
                }
            }

            // Remaining workers attack nearest enemy
            for (Unit u : freeWorkers) {
                Unit closestEnemy = null;
                int minDist = Integer.MAX_VALUE;
                for (Unit u2 : pgs.getUnits()) {
                    if (u2.getPlayer() >= 0 && u2.getPlayer() != player) {
                        int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                        if (d < minDist) {
                            closestEnemy = u2;
                            minDist = d;
                        }
                    }
                }
                if (closestEnemy != null) {
                    attack(u, closestEnemy);
                }
            }
        }

        return translateActions(player, gs);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}
