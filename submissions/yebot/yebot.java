/*
 * LLM-Guided MCTS Agent for MicroRTS
 *
 * Architecture:
 *   - MCTS tree search for lookahead (written from scratch)
 *   - LLM used as evaluation function for leaf nodes
 *   - LLM also used for move generation (policy network analog)
 *   - No built-in AIs used anywhere
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
import java.util.regex.*;

public class yebot extends AbstractionLayerAI {

    // ─── API Config ────────────────────────────────────────────────────────────
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "llama4:latest";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int REQUEST_TIMEOUT = 30000;

    // ─── MCTS Config ───────────────────────────────────────────────────────────
    private static final int MCTS_ITERATIONS      = 10;   // tree expansions per turn
    private static final int SIMULATION_DEPTH     = 15;   // game ticks per rollout
    private static final double UCB_C             = 1.41; // exploration constant
    private static final int LLM_EVAL_INTERVAL    = 3;    // eval every Nth MCTS node
    private static final int ACTION_INTERVAL      = 10;   // ticks between MCTS runs

    // ─── Unit Types ────────────────────────────────────────────────────────────
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ─── State ─────────────────────────────────────────────────────────────────
    private int lastActionTick = -100;
    private PlayerAction cachedAction = null;

    // ─── Prompts ───────────────────────────────────────────────────────────────

    /**
     * Used by the POLICY LLM: given a game state, generate candidate moves.
     * Runs once per MCTS iteration to produce the action to try at the root.
     */
    private static final String POLICY_SYSTEM_PROMPT = """
You are a MicroRTS move generator. Given a game state, output a small set of
candidate actions for the ALLY units that are worth exploring.

=== UNIT STATS ===
| Unit     | HP | Cost | Damage | Range | Speed | From     |
|----------|----|----- |--------|-------|-------|----------|
| Worker   | 1  | 1    | 1      | 1     | 1     | Base     |
| Light    | 4  | 2    | 2      | 1     | 2     | Barracks |
| Heavy    | 8  | 3    | 4      | 1     | 1     | Barracks |
| Ranged   | 3  | 2    | 1      | 3     | 1     | Barracks |
| Base     | 10 | 10   | -      | -     | -     | -        |
| Barracks | 5  | 5    | -      | -     | -     | -        |

=== ACTIONS ===
- move((x, y))
- harvest((res_x, res_y), (base_x, base_y))   [WORKER only]
- build((x, y), barracks)                      [WORKER only]
- train(worker)                                 [BASE only]
- train(light|heavy|ranged)                    [BARRACKS only]
- attack((enemy_x, enemy_y))

=== RULES ===
1. ONE action per unit (each position once)
2. Only command units with Status=idling
3. Buildings CANNOT move or attack
4. harvest() second arg MUST be YOUR base

=== OUTPUT (JSON ONLY) ===
{
  "thinking": "one sentence",
  "moves": [
    {
      "raw_move": "(x, y): unit_type action((args))",
      "unit_position": [x, y],
      "unit_type": "worker|light|heavy|ranged|base|barracks",
      "action_type": "move|harvest|build|train|attack"
    }
  ]
}
""";

    /**
     * Used by the EVALUATION LLM: given a simulated state, score it 0-100.
     * Runs at MCTS leaf nodes to backpropagate value estimates.
     */
    private static final String EVAL_SYSTEM_PROMPT = """
You are a MicroRTS position evaluator. Score the ALLY player's position from 0 to 100.

Scoring guide:
- 100: Ally has clearly won (enemy base destroyed or no enemy units)
- 75+: Strong advantage (more units, more resources, enemy base damaged)
- 50:  Even position
- 25-: Disadvantage (fewer units, lost base HP, no barracks)
- 0:   Lost (ally base destroyed or no ally units)

Consider:
1. Unit count and total HP comparison (ally vs enemy)
2. Resource advantage
3. Base HP comparison
4. Map control (units closer to enemy base)
5. Production capacity (barracks present?)

OUTPUT JSON ONLY:
{
  "score": 65,
  "reason": "one sentence explaining the score"
}
""";

    // ══════════════════════════════════════════════════════════════════════════
    //  MCTS NODE
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * A node in the MCTS tree.
     * Each node represents a game state reached by taking `actionTaken` from parent.
     */
    private class MCTSNode {
        MCTSNode parent;
        PlayerAction actionTaken;   // action that led to this state from parent
        GameState state;            // game state at this node
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

        // UCB1 score for selection
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
        lastActionTick = -100;
        cachedAction = null;
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
        int tick = gs.getTime();

        if (tick - lastActionTick >= ACTION_INTERVAL) {
            lastActionTick = tick;
            System.out.println("[MCTS] Tick " + tick + " — running search...");
            cachedAction = runMCTS(player, gs);
        }

        // If we have a cached action, apply it; otherwise fall through to translateActions
        if (cachedAction != null) {
            PlayerAction result = cachedAction;
            cachedAction = null;
            PlayerAction filtered = filterValidAction(result, player, gs);
            // fillWithNones ensures idle units don't stall the game engine
            filtered.fillWithNones(gs, player, 1);
            return filtered;
        }

        PlayerAction fallback = translateActions(player, gs);
        fallback.fillWithNones(gs, player, 1);
        return fallback;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  MCTS CORE
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Run MCTS from the current game state.
     * Returns the best PlayerAction found.
     */
    private PlayerAction runMCTS(int player, GameState gs) throws Exception {
        MCTSNode root = new MCTSNode(null, null, gs.clone(), player);

        for (int i = 0; i < MCTS_ITERATIONS; i++) {
            // 1. SELECTION — walk tree using UCB1
            MCTSNode selected = select(root);

            // 2. EXPANSION — use LLM policy to generate children
            if (selected.visitCount > 0 || selected == root) {
                expand(selected);
            }

            // 3. Choose a child to simulate from (or simulate from selected)
            MCTSNode toSimulate = selected;
            if (!selected.children.isEmpty()) {
                // Pick unvisited child first, else random
                toSimulate = selected.children.stream()
                        .filter(c -> c.visitCount == 0)
                        .findFirst()
                        .orElse(selected.children.get(
                                new Random().nextInt(selected.children.size())));
            }

            // 4. SIMULATION — fast-forward game state
            GameState simState = simulate(toSimulate.state.clone(), player);

            // 5. EVALUATION — LLM scores the resulting position
            double value = evaluateState(player, simState, i);

            // 6. BACKPROPAGATION
            backpropagate(toSimulate, value);
        }

        // Pick best child of root by average value
        return bestAction(root);
    }

    // ── Selection ──────────────────────────────────────────────────────────────

    private MCTSNode select(MCTSNode node) {
        while (!node.children.isEmpty()) {
            // If any child unvisited, select it immediately
            Optional<MCTSNode> unvisited = node.children.stream()
                    .filter(c -> c.visitCount == 0)
                    .findFirst();
            if (unvisited.isPresent()) return unvisited.get();

            // Otherwise use UCB1
            node = node.children.stream()
                    .max(Comparator.comparingDouble(MCTSNode::ucb1))
                    .orElse(node);

            // Stop if leaf
            if (!node.expanded) break;
        }
        return node;
    }

    // ── Expansion ──────────────────────────────────────────────────────────────

    /**
     * Use the POLICY LLM to generate candidate moves, create child nodes.
     */
    private void expand(MCTSNode node) {
        if (node.expanded) return;
        node.expanded = true;

        try {
            String statePrompt = buildGameStatePrompt(node.player, node.state);
            String response = callLLM(statePrompt, POLICY_SYSTEM_PROMPT);
            List<PlayerAction> candidates = parseCandidateActions(response, node.player, node.state);

            for (PlayerAction action : candidates) {
                try {
                    GameState childState = node.state.clone();
                    // Issue the action in simulation
                    childState.issueSafe(action);
                    // Advance a few ticks so the action takes effect
                    for (int t = 0; t < 3; t++) {
                        if (childState.cycle()) break;
                    }
                    MCTSNode child = new MCTSNode(node, action, childState, node.player);
                    node.children.add(child);
                } catch (Exception e) {
                    // Skip invalid action
                }
            }

            // Always add a "do nothing" child as fallback
            if (node.children.isEmpty()) {
                MCTSNode passChild = new MCTSNode(node, new PlayerAction(), node.state.clone(), node.player);
                node.children.add(passChild);
            }

        } catch (Exception e) {
            System.err.println("[MCTS] Expansion failed: " + e.getMessage());
            MCTSNode passChild = new MCTSNode(node, new PlayerAction(), node.state.clone(), node.player);
            node.children.add(passChild);
        }
    }

    // ── Simulation ─────────────────────────────────────────────────────────────

    /**
     * Fast-forward the game state by SIMULATION_DEPTH ticks.
     * No AI is used — units with no actions just idle (reflecting real play).
     */
    private GameState simulate(GameState state, int player) {
        try {
            for (int t = 0; t < SIMULATION_DEPTH; t++) {
                boolean done = state.cycle();
                if (done) break; // game over
            }
        } catch (Exception e) {
            // Partial simulation is fine
        }
        return state;
    }

    // ── Evaluation ─────────────────────────────────────────────────────────────

    /**
     * Use the EVALUATION LLM to score a game state.
     * Falls back to heuristic scoring every other call to save API time.
     */
    private double evaluateState(int player, GameState gs, int iteration) {
        // Use LLM every LLM_EVAL_INTERVAL iterations, heuristic otherwise
        if (iteration % LLM_EVAL_INTERVAL == 0) {
            return evaluateWithLLM(player, gs);
        } else {
            return evaluateWithHeuristic(player, gs);
        }
    }

    private double evaluateWithLLM(int player, GameState gs) {
        try {
            String statePrompt = buildGameStatePrompt(player, gs);
            String response = callLLM(statePrompt, EVAL_SYSTEM_PROMPT);

            JsonObject json = parseJsonResponse(response);
            if (json != null && json.has("score")) {
                double score = json.get("score").getAsDouble();
                System.out.println("[MCTS] LLM eval: " + score +
                        (json.has("reason") ? " — " + json.get("reason").getAsString() : ""));
                return score / 100.0; // normalize to 0-1
            }
        } catch (Exception e) {
            System.err.println("[MCTS] LLM eval failed, falling back to heuristic");
        }
        return evaluateWithHeuristic(player, gs);
    }

    /**
     * Fast heuristic evaluation — used when skipping LLM call.
     * Scores based on unit counts, HP, resources, base health.
     */
    private double evaluateWithHeuristic(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemy = 1 - player;

        double allyScore  = 0;
        double enemyScore = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == -1) continue; // resource

            double unitValue = getUnitValue(u);

            if (u.getPlayer() == player) {
                allyScore += unitValue;
            } else {
                enemyScore += unitValue;
            }
        }

        // Resources
        allyScore  += gs.getPlayer(player).getResources() * 0.5;
        enemyScore += gs.getPlayer(enemy).getResources()  * 0.5;

        double total = allyScore + enemyScore;
        if (total == 0) return 0.5;

        return allyScore / total;
    }

    private double getUnitValue(Unit u) {
        String name = u.getType().name;
        double hpFraction = (double) u.getHitPoints() / u.getType().hp;
        double baseVal;
        switch (name) {
            case "Base":     baseVal = 10.0; break;
            case "Barracks": baseVal = 5.0;  break;
            case "Heavy":    baseVal = 4.0;  break;
            case "Light":    baseVal = 2.5;  break;
            case "Ranged":   baseVal = 2.5;  break;
            case "Worker":   baseVal = 1.0;  break;
            default:         baseVal = 1.0;
        }
        return baseVal * hpFraction;
    }

    // ── Backpropagation ────────────────────────────────────────────────────────

    private void backpropagate(MCTSNode node, double value) {
        while (node != null) {
            node.visitCount++;
            node.totalValue += value;
            node = node.parent;
        }
    }

    // ── Best Action Selection ──────────────────────────────────────────────────

    private PlayerAction bestAction(MCTSNode root) {
        if (root.children.isEmpty()) {
            System.out.println("[MCTS] No children — returning empty action");
            return new PlayerAction();
        }

        MCTSNode best = root.children.stream()
                .max(Comparator.comparingDouble(MCTSNode::avgValue))
                .orElse(root.children.get(0));

        System.out.printf("[MCTS] Best action: avg=%.3f visits=%d%n",
                best.avgValue(), best.visitCount);

        return best.actionTaken != null ? best.actionTaken : new PlayerAction();
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  ACTION PARSING
    //  Converts LLM policy output → PlayerAction (MicroRTS format)
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Parse the LLM policy response into a list of candidate PlayerActions.
     * Each "moves" array in the LLM response becomes one PlayerAction.
     */
    private List<PlayerAction> parseCandidateActions(String response, int player, GameState gs) {
        List<PlayerAction> actions = new ArrayList<>();

        try {
            JsonObject json = parseJsonResponse(response);
            if (json == null) return actions;

            JsonArray moves = json.getAsJsonArray("moves");
            if (moves == null) return actions;

            PhysicalGameState pgs = gs.getPhysicalGameState();
            Set<String> usedPositions = new HashSet<>();
            PlayerAction action = new PlayerAction();

            for (JsonElement moveEl : moves) {
                if (!moveEl.isJsonObject()) continue;
                JsonObject move = moveEl.getAsJsonObject();

                try {
                    JsonArray pos = move.getAsJsonArray("unit_position");
                    if (pos == null || pos.size() < 2) continue;

                    int unitX = pos.get(0).getAsInt();
                    int unitY = pos.get(1).getAsInt();
                    String posKey = unitX + "," + unitY;
                    if (usedPositions.contains(posKey)) continue;
                    usedPositions.add(posKey);

                    Unit unit = pgs.getUnitAt(unitX, unitY);
                    if (unit == null || unit.getPlayer() != player) continue;
                    if (gs.getActionAssignment(unit) != null) continue;

                    String actionType = move.get("action_type").getAsString();
                    String rawMove    = move.get("raw_move").getAsString();

                    UnitAction ua = parseUnitAction(unit, actionType, rawMove, player, gs, pgs);
                    if (ua != null) {
                        action.addUnitAction(unit, ua);
                    }

                } catch (Exception e) {
                    // Skip bad move entry
                }
            }

            if (!action.isEmpty()) {
                actions.add(action);
            }

        } catch (Exception e) {
            System.err.println("[MCTS] Action parse error: " + e.getMessage());
        }

        return actions;
    }

    /**
     * Convert a single LLM move description into a UnitAction.
     */
    private UnitAction parseUnitAction(Unit unit, String actionType, String rawMove,
                                        int player, GameState gs, PhysicalGameState pgs) {
        try {
            switch (actionType.toLowerCase()) {

                case "move": {
                    if (unit.getType() == baseType || unit.getType() == barracksType) return null;
                    Matcher m = Pattern.compile("move\\(\\((\\d+),\\s*(\\d+)\\)\\)").matcher(rawMove);
                    if (m.find()) {
                        int tx = Integer.parseInt(m.group(1));
                        int ty = Integer.parseInt(m.group(2));
                        UnitAction moveAction = pf.findPathToAdjacentPosition(unit,
                                tx + ty * pgs.getWidth(), gs, null);
                        if (moveAction != null) {
                            return new UnitAction(UnitAction.TYPE_MOVE, moveAction.getDirection());
                        }
                    }
                    break;
                }

                case "harvest": {
                    if (unit.getType() != workerType) return null;
                    Matcher m = Pattern.compile(
                            "harvest\\(\\((\\d+),\\s*(\\d+)\\),\\s*\\((\\d+),\\s*(\\d+)\\)\\)")
                            .matcher(rawMove);
                    if (m.find()) {
                        int resX  = Integer.parseInt(m.group(1));
                        int resY  = Integer.parseInt(m.group(2));
                        int baseX = Integer.parseInt(m.group(3));
                        int baseY = Integer.parseInt(m.group(4));

                        Unit resource = pgs.getUnitAt(resX, resY);
                        Unit base     = pgs.getUnitAt(baseX, baseY);

                        if (resource != null && base != null
                                && base.getType() == baseType
                                && base.getPlayer() == player) {

                            if (unit.getResources() > 0) {
                                // Carry back to base — just move toward it; return direction comes from adjacency
                                UnitAction moveAction = pf.findPathToAdjacentPosition(unit,
                                        baseX + baseY * pgs.getWidth(), gs, null);
                                if (moveAction != null) {
                                    int dir = moveAction.getDirection();
                                    // If adjacent, issue RETURN; otherwise MOVE
                                    int dx = Math.abs(unit.getX() - baseX);
                                    int dy = Math.abs(unit.getY() - baseY);
                                    if (dx + dy == 1) {
                                        return new UnitAction(UnitAction.TYPE_RETURN, dir);
                                    } else {
                                        return new UnitAction(UnitAction.TYPE_MOVE, dir);
                                    }
                                }
                            } else {
                                // Go harvest
                                UnitAction moveAction = pf.findPathToAdjacentPosition(unit,
                                        resX + resY * pgs.getWidth(), gs, null);
                                if (moveAction != null) {
                                    int dir = moveAction.getDirection();
                                    int dx = Math.abs(unit.getX() - resX);
                                    int dy = Math.abs(unit.getY() - resY);
                                    if (dx + dy == 1) {
                                        return new UnitAction(UnitAction.TYPE_HARVEST, dir);
                                    } else {
                                        return new UnitAction(UnitAction.TYPE_MOVE, dir);
                                    }
                                }
                            }
                        }
                    }
                    break;
                }

                case "build": {
                    if (unit.getType() != workerType) return null;
                    Matcher m = Pattern.compile(
                            "build\\(\\((\\d+),\\s*(\\d+)\\),\\s*(\\w+)\\)").matcher(rawMove);
                    if (m.find()) {
                        int bx = Integer.parseInt(m.group(1));
                        int by = Integer.parseInt(m.group(2));
                        String bName = m.group(3).toLowerCase();
                        UnitType bt  = bName.equals("barracks") ? barracksType : baseType;

                        if (gs.getPlayer(player).getResources() >= bt.cost) {
                            UnitAction moveAction = pf.findPathToAdjacentPosition(unit,
                                    bx + by * pgs.getWidth(), gs, null);
                            if (moveAction != null) {
                                int dir = moveAction.getDirection();
                                int dx = Math.abs(unit.getX() - bx);
                                int dy = Math.abs(unit.getY() - by);
                                if (dx + dy == 1) {
                                    return new UnitAction(UnitAction.TYPE_PRODUCE, dir, bt);
                                } else {
                                    return new UnitAction(UnitAction.TYPE_MOVE, dir);
                                }
                            }
                        }
                    }
                    break;
                }

                case "train": {
                    Matcher m = Pattern.compile("train\\((\\w+)\\)").matcher(rawMove);
                    if (m.find()) {
                        String uName = m.group(1).toLowerCase();
                        UnitType trainType = null;

                        if (unit.getType() == baseType && uName.equals("worker")) {
                            trainType = workerType;
                        } else if (unit.getType() == barracksType) {
                            switch (uName) {
                                case "light":  trainType = lightType;  break;
                                case "heavy":  trainType = heavyType;  break;
                                case "ranged": trainType = rangedType; break;
                            }
                        }

                        if (trainType != null
                                && gs.getPlayer(player).getResources() >= trainType.cost) {
                            // Find a free adjacent direction to spawn
                            for (int dir = 0; dir < 4; dir++) {
                                int nx = unit.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
                                int ny = unit.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
                                if (nx >= 0 && ny >= 0
                                        && nx < pgs.getWidth() && ny < pgs.getHeight()
                                        && pgs.getUnitAt(nx, ny) == null) {
                                    return new UnitAction(UnitAction.TYPE_PRODUCE, dir, trainType);
                                }
                            }
                        }
                    }
                    break;
                }

                case "attack": {
                    if (unit.getType() == baseType || unit.getType() == barracksType) return null;
                    Matcher m = Pattern.compile(
                            "attack\\(\\((\\d+),\\s*(\\d+)\\)\\)").matcher(rawMove);
                    if (m.find()) {
                        int tx = Integer.parseInt(m.group(1));
                        int ty = Integer.parseInt(m.group(2));
                        Unit target = pgs.getUnitAt(tx, ty);

                        if (target != null && target.getPlayer() != player
                                && target.getPlayer() != -1) {

                            int dx = Math.abs(unit.getX() - tx);
                            int dy = Math.abs(unit.getY() - ty);
                            int dist = dx + dy;

                            if (dist <= unit.getType().attackRange) {
                                return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                                        tx + ty * pgs.getWidth());
                            } else {
                                UnitAction moveAction = pf.findPathToAdjacentPosition(unit,
                                        tx + ty * pgs.getWidth(), gs, null);
                                if (moveAction != null) {
                                    return new UnitAction(UnitAction.TYPE_MOVE,
                                            moveAction.getDirection());
                                }
                            }
                        }
                    }
                    break;
                }
            }
        } catch (Exception e) {
            // Return null for any parsing failure
        }
        return null;
    }

    /**
     * Filter a PlayerAction to remove commands for units that no longer exist
     * or are no longer idle (since the MCTS ran a few ticks ago).
     */
    private PlayerAction filterValidAction(PlayerAction action, int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        PlayerAction valid = new PlayerAction();

        for (rts.Pair<Unit, UnitAction> pair : action.getActions()) {
            Unit storedUnit = pair.m_a;
            UnitAction ua   = pair.m_b;

            // Find the unit by ID in the current state
            Unit liveUnit = null;
            for (Unit u : pgs.getUnits()) {
                if (u.getID() == storedUnit.getID()) {
                    liveUnit = u;
                    break;
                }
            }

            if (liveUnit != null
                    && liveUnit.getPlayer() == player
                    && gs.getActionAssignment(liveUnit) == null) {
                valid.addUnitAction(liveUnit, ua);
            }
        }

        return valid;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  GAME STATE DESCRIPTION
    // ══════════════════════════════════════════════════════════════════════════

    private String buildGameStatePrompt(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        StringBuilder sb = new StringBuilder();
        sb.append("Turn: ").append(gs.getTime()).append("/1500\n");
        sb.append("Resources: ").append(p.getResources()).append("\n");
        sb.append("Map: ").append(pgs.getWidth()).append("x").append(pgs.getHeight()).append("\n\n");
        sb.append("Units:\n");

        int idleAllies = 0;

        for (Unit u : pgs.getUnits()) {
            String team;
            if (u.getPlayer() == player)       team = "Ally";
            else if (u.getPlayer() == -1)      team = "Resource";
            else                               team = "Enemy";

            sb.append("(").append(u.getX()).append(", ").append(u.getY()).append(") ");
            sb.append(team).append(" ").append(u.getType().name);
            sb.append(" {HP=").append(u.getHitPoints());

            if (u.getResources() > 0) sb.append(", Res=").append(u.getResources());

            UnitActionAssignment uaa = gs.getActionAssignment(u);
            if (uaa != null) {
                sb.append(", Status=busy}");
            } else {
                sb.append(", Status=idling}");
                if (u.getPlayer() == player) idleAllies++;
            }
            sb.append("\n");
        }

        sb.append("\nIdle ally units to command: ").append(idleAllies);
        return sb.toString();
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LLM API CALL (Ollama OpenAI-compatible)
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
            sysMsg.addProperty("content", systemPrompt +
                    "\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no backticks.");
            messages.add(sysMsg);

            JsonObject userMsg = new JsonObject();
            userMsg.addProperty("role", "user");
            userMsg.addProperty("content", userPrompt);
            messages.add(userMsg);

            request.add("messages", messages);

            JsonObject responseFormat = new JsonObject();
            responseFormat.addProperty("type", "json_object");
            request.add("response_format", responseFormat);

            request.addProperty("temperature", 0.3);
            request.addProperty("max_tokens", 1024);

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
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8))) {
                    StringBuilder err = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) err.append(line);
                    System.err.println("[MCTS] API error " + code + ": " + err);
                }
            }

        } catch (Exception e) {
            System.err.println("[MCTS] LLM call failed: " + e.getMessage());
        }

        return "{\"thinking\":\"error\",\"moves\":[],\"score\":50}";
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  JSON UTILITIES
    // ══════════════════════════════════════════════════════════════════════════

    private JsonObject parseJsonResponse(String response) {
        // Strip <think>...</think> blocks (qwen3 thinking mode)
        response = response.replaceAll("(?s)<think>.*?</think>", "").trim();

        try {
            return JsonParser.parseString(response).getAsJsonObject();
        } catch (Exception e) {
            int start = response.indexOf("{");
            int end   = response.lastIndexOf("}") + 1;
            if (start >= 0 && end > start) {
                try {
                    return JsonParser.parseString(response.substring(start, end)).getAsJsonObject();
                } catch (Exception ignored) {}
            }
        }
        return null;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  REQUIRED OVERRIDES
    // ══════════════════════════════════════════════════════════════════════════

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}