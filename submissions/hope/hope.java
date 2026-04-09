package ai.abstraction.submissions.hope;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.WorkerRush;
import ai.abstraction.LightRush;
import ai.abstraction.HeavyRush;
import ai.abstraction.RangedRush;
import ai.abstraction.EconomyRush;
import ai.abstraction.BoomEconomy;
import ai.abstraction.TurtleDefense;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.ParameterSpecification;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import com.google.gson.*;

import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.*;

/**
 * HOPE: Hybrid Ollama Plus Extrapolation 
 *
 * Use combination of both LLM and MCTS in conjunction with a predictive analysis of opponent's 
 * strategy to find the best move to take.
 * 
 * Also called HOPE because llama3.1:8b requires a lot of hope to use.
 * 
 * Strategies:
 * - WORKER_RUSH: Fast early aggression with workers (no barracks needed)
 * - LIGHT_RUSH: Build barracks, train light units (balanced speed/cost)
 * - HEAVY_RUSH: Train heavy units (high HP, high damage, counters infantry)
 * - RANGED_RUSH: Train ranged units (attack from distance, counters melee)
 */

/**
 * HOPE was originally based on this:
 * HybridLLMRush: Combines efficient rule-based Rush strategies with periodic LLM strategic guidance.
 *
 * The agent executes proven rush strategies most of the time while consulting an LLM every N ticks
 * to decide which strategy to use. This approach balances tactical efficiency with strategic adaptability.
 */
public class hope extends AbstractionLayerAI {

    public enum RushStrategy {
        WORKER_RUSH,
        LIGHT_RUSH,
        HEAVY_RUSH,
        RANGED_RUSH,
        BALANCED,
        ECONOMY_BOOM,
        TURTLE
    }

    // Strategy instances (composition pattern)
    private WorkerRush workerRushAI;
    private LightRush lightRushAI;
    private HeavyRush heavyRushAI;
    private RangedRush rangedRushAI;
    private TurtleDefense turtleAI;
    private BoomEconomy economyAI;
    private EconomyRush balancedAI; //uses EconomyRush


    // Unit type table reference
    protected UnitTypeTable utt;

    // Current strategy state
    private RushStrategy currentStrategy = RushStrategy.WORKER_RUSH;
    private int lastLLMConsultation = -9999;  // Force first consultation

    // HOPE 2.0
    private int lastStrategySwitch = -9999;                                
    private static final int MIN_STRATEGY_COMMITMENT = 100;  // Ticks before allowing a switch   

    // Configuration (from environment variables)
    private static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String MODEL =
            System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    private static final int LLM_INTERVAL =
            Integer.parseInt(System.getenv().getOrDefault("HYBRID_LLM_INTERVAL", "40"));

    // Statistics
    private int strategyChanges = 0;
    private int llmConsultations = 0;
    private int llmErrors = 0;

    /**
     * Constructor with UnitTypeTable
     */
    public hope(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    /**
     * Constructor with UnitTypeTable and PathFinding
     */
    public hope(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        if (workerRushAI != null) workerRushAI.reset();
        if (lightRushAI != null) lightRushAI.reset();
        if (heavyRushAI != null) heavyRushAI.reset();
        if (rangedRushAI != null) rangedRushAI.reset();
        if (economyAI != null) economyAI.reset();
        if (balancedAI != null) balancedAI.reset();
        if (turtleAI != null) turtleAI.reset();
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        // Initialize all strategy instances
        workerRushAI = new WorkerRush(a_utt, pf);
        lightRushAI = new LightRush(a_utt, pf);
        heavyRushAI = new HeavyRush(a_utt, pf);
        rangedRushAI = new RangedRush(a_utt, pf);
        economyAI = new BoomEconomy(a_utt, pf);
        balancedAI = new EconomyRush(a_utt, pf);
        turtleAI = new TurtleDefense(a_utt, pf);

        System.out.println("[hope] Initialized with model=" + MODEL +
                           ", interval=" + LLM_INTERVAL + ", initial_strategy=" + currentStrategy);
    }

    @Override
    public AI clone() {
        hope clone = new hope(utt, pf);
        clone.currentStrategy = this.currentStrategy;
        return clone;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        int currentTime = gs.getTime();

        // Check if it's time to consult the LLM
        if (currentTime - lastLLMConsultation >= LLM_INTERVAL) {
            RushStrategy newStrategy = consultLLMForStrategy(player, gs);
            if (newStrategy != null && newStrategy != currentStrategy) {
                if (currentTime - lastStrategySwitch >= MIN_STRATEGY_COMMITMENT)
                {
                    switchStrategy(newStrategy, currentTime);
                    lastStrategySwitch = currentTime;
                }
            }
            lastLLMConsultation = currentTime;
        }

        // Delegate to the current strategy
        return getCurrentStrategyAI().getAction(player, gs);
    }

    /**
     * Get the AI instance for the current strategy
     */
    private AbstractionLayerAI getCurrentStrategyAI() {
        switch (currentStrategy) {
            case WORKER_RUSH:
                return workerRushAI;
            case LIGHT_RUSH:
                return lightRushAI;
            case HEAVY_RUSH:
                return heavyRushAI;
            case RANGED_RUSH:
                return rangedRushAI;
            case TURTLE:
                return turtleAI;
            case ECONOMY_BOOM:
                return economyAI;
            case BALANCED:
                return balancedAI;
            default:
                return workerRushAI;
        }
    }

    /**
     * Switch to a new strategy
     */
    private void switchStrategy(RushStrategy newStrategy, int currentTime) {
        System.out.println("[hope] T=" + currentTime + ": Strategy switch " +
                           currentStrategy + " -> " + newStrategy);
        currentStrategy = newStrategy;
        strategyChanges++;

        // Reset the new strategy's action queue to avoid conflicts
        getCurrentStrategyAI().reset();
    }

    /**
     * Consult the LLM to decide which strategy to use
     */
    private RushStrategy consultLLMForStrategy(int player, GameState gs) {
        llmConsultations++;

        try {
            String prompt = buildStrategicPrompt(player, gs);
            String response = callOllamaAPI(prompt);
            return parseStrategyResponse(response);
        } catch (Exception e) {
            llmErrors++;
            System.err.println("[hope] LLM consultation failed: " + e.getMessage());
            return null;  // Keep current strategy on error
        }
    }

    private String enemyTracker(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemy = 1 - player;

        // GET UNIT POSTITIONS:
        boolean UNDER_ATTACK = false;
        int num_enemies = 0;
        double avg_dist = 0;
        int sumX = 0;
        int sumY = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != enemy) continue;

            if (u.getType().name.equals("Light") || u.getType().name.equals("Ranged") || u.getType().name.equals("Worker") || u.getType().name.equals("Heavy") )
            {
                num_enemies++;
                sumX += u.getX();
                sumY += u.getY();
                for (Unit myUnit : pgs.getUnits()) {
                    if (myUnit.getPlayer() == player && myUnit.getType().name.equals("Base")) {
                        double dist = Math.abs(u.getX() - myUnit.getX()) + Math.abs(u.getY() - myUnit.getY());
                        avg_dist += dist;
                        if (dist < 5) UNDER_ATTACK = true;
                        break;
                    }
                }
            }
        }

        int centroidX = num_enemies > 0 ? sumX / num_enemies : 0;
        int centroidY = num_enemies > 0 ? sumY / num_enemies : 0;
        avg_dist = num_enemies > 0 ? avg_dist / num_enemies : 0;
        
        if(UNDER_ATTACK)
        {
            String out = String.format("WE ARE UNDER ATTACK BY %d ENEMIES. GROUP CENTER LOCATED AT %d X and %d Y.", num_enemies, centroidX, centroidY );
            return out;
        }

        else if(num_enemies == 0)
        {
            return "Enemy has yet to produce any forces and thus has no army position.";
        }

        else
        {
            String out = String.format("The enemies' forces are about %.1f away from the base at location %d X and %d Y.", avg_dist, centroidX, centroidY );
            return out;
        }

    }

    private String inferEnemyStrategy(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int enemy = 1 - player;
        
        boolean enemyHasBarracks = false;
        boolean enemyHasLight = false;
        int enemyLightCount = 0;
        boolean enemyHasRanged = false;
        int enemyRangedCount = 0;
        boolean enemyHasHeavy = false;
        int enemyHeavyCount = 0;
        boolean enemyWorkersAttacking = false;
        int enemyWorkerCount = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != enemy) continue;
            
            if (u.getType().name.equals("Barracks")) enemyHasBarracks = true;
            if (u.getType().name.equals("Light")){
                enemyHasLight = true;
                enemyLightCount++;
            }
            if (u.getType().name.equals("Ranged")){ 
                enemyHasRanged = true;
                enemyRangedCount++;
            }
            if (u.getType().name.equals("Heavy")){
                enemyHasHeavy = true;
                enemyHeavyCount++;
            }
            if (u.getType().name.equals("Worker")) {
                enemyWorkerCount++;
                // Check if worker is moving toward your base (simple proximity check)
                for (Unit myUnit : pgs.getUnits()) {
                    if (myUnit.getPlayer() == player && myUnit.getType().name.equals("Base")) {
                        double dist = Math.abs(u.getX() - myUnit.getX()) + Math.abs(u.getY() - myUnit.getY());
                        if (dist < 10) enemyWorkersAttacking = true;
                        break;
                    }
                }
            }
        }
        
        // for 8x8 
        if(pgs.getWidth() <= 8)
        {
            if (enemyWorkerCount >= 2 && enemyWorkersAttacking && gs.getTime() < 150) {
                return "WORKER_RUSH (confirmed)";
            } else if (enemyHasRanged) {
                return "RANGED_RUSH";
            } else if (enemyHasLight) {
                return "LIGHT_RUSH";
            } else if (enemyHasHeavy) {
                return "HEAVY_RUSH";
            } else if (enemyHasBarracks) {
                return "BUILDING_BARRACKS (likely LIGHT_RUSH soon)";
            }
            return "UNKNOWN... probably TURTLE";
        }
        else
        {
            if ( (enemyHasHeavy && enemyHasLight) || (enemyHasHeavy && enemyHasRanged) || (enemyHasLight && enemyHasRanged) )
            {
                // balanced
                return "BALANCED";
            }
            else if(enemyWorkerCount >= 6)
            {
                if(enemyWorkersAttacking)
                {
                    return "WORKER_RUSH";
                }
                return "ECONOMY_BOOM";
            }
            else if(enemyLightCount >= 2)
            {
                return "LIGHT_RUSH";
            }
            else if(enemyHeavyCount >= 2)
            {
                return "HEAVY_RUSH";
            }
            else if(enemyRangedCount >= 2)
            {
                return "RANGED_RUSH";
            }
            else
            {
                if(gs.getTime() < 100)
                {
                    return "UNKNOWN... too early to tell!";
                }
                return "UNKNOWN... probably TURTLE";
            }

        }

        //return "UNKNOWN... probably TURTLE";
    }

    /**
     * Build a simplified strategic prompt for the LLM
     */
    private String buildStrategicPrompt(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);
        int enemyPlayer = 1 - player;

        // Count units for both players
        int myWorkers = 0, myLight = 0, myHeavy = 0, myRanged = 0;
        int myBases = 0, myBarracks = 0;
        int enemyWorkers = 0, enemyLight = 0, enemyHeavy = 0, enemyRanged = 0;
        int enemyBases = 0, enemyBarracks = 0;

        UnitType workerType = utt.getUnitType("Worker");
        UnitType lightType = utt.getUnitType("Light");
        UnitType heavyType = utt.getUnitType("Heavy");
        UnitType rangedType = utt.getUnitType("Ranged");
        UnitType baseType = utt.getUnitType("Base");
        UnitType barracksType = utt.getUnitType("Barracks");

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == workerType) myWorkers++;
                else if (u.getType() == lightType) myLight++;
                else if (u.getType() == heavyType) myHeavy++;
                else if (u.getType() == rangedType) myRanged++;
                else if (u.getType() == baseType) myBases++;
                else if (u.getType() == barracksType) myBarracks++;
            } else if (u.getPlayer() == enemyPlayer) {
                if (u.getType() == workerType) enemyWorkers++;
                else if (u.getType() == lightType) enemyLight++;
                else if (u.getType() == heavyType) enemyHeavy++;
                else if (u.getType() == rangedType) enemyRanged++;
                else if (u.getType() == baseType) enemyBases++;
                else if (u.getType() == barracksType) enemyBarracks++;
            }
        }

        // Calculate military strength (simplified)
        int myStrength = myWorkers + myLight * 2 + myHeavy * 4 + myRanged * 2;
        int enemyStrength = enemyWorkers + enemyLight * 2 + enemyHeavy * 4 + enemyRanged * 2;
        String enemyStrategy = inferEnemyStrategy(player, gs);
        String enemyLocation = enemyTracker(player, gs);
        int mapSize = pgs.getWidth();

        // Determine game phase
        int maxCycles = 3000;  // Default, could be read from config
        String gamePhase;
        if (gs.getTime() < maxCycles / 4) {
            gamePhase = "EARLY";
        } else if (gs.getTime() < maxCycles * 3 / 4) {
            gamePhase = "MID";
        } else {
            gamePhase = "LATE";
        }

        // change prompts for strategies.
        StringBuilder sb = new StringBuilder();
        sb.append("You are a strategic advisor for a real-time strategy game.\n\n");
        sb.append("STRATEGIES:\n");
        sb.append("- WORKER_RUSH: Send two workers immediately to attack enemy. Do not spend time getting resources. Keep making workers and sending them to attack the enemy base. GOOD ON 8x8 MAP SIZE!\n");
        sb.append("- LIGHT_RUSH: Send workers to attack enemy base. Create more workers and harvest resources to build barracks. Create light units as soon as possible and send them to attack enemy base. COUNTERS RANGED UNITS!\n");
        sb.append("- HEAVY_RUSH: Send workers to attack enemy base. Create more workers and harvest resources to build barracks. Create heavy units as soon as possible and send them to attack enemy base. COUNTERS LIGHT UNITS!\n");
        sb.append("- TURTLE: Defend your base, build barracks, train heavy units. Only attack when fully prepared. GOOD WHEN UNDER ATTACK AND LOSING!\n");
        sb.append("- ECONOMY_BOOM: Maximize resource collection early. Train many workers. Build military late but with superior resources. GOOD FOR SLOW GAMES WITH MANY RESOURCES!\n");
        sb.append("- BALANCED: Play a balanced strategy. Build a mix of units. Harvest resources early, then build military. Attack when you have an advantage. GOOD FOR SLOW GAMES OR WHEN ENEMY IS USING MIXED ARMY!\n");
        sb.append("- RANGED_RUSH: Send workers immediately to attack enemy. While they are attacking enemy base, build more workers to harvest and start creating ranged units. Use those units to kite enemies and do not get close enough to engage in melee combat. COUNTERS HEAVY UNITS!\n\n");
        sb.append("GAME STATE:\n");
        sb.append("- Game phase: ").append(gamePhase).append("\n");
        sb.append("- Time: ").append(gs.getTime()).append("/").append(maxCycles).append("\n");
        sb.append("- Your resources: ").append(p.getResources()).append("\n");
        //sb.append("- Global resources: ").append(pgs.getResources()).append("\n");
        sb.append("- Your forces: ").append(myWorkers).append(" workers, ");
        sb.append(myLight).append(" light, ").append(myHeavy).append(" heavy, ");
        sb.append(myRanged).append(" ranged\n");
        sb.append("- Your buildings: ").append(myBases).append(" base, ");
        sb.append(myBarracks).append(" barracks\n");
        sb.append("- Enemy forces: ").append(enemyWorkers).append(" workers, ");
        sb.append(enemyLight).append(" light, ").append(enemyHeavy).append(" heavy, ");
        sb.append(enemyRanged).append(" ranged\n");
        sb.append("Current status of the enemy army: ").append(enemyLocation).append("\n");
        sb.append("- Your strength: ").append(myStrength).append(", Enemy strength: ");
        sb.append(enemyStrength).append("\n\n");
        sb.append("Enemy appears to be using: ").append(enemyStrategy).append("\n\n");
        if(mapSize <= 8){
            sb.append("COUNTER STRATEGIES (8x8 MAP):\n");
            sb.append("- You MUST use WORKER_RUSH immediately.\n");
            sb.append("  * Send two workers immediately to attack enemy.\n");
            sb.append("  * Do not spend time getting resources.\n");
            sb.append("  * Keep making workers and sending them to attack the enemy base.\n");
            sb.append("  * WORKER_RUSH beats everything on 8x8 except for some TURTLE strategies if workers are not created fast enough and sent to attack enemy base.\n\n");
        }
        else if(mapSize > 8)
        {
            sb.append("COUNTER STRATEGIES (16x16 or 32x32 MAP):\n");
            sb.append("- If game phase is EARLY or MID and enemy is using a rush strategy:\n");
            sb.append("  * HEAVY_RUSH counters LIGHT_RUSH.\n");
            sb.append("  * LIGHT_RUSH counters RANGED_RUSH.\n");
            sb.append("  * RANGED_RUSH counters HEAVY_RUSH.\n");
            sb.append("  * Continue using strategy to win if enemy has losing score even in LATE phase.\n");
            sb.append("- If game phase is MID or LATE and there are still many resources on the map:\n");
            sb.append("  * ECONOMY_BOOM works to maintain high resources and build up forces.\n");
            sb.append("  * BALANCED can work to counter general enemy attacks since they are not favoring one unit over another.\n");
            sb.append("- If strength is much lower than enemy and enemy is attacking:\n");
            sb.append("  * TURTLE works to repel enemy attack and hopefully rebound from bad situation.\n");
            sb.append("  * Remember to begin rebuilding forces as soon as possible and harvest resources while defending BASE at all costs.\n");
        }
        sb.append("CURRENT GAME PHASE:\n");
        sb.append("- If time < 100 and enemy has more workers fighting: WORKER_RUSH is mandatory.\n");
        sb.append("Current strategy: ").append(currentStrategy).append("\n\n");
        sb.append("Which strategy should we use? Reply with a JSON object containing ONE word for the strategy:\n");
        sb.append("{\"strategy\": \"WORKER_RUSH\"} or {\"strategy\": \"ECONOMY_BOOM\"} or {\"strategy\": \"LIGHT_RUSH\"} or ");
        sb.append("{\"strategy\": \"HEAVY_RUSH\"} or {\"strategy\": \"RANGED_RUSH\"} or {\"strategy\": \"TURTLE\"} or {\"strategy\": \"BALANCED\"}\n");

        return sb.toString();
    }

    /**
     * Call the Ollama API
     */
    private String callOllamaAPI(String prompt) throws Exception {
        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("prompt", "/no_think " + prompt);
        body.addProperty("stream", false);
        body.addProperty("format", "json");

        URL url = new URL(OLLAMA_HOST + "/api/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setConnectTimeout(5000);
        conn.setReadTimeout(10000);
        conn.setDoOutput(true);

        try (OutputStream os = conn.getOutputStream()) {
            byte[] input = body.toString().getBytes(StandardCharsets.UTF_8);
            os.write(input);
        }

        int code = conn.getResponseCode();
        InputStream is = (code == HttpURLConnection.HTTP_OK)
                ? conn.getInputStream()
                : conn.getErrorStream();

        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
            for (String line; (line = br.readLine()) != null; ) {
                sb.append(line);
            }
        }

        if (code != HttpURLConnection.HTTP_OK) {
            throw new IOException("Ollama API error (" + code + "): " + sb.toString());
        }

        // Parse Ollama response to get the model's text output
        JsonObject top = JsonParser.parseString(sb.toString()).getAsJsonObject();
        if (top.has("response") && !top.get("response").getAsString().isEmpty()) {
            return top.get("response").getAsString();
        }
        throw new IOException("No response field in Ollama output");
    }

    /**
     * Parse the LLM response to extract strategy choice
     */
    private RushStrategy parseStrategyResponse(String response) {
        if (response == null || response.isEmpty()) {
            return null;
        }

        try {
            // Try to parse as JSON first
            String cleaned = response.trim();
            if (cleaned.startsWith("{")) {
                JsonObject json = JsonParser.parseString(cleaned).getAsJsonObject();
                if (json.has("strategy")) {
                    String strategyStr = json.get("strategy").getAsString().toUpperCase();
                    return parseStrategyString(strategyStr);
                }
            }
        } catch (Exception e) {
            // Fall back to text parsing
        }

        // Try to find strategy name in plain text
        String upper = response.toUpperCase();
        if (upper.contains("WORKER_RUSH")) return RushStrategy.WORKER_RUSH;
        if (upper.contains("LIGHT_RUSH")) return RushStrategy.LIGHT_RUSH;
        if (upper.contains("HEAVY_RUSH")) return RushStrategy.HEAVY_RUSH;
        if (upper.contains("RANGED_RUSH")) return RushStrategy.RANGED_RUSH;
        if (upper.contains("TURTLE")) return RushStrategy.TURTLE;
        if (upper.contains("BALANCED")) return RushStrategy.BALANCED;
        if (upper.contains("ECONOMY_BOOM")) return RushStrategy.ECONOMY_BOOM;

        System.out.println("[hope] Could not parse strategy from: " + response);
        return null;
    }

    /**
     * Parse strategy string to enum
     */
    private RushStrategy parseStrategyString(String s) {
        switch (s) {
            case "WORKER_RUSH": return RushStrategy.WORKER_RUSH;
            case "LIGHT_RUSH": return RushStrategy.LIGHT_RUSH;
            case "HEAVY_RUSH": return RushStrategy.HEAVY_RUSH;
            case "RANGED_RUSH": return RushStrategy.RANGED_RUSH;
            case "TURTLE": return RushStrategy.TURTLE;
            case "BALANCED": return RushStrategy.BALANCED;
            case "ECONOMY_BOOM": return RushStrategy.ECONOMY_BOOM;
            default: return null;
        }
    }

    /**
     * Get current strategy (for testing/debugging)
     */
    public RushStrategy getCurrentStrategy() {
        return currentStrategy;
    }

    /**
     * Set strategy manually (for testing/debugging)
     */
    public void setStrategy(RushStrategy strategy) {
        this.currentStrategy = strategy;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("PathFinding", PathFinding.class, new AStarPathFinding()));
        return parameters;
    }

    @Override
    public String toString() {
        return "hope(model=" + MODEL + ", strategy=" + currentStrategy +
               ", changes=" + strategyChanges + ", consultations=" + llmConsultations +
               ", errors=" + llmErrors + ")";
    }
}
