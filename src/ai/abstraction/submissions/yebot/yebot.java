/*
 * PureLLM Agent using DeepSeek API
 * All game decisions are made by the LLM - no hardcoded rules.
 * 
 * @author Ye
 * Team: deepseek_purellm
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

    // API Configuration - Set via environment variables
    private static final String DEEPSEEK_API_KEY = System.getenv("DEEPSEEK_API_KEY") != null 
            ? System.getenv("DEEPSEEK_API_KEY") : "";
    private static final String DEEPSEEK_MODEL = System.getenv("DEEPSEEK_MODEL") != null 
            ? System.getenv("DEEPSEEK_MODEL") : "deepseek-chat";
    private static final String API_URL = "https://api.deepseek.com/v1/chat/completions";

    // Timing Configuration
    private static final int LLM_INTERVAL = 10;      // Call LLM every N game ticks
    private static final int REQUEST_TIMEOUT = 30000; // 30 second timeout

    // Unit types
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // Game state tracking
    private int lastLLMTick = -100;

    // The prompt - carefully designed based on PureLLM experiment logs
    private static final String SYSTEM_PROMPT = """
You are an AI playing MicroRTS. You control the ALLY units. Your goal: DESTROY ALL ENEMY UNITS.

=== UNIT STATS ===
| Unit     | HP | Cost | Damage | Range | Speed | From     |
|----------|----|----- |--------|-------|-------|----------|
| Worker   | 1  | 1    | 1      | 1     | 1     | Base     |
| Light    | 4  | 2    | 2      | 1     | 2     | Barracks |
| Heavy    | 8  | 3    | 4      | 1     | 1     | Barracks |
| Ranged   | 3  | 2    | 1      | 3     | 1     | Barracks |

| Building | HP | Cost |
|----------|----|----- |
| Base     | 10 | 10   |
| Barracks | 5  | 5    |

=== ACTIONS ===
- move((x, y)) - Move toward target
- harvest((resource_x, resource_y), (base_x, base_y)) - Gather resources (WORKER ONLY)
- build((x, y), barracks) - Build barracks at location (WORKER ONLY)
- train(worker) - Train worker (BASE ONLY)
- train(light|heavy|ranged) - Train military unit (BARRACKS ONLY)
- attack((enemy_x, enemy_y)) - Attack enemy at location

=== CRITICAL RULES ===
1. ONE ACTION PER UNIT - Each unit_position can appear ONLY ONCE
2. Only command units marked "Status=idling"
3. harvest() second argument MUST be YOUR base position
4. Buildings (base, barracks) CANNOT move or attack

=== WINNING STRATEGY ===
**PHASE 1 (Turn 0-50): Economy**
- Worker harvests from nearest resource to base
- Base trains 1-2 workers

**PHASE 2 (Turn 50-100): Build Military**
- Worker builds barracks when resources >= 5
- CRITICAL: Once barracks exists, TRAIN MILITARY UNITS!
  Example: "(3, 2): barracks train(light)"

**PHASE 3 (Turn 100+): Attack**
- Use LIGHT/HEAVY/RANGED to attack (NOT workers!)
- Workers have 1 HP - they die instantly in combat
- Light units (4 HP, 2 damage) are your main fighters

=== OUTPUT FORMAT (JSON ONLY) ===
{
  "thinking": "Brief strategy (1 sentence)",
  "moves": [
    {
      "raw_move": "(x, y): unit_type action((args))",
      "unit_position": [x, y],
      "unit_type": "worker|light|heavy|ranged|base|barracks",
      "action_type": "move|harvest|build|train|attack"
    }
  ]
}

=== COMMON MISTAKES TO AVOID ===
❌ "(2, 1): base attack((5, 6))" - Bases CANNOT attack!
❌ Commanding same unit twice in one turn
❌ Attacking with workers instead of military units
❌ Forgetting to train units from barracks
❌ harvest((0,0), (5,6)) where (5,6) is NOT your base
""";

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
        lastLLMTick = -100;
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        lightType = utt.getUnitType("Light");
        heavyType = utt.getUnitType("Heavy");
        rangedType = utt.getUnitType("Ranged");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
    }

    @Override
    public AI clone() {
        return new yebot(utt, pf);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        int currentTick = gs.getTime();

        // Only call LLM every N ticks to reduce API calls
        if (currentTick - lastLLMTick >= LLM_INTERVAL) {
            lastLLMTick = currentTick;
            
            // Build game state description
            String gameStatePrompt = buildGameStatePrompt(player, gs);
            
            // Call LLM
            String response = callDeepSeekAPI(gameStatePrompt);
            
            // Parse and execute moves
            executeLLMResponse(response, player, gs);
        }

        return translateActions(player, gs);
    }

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
            if (u.getPlayer() == player) {
                team = "Ally";
            } else if (u.getPlayer() == -1) {
                team = "Resource";
            } else {
                team = "Enemy";
            }

            StringBuilder unitInfo = new StringBuilder();
            unitInfo.append("(").append(u.getX()).append(", ").append(u.getY()).append(") ");
            unitInfo.append(team).append(" ").append(u.getType().name);
            unitInfo.append(" {HP=").append(u.getHitPoints());
            
            if (u.getResources() > 0) {
                unitInfo.append(", Res=").append(u.getResources());
            }

            UnitActionAssignment uaa = gs.getActionAssignment(u);
            if (uaa != null) {
                unitInfo.append(", Status=busy}");
            } else {
                unitInfo.append(", Status=idling}");
                if (u.getPlayer() == player) {
                    idleAllies++;
                }
            }

            sb.append(unitInfo).append("\n");
        }

        sb.append("\nIdle ally units to command: ").append(idleAllies);
        
        return sb.toString();
    }

    private String callDeepSeekAPI(String gameStatePrompt) {
        if (DEEPSEEK_API_KEY.isEmpty()) {
            System.err.println("[PureLLM] ERROR: DEEPSEEK_API_KEY not set!");
            return "{\"thinking\":\"No API key\",\"moves\":[]}";
        }

        try {
            URL url = new URL(API_URL);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Authorization", "Bearer " + DEEPSEEK_API_KEY);
            conn.setDoOutput(true);
            conn.setConnectTimeout(REQUEST_TIMEOUT);
            conn.setReadTimeout(REQUEST_TIMEOUT);

            // Build request
            JsonObject request = new JsonObject();
            request.addProperty("model", DEEPSEEK_MODEL);
            
            JsonArray messages = new JsonArray();
            
            JsonObject systemMsg = new JsonObject();
            systemMsg.addProperty("role", "system");
            systemMsg.addProperty("content", SYSTEM_PROMPT);
            messages.add(systemMsg);
            
            JsonObject userMsg = new JsonObject();
            userMsg.addProperty("role", "user");
            userMsg.addProperty("content", gameStatePrompt);
            messages.add(userMsg);
            
            request.add("messages", messages);
            
            JsonObject responseFormat = new JsonObject();
            responseFormat.addProperty("type", "json_object");
            request.add("response_format", responseFormat);
            
            request.addProperty("temperature", 0.3);
            request.addProperty("max_tokens", 1024);

            // Send request
            try (OutputStream os = conn.getOutputStream()) {
                os.write(request.toString().getBytes(StandardCharsets.UTF_8));
            }

            // Read response
            int responseCode = conn.getResponseCode();
            if (responseCode == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder response = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) {
                        response.append(line);
                    }

                    JsonObject jsonResponse = JsonParser.parseString(response.toString()).getAsJsonObject();
                    JsonArray choices = jsonResponse.getAsJsonArray("choices");
                    if (choices != null && choices.size() > 0) {
                        return choices.get(0).getAsJsonObject()
                                .getAsJsonObject("message")
                                .get("content").getAsString();
                    }
                }
            } else {
                System.err.println("[PureLLM] API error: " + responseCode);
            }

        } catch (Exception e) {
            System.err.println("[PureLLM] API call failed: " + e.getMessage());
        }

        return "{\"thinking\":\"API error\",\"moves\":[]}";
    }

    private void executeLLMResponse(String response, int player, GameState gs) {
        try {
            JsonObject json = parseJsonResponse(response);
            if (json == null) return;

            JsonArray moves = json.getAsJsonArray("moves");
            if (moves == null) return;

            PhysicalGameState pgs = gs.getPhysicalGameState();
            Set<String> usedPositions = new HashSet<>();

            for (JsonElement moveEl : moves) {
                if (!moveEl.isJsonObject()) continue;
                JsonObject move = moveEl.getAsJsonObject();

                try {
                    // Get unit position
                    JsonArray pos = move.getAsJsonArray("unit_position");
                    if (pos == null || pos.size() < 2) continue;
                    
                    int unitX = pos.get(0).getAsInt();
                    int unitY = pos.get(1).getAsInt();
                    
                    // Prevent duplicate commands to same unit
                    String posKey = unitX + "," + unitY;
                    if (usedPositions.contains(posKey)) continue;
                    usedPositions.add(posKey);

                    // Find unit
                    Unit unit = pgs.getUnitAt(unitX, unitY);
                    if (unit == null || unit.getPlayer() != player) continue;
                    if (gs.getActionAssignment(unit) != null) continue;

                    // Get action info
                    String actionType = move.get("action_type").getAsString();
                    String rawMove = move.get("raw_move").getAsString();

                    // Execute action
                    executeAction(unit, actionType, rawMove, player, gs, pgs);

                } catch (Exception e) {
                    // Skip invalid move
                }
            }

        } catch (Exception e) {
            System.err.println("[PureLLM] Parse error: " + e.getMessage());
        }
    }

    private JsonObject parseJsonResponse(String response) {
        try {
            return JsonParser.parseString(response).getAsJsonObject();
        } catch (Exception e) {
            // Try to extract JSON from response
            int start = response.indexOf("{");
            int end = response.lastIndexOf("}") + 1;
            if (start >= 0 && end > start) {
                try {
                    return JsonParser.parseString(response.substring(start, end)).getAsJsonObject();
                } catch (Exception e2) {
                    // Give up
                }
            }
        }
        return null;
    }

    private void executeAction(Unit unit, String actionType, String rawMove, 
                               int player, GameState gs, PhysicalGameState pgs) {
        switch (actionType.toLowerCase()) {
            case "move": {
                if (unit.getType() == baseType || unit.getType() == barracksType) return;
                Matcher m = Pattern.compile("move\\(\\((\\d+),\\s*(\\d+)\\)\\)").matcher(rawMove);
                if (m.find()) {
                    int x = Integer.parseInt(m.group(1));
                    int y = Integer.parseInt(m.group(2));
                    move(unit, x, y);
                }
                break;
            }
            
            case "harvest": {
                if (unit.getType() != workerType) return;
                Matcher m = Pattern.compile("harvest\\(\\((\\d+),\\s*(\\d+)\\),\\s*\\((\\d+),\\s*(\\d+)\\)\\)").matcher(rawMove);
                if (m.find()) {
                    int resX = Integer.parseInt(m.group(1));
                    int resY = Integer.parseInt(m.group(2));
                    int baseX = Integer.parseInt(m.group(3));
                    int baseY = Integer.parseInt(m.group(4));
                    
                    Unit resource = pgs.getUnitAt(resX, resY);
                    Unit base = pgs.getUnitAt(baseX, baseY);
                    
                    if (resource != null && base != null && 
                        base.getType() == baseType && base.getPlayer() == player) {
                        harvest(unit, resource, base);
                    }
                }
                break;
            }
            
            case "build": {
                if (unit.getType() != workerType) return;
                Matcher m = Pattern.compile("build\\(\\((\\d+),\\s*(\\d+)\\),\\s*(\\w+)\\)").matcher(rawMove);
                if (m.find()) {
                    int x = Integer.parseInt(m.group(1));
                    int y = Integer.parseInt(m.group(2));
                    String buildingName = m.group(3).toLowerCase();
                    
                    UnitType buildingType = buildingName.equals("barracks") ? barracksType : baseType;
                    build(unit, buildingType, x, y);
                }
                break;
            }
            
            case "train": {
                Matcher m = Pattern.compile("train\\((\\w+)\\)").matcher(rawMove);
                if (m.find()) {
                    String unitName = m.group(1).toLowerCase();
                    UnitType trainType = null;
                    
                    if (unit.getType() == baseType && unitName.equals("worker")) {
                        trainType = workerType;
                    } else if (unit.getType() == barracksType) {
                        switch (unitName) {
                            case "light": trainType = lightType; break;
                            case "heavy": trainType = heavyType; break;
                            case "ranged": trainType = rangedType; break;
                        }
                    }
                    
                    if (trainType != null) {
                        train(unit, trainType);
                    }
                }
                break;
            }
            
            case "attack": {
                // Buildings cannot attack
                if (unit.getType() == baseType || unit.getType() == barracksType) return;
                
                Matcher m = Pattern.compile("attack\\(\\((\\d+),\\s*(\\d+)\\)\\)").matcher(rawMove);
                if (m.find()) {
                    int x = Integer.parseInt(m.group(1));
                    int y = Integer.parseInt(m.group(2));
                    Unit target = pgs.getUnitAt(x, y);
                    
                    if (target != null && target.getPlayer() != player && target.getPlayer() != -1) {
                        attack(unit, target);
                    }
                }
                break;
            }
        }
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}