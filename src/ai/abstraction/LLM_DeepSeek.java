package ai.abstraction;

import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.ParameterSpecification;

import java.time.Instant;
import java.util.*;
import java.util.regex.*;
import java.io.*;
import java.net.*;
import com.google.gson.*;
import rts.GameState;
import rts.PhysicalGameState;
import rts.UnitAction;
import rts.Player;
import rts.PlayerAction;
import rts.units.*;
import java.text.SimpleDateFormat;

/**
 * DeepSeek Cloud API-based AI for MicroRTS.
 * Uses DeepSeek's OpenAI-compatible API for fast cloud inference.
 * 
 * Environment Variables:
 *   DEEPSEEK_API_KEY - Required: Your DeepSeek API key
 *   DEEPSEEK_MODEL - Optional: Model to use (default: deepseek-chat)
 * 
 * Get API key at: https://platform.deepseek.com
 */
public class LLM_DeepSeek extends AbstractionLayerAI {

    // ==== DEEPSEEK API CONFIG ====
    static final String DEEPSEEK_API_KEY = 
            System.getenv().getOrDefault("DEEPSEEK_API_KEY", "");
    
    static final String DEEPSEEK_MODEL = 
            System.getenv().getOrDefault("DEEPSEEK_MODEL", "deepseek-chat");
    
    static final String DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions";
    
    // How often the LLM should act (higher = fewer API calls, faster games)
    // 10-20 is good balance between responsiveness and speed
    static final Integer LLM_INTERVAL = 10;
    
    // Request timeout in milliseconds
    static final int REQUEST_TIMEOUT = 30000;

    protected UnitTypeTable utt;
    UnitType resourceType, workerType, lightType, heavyType, rangedType, baseType, barracksType;

    Instant promptTime, responseTime;
    long Latency = 0;
    int totalTokens = 0;
    private boolean logsInitialized = false;
    String fileName01 = "";

    String PROMPT = """
=== GAME RULES ===
Two players, Player 0 (Ally) and Player 1 (Enemy) compete to eliminate all opposing units in a Real Time Strategy (RTS) game.
Each step, each player can assign actions to their units if they are not already doing an action.
The game is over once all units and buildings from either team are killed or destroyed.

=== CRITICAL RULES (VIOLATIONS CAUSE GAME ERRORS) ===
1. ONE ACTION PER UNIT PER TURN - Each unit_position can appear ONLY ONCE in moves[] array
2. You can ONLY command units marked as "Ally" - NEVER command "Enemy" or "Neutral" units
3. harvest() second argument MUST be your Ally Base position from the game state
4. Only command units that are "idling" - units already doing actions will ignore new commands

=== AVAILABLE ACTIONS ===
| Action  | Arguments                                      | Who Can Use      | Description                                    |
|---------|------------------------------------------------|------------------|------------------------------------------------|
| move    | ((Target_x, Target_y))                         | All mobile units | Move toward target location                    |
| train   | (Unit_Type)                                    | Base, Barracks   | Train a new unit (costs resources)             |
| build   | ((Target_x, Target_y), Building_Type)          | Worker only      | Build base or barracks at target location      |
| harvest | ((Resource_x, Resource_y), (Base_x, Base_y))   | Worker only      | Gather resource and return to YOUR base        |
| attack  | ((Enemy_x, Enemy_y))                           | All units        | Navigate to and attack enemy at position       |
| idle    | ()                                             | All units        | Do nothing this turn                           |

=== UNIT STATS ===
| Unit Type | HP | Cost | Attack Damage | Attack Range | Speed | Trained From |
|-----------|----|------|---------------|--------------|-------|--------------|
| worker    | 1  | 1    | 1             | 1            | 1     | Base         |
| light     | 4  | 2    | 2             | 1            | 2     | Barracks     |
| heavy     | 8  | 3    | 4             | 1            | 1     | Barracks     |
| ranged    | 3  | 2    | 1             | 3            | 1     | Barracks     |

=== BUILDING STATS ===
| Building  | HP | Cost | Produces                    |
|-----------|----|------|-----------------------------|
| base      | 10 | 10   | Workers, stores resources   |
| barracks  | 5  | 5    | Light, Heavy, Ranged units  |

=== AGGRESSIVE STRATEGY (FOLLOW THIS OR LOSE) ===
1. EARLY GAME (Turn 0-50):
   - Worker harvests from nearest resource node to YOUR base
   - Base trains 1-2 workers maximum
   - BUILD BARRACKS as soon as you have 5+ resources (URGENT!)

2. MID GAME (Turn 50-150):
   - Barracks trains light units (fast, cheap, good for rushing)
   - ATTACK enemy workers to cripple their economy
   - Keep only 1 worker harvesting

3. LATE GAME (Turn 150+):
   - Train heavy units for damage, ranged for safety
   - ALL-OUT ATTACK on enemy base
   - Destroy enemy barracks first, then base

=== REQUIRED JSON OUTPUT FORMAT ===
{
  "thinking": "One sentence strategy",
  "moves": [
    {
      "raw_move": "(x, y): unit_type action((args))",
      "unit_position": [x, y],
      "unit_type": "worker|light|heavy|ranged|base|barracks",
      "action_type": "move|train|build|harvest|attack|idle"
    }
  ]
}

=== COMMON ERRORS TO AVOID ===
ERROR 1 - Duplicate unit positions (GAME CRASH):
  BAD:  [{"unit_position": [1, 0], ...}, {"unit_position": [1, 0], ...}]
  GOOD: [{"unit_position": [1, 0], ...}, {"unit_position": [2, 0], ...}]

ERROR 2 - Wrong base position in harvest:
  BAD:  "harvest((0, 0), (1, 1))"   // (1,1) is not your base!
  GOOD: "harvest((0, 0), (2, 1))"   // Use YOUR actual base position from game state

ERROR 3 - Commanding enemy units:
  BAD:  Issuing commands to units marked "Enemy"
  GOOD: Only command units marked "Ally"

=== EXAMPLE VALID MOVES ===
Worker harvests:  {"raw_move": "(1, 1): worker harvest((0, 0), (2, 1))", "unit_position": [1, 1], "unit_type": "worker", "action_type": "harvest"}
Base trains:      {"raw_move": "(2, 1): base train(worker)", "unit_position": [2, 1], "unit_type": "base", "action_type": "train"}
Worker builds:    {"raw_move": "(1, 2): worker build((2, 2), barracks)", "unit_position": [1, 2], "unit_type": "worker", "action_type": "build"}
Barracks trains:  {"raw_move": "(2, 2): barracks train(light)", "unit_position": [2, 2], "unit_type": "barracks", "action_type": "train"}
Light attacks:    {"raw_move": "(4, 3): light attack((5, 6))", "unit_position": [4, 3], "unit_type": "light", "action_type": "attack"}

REMEMBER: Passive play LOSES. Build barracks early, attack often, destroy enemy economy!
""";

    public LLM_DeepSeek(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
        System.out.println("[LLM_DeepSeek] Initialized with model: " + DEEPSEEK_MODEL);
        if (DEEPSEEK_API_KEY.isEmpty()) {
            System.err.println("[LLM_DeepSeek] WARNING: DEEPSEEK_API_KEY not set!");
        }
    }

    public LLM_DeepSeek(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public void reset() {
        super.reset();
        TIME_BUDGET = -1;
        ITERATIONS_BUDGET = -1;
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        resourceType = utt.getUnitType("Resource");
        workerType = utt.getUnitType("Worker");
        lightType = utt.getUnitType("Light");
        heavyType = utt.getUnitType("Heavy");
        rangedType = utt.getUnitType("Ranged");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
    }

    @Override
    public AI clone() {
        return new LLM_DeepSeek(utt, pf);
    }

    private void initLogsIfNeeded() {
        if (logsInitialized) return;
        String ts = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new Date());
        fileName01 = "Response" + ts + "_DeepSeek_" + DEEPSEEK_MODEL.replace(":", "-") + ".csv";
        try (FileWriter writer = new FileWriter(fileName01)) {
            writer.append("Thinking,Moves,Latency,Tokens\n");
        } catch (IOException e) {
            e.printStackTrace();
        }
        logsInitialized = true;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        initLogsIfNeeded();

        if (gs.getTime() % LLM_INTERVAL != 0) {
            return translateActions(player, gs);
        }

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        ArrayList<String> features = new ArrayList<>();
        int maxActions = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) maxActions++;

            String unitStats;
            UnitAction unitAction = gs.getUnitAction(u);
            String unitActionString = unitActionToString(unitAction);
            String unitType;

            if (u.getType() == resourceType) {
                unitType = "Resource Node";
                unitStats = "{resources=" + u.getResources() + "}";
            } else if (u.getType() == baseType) {
                unitType = "Base Unit";
                unitStats = "{resources=" + p.getResources() + ", current_action=\"" + unitActionString + "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == barracksType) {
                unitType = "Barracks Unit";
                unitStats = "{current_action=\"" + unitActionString + "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == workerType) {
                unitType = "Worker Unit";
                unitStats = "{current_action=\"" + unitActionString + "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == lightType) {
                unitType = "Light Unit";
                unitStats = "{current_action=\"" + unitActionString + "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == heavyType) {
                unitType = "Heavy Unit";
                unitStats = "{current_action=\"" + unitActionString + "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == rangedType) {
                unitType = "Ranged Unit";
                unitStats = "{current_action=\"" + unitActionString + "\", HP=" + u.getHitPoints() + "}";
            } else {
                unitType = "Unknown";
                unitStats = "{}";
            }

            String unitPos = "(" + u.getX() + ", " + u.getY() + ")";
            String team = (u.getPlayer() == player) ? "Ally" :
                    (u.getType() == resourceType ? "Neutral" : "Enemy");

            features.add(unitPos + " " + team + " " + unitType + " " + unitStats);
        }

        String mapPrompt = "Map size: " + pgs.getWidth() + "x" + pgs.getHeight();
        String turnPrompt = "Turn: " + gs.getTime() + "/" + 3000;
        String maxActionsPrompt = "Max actions: " + maxActions;
        String featuresPrompt = "Feature locations:\n" + String.join("\n", features);

        String finalPrompt = PROMPT + "\n\n" + mapPrompt + "\n" + turnPrompt + "\n" + maxActionsPrompt + "\n\n" + featuresPrompt + "\n";

        String response = prompt(finalPrompt);
        System.out.println("[LLM_DeepSeek] Response received, latency: " + Latency + "ms");

        // Log response to CSV file
        try (FileWriter writer = new FileWriter(fileName01, true)) {
            String csvSafeResponse = response.replace("\"", "\"\"").replace("\n", "\\n").replace("\r", "");
            writer.append("\"" + csvSafeResponse + "\"," + Latency + "," + totalTokens + "\n");
        } catch (IOException e) {
            System.err.println("[LLM_DeepSeek] Error writing log: " + e.getMessage());
        }

        JsonObject jsonResponse = parseJsonStrictThenLenient(response);
        JsonArray moveElements = jsonResponse.getAsJsonArray("moves");

        if (moveElements == null || moveElements.size() == 0) {
            return translateActions(player, gs);
        }

        for (JsonElement moveElement : moveElements) {
            try {
                if (!moveElement.isJsonObject()) continue;
                JsonObject move = moveElement.getAsJsonObject();

                if (!move.has("unit_position") || !move.get("unit_position").isJsonArray()) continue;
                JsonArray unitPosition = move.getAsJsonArray("unit_position");
                if (unitPosition.size() < 2) continue;

                int unitX = unitPosition.get(0).getAsInt();
                int unitY = unitPosition.get(1).getAsInt();
                Unit unit = pgs.getUnitAt(unitX, unitY);

                if (unit == null || unit.getPlayer() != player) continue;
                if (!move.has("action_type") || !move.has("raw_move")) continue;

                String actionType = move.get("action_type").getAsString();
                String rawMove = move.get("raw_move").getAsString();

                switch (actionType) {
                    case "move": {
                        if (unit.getType() == baseType || unit.getType() == barracksType) break;
                        Pattern pattern = Pattern.compile("\\(\\s*\\d+,\\s*\\d+\\):.*?move\\(\\(\\s*(\\d+),\\s*(\\d+)\\s*\\)\\)");
                        Matcher matcher = pattern.matcher(rawMove);
                        if (matcher.find()) {
                            move(unit, Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
                        }
                        break;
                    }
                    case "harvest": {
                        if (unit.getType() != workerType) break;
                        Pattern pattern = Pattern.compile("\\(\\s*\\d+,\\s*\\d+\\):.*?harvest\\(\\((\\d+),\\s*(\\d+)\\),\\s*\\((\\d+),\\s*(\\d+)\\)\\)");
                        Matcher matcher = pattern.matcher(rawMove);
                        if (matcher.find()) {
                            Unit resourceUnit = pgs.getUnitAt(Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
                            Unit baseUnit = pgs.getUnitAt(Integer.parseInt(matcher.group(3)), Integer.parseInt(matcher.group(4)));
                            if (resourceUnit != null && baseUnit != null) harvest(unit, resourceUnit, baseUnit);
                        }
                        break;
                    }
                    case "train": {
                        if (unit.getType() != baseType && unit.getType() != barracksType) break;
                        Pattern pattern = Pattern.compile("\\(\\s*\\d+,\\s*\\d+\\):.*?train\\(\\s*['\"]?(\\w+)['\"]?\\s*\\)");
                        Matcher matcher = pattern.matcher(rawMove);
                        if (matcher.find()) {
                            train(unit, stringToUnitType(matcher.group(1)));
                        }
                        break;
                    }
                    case "build": {
                        if (unit.getType() != workerType) break;
                        Pattern pattern = Pattern.compile("\\(\\s*\\d+,\\s*\\d+\\):.*?build\\(\\s*\\(\\s*(\\d+),\\s*(\\d+)\\s*\\),\\s*['\"]?(\\w+)['\"]?\\s*\\)");
                        Matcher matcher = pattern.matcher(rawMove);
                        if (matcher.find()) {
                            build(unit, stringToUnitType(matcher.group(3)), Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
                        }
                        break;
                    }
                    case "attack": {
                        Pattern pattern = Pattern.compile("\\(\\s*\\d+,\\s*\\d+\\):.*?attack\\(\\s*\\(\\s*(\\d+),\\s*(\\d+)\\s*\\)\\s*\\)");
                        Matcher matcher = pattern.matcher(rawMove);
                        if (matcher.find()) {
                            Unit enemyUnit = pgs.getUnitAt(Integer.parseInt(matcher.group(1)), Integer.parseInt(matcher.group(2)));
                            if (enemyUnit != null) attack(unit, enemyUnit);
                        }
                        break;
                    }
                    case "idle": {
                        idle(unit);
                        break;
                    }
                }
            } catch (Exception ex) {
                System.out.println("[LLM_DeepSeek] Error applying move: " + ex.getMessage());
            }
        }

        // Auto-attack if adjacent to enemy
        for (Unit u1 : pgs.getUnits()) {
            if (u1.getPlayer() != player || !u1.getType().canAttack) continue;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getPlayer() == player) continue;
                int d = Math.abs(u2.getX() - u1.getX()) + Math.abs(u2.getY() - u1.getY());
                if (d == 1 && getAbstractAction(u1) == null) {
                    attack(u1, u2);
                    break;
                }
            }
        }

        return translateActions(player, gs);
    }

    /**
     * Call DeepSeek API using OpenAI-compatible chat completions endpoint
     */
    public String prompt(String finalPrompt) {
        if (DEEPSEEK_API_KEY.isEmpty()) {
            System.err.println("[LLM_DeepSeek] ERROR: DEEPSEEK_API_KEY not set!");
            return "{\"thinking\":\"no_api_key\",\"moves\":[]}";
        }

        try {
            // Build request body (OpenAI-compatible format)
            JsonObject body = new JsonObject();
            body.addProperty("model", DEEPSEEK_MODEL);
            
            JsonArray messages = new JsonArray();
            JsonObject userMessage = new JsonObject();
            userMessage.addProperty("role", "user");
            userMessage.addProperty("content", finalPrompt);
            messages.add(userMessage);
            body.add("messages", messages);
            
            // Request JSON response
            JsonObject responseFormat = new JsonObject();
            responseFormat.addProperty("type", "json_object");
            body.add("response_format", responseFormat);
            
            // Optional parameters
            body.addProperty("temperature", 0.3);
            body.addProperty("max_tokens", 2048);

            URL url = new URL(DEEPSEEK_ENDPOINT);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setRequestProperty("Authorization", "Bearer " + DEEPSEEK_API_KEY);
            conn.setConnectTimeout(REQUEST_TIMEOUT);
            conn.setReadTimeout(REQUEST_TIMEOUT);
            conn.setDoOutput(true);

            promptTime = Instant.now();

            try (OutputStream os = conn.getOutputStream()) {
                os.write(body.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            }

            int code = conn.getResponseCode();
            InputStream is = (code == HttpURLConnection.HTTP_OK) ? conn.getInputStream() : conn.getErrorStream();

            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(is, java.nio.charset.StandardCharsets.UTF_8))) {
                for (String line; (line = br.readLine()) != null; ) sb.append(line);
            }

            responseTime = Instant.now();
            Latency = responseTime.toEpochMilli() - promptTime.toEpochMilli();

            if (code != HttpURLConnection.HTTP_OK) {
                System.err.println("[LLM_DeepSeek] API Error (" + code + "): " + sb);
                return "{\"thinking\":\"api_error\",\"moves\":[]}";
            }

            // Parse OpenAI-compatible response
            JsonObject responseJson = JsonParser.parseString(sb.toString()).getAsJsonObject();
            
            // Extract token usage if available
            if (responseJson.has("usage")) {
                JsonObject usage = responseJson.getAsJsonObject("usage");
                totalTokens = usage.has("total_tokens") ? usage.get("total_tokens").getAsInt() : 0;
            }
            
            // Extract message content
            JsonArray choices = responseJson.getAsJsonArray("choices");
            if (choices != null && choices.size() > 0) {
                JsonObject firstChoice = choices.get(0).getAsJsonObject();
                JsonObject message = firstChoice.getAsJsonObject("message");
                if (message != null && message.has("content")) {
                    return message.get("content").getAsString();
                }
            }

            System.err.println("[LLM_DeepSeek] Unexpected response format: " + sb);
            return "{\"thinking\":\"parse_error\",\"moves\":[]}";

        } catch (Exception e) {
            System.err.println("[LLM_DeepSeek] Exception: " + e.getMessage());
            e.printStackTrace();
            return "{\"thinking\":\"exception\",\"moves\":[]}";
        }
    }

    static String sanitizeModelJson(String s) {
        if (s == null) return "";
        s = s.trim();
        if (s.startsWith("```")) {
            int first = s.indexOf('\n');
            if (first >= 0) s = s.substring(first + 1);
            int close = s.lastIndexOf("```");
            if (close > 0) s = s.substring(0, close);
            s = s.trim();
        }
        int obj = s.indexOf('{');
        int arr = s.indexOf('[');
        int start = (obj == -1) ? arr : (arr == -1 ? obj : Math.min(obj, arr));
        if (start > 0) s = s.substring(start).trim();
        return s;
    }

    static JsonObject parseJsonStrictThenLenient(String raw) {
        String cleaned = sanitizeModelJson(raw);
        try {
            return JsonParser.parseString(cleaned).getAsJsonObject();
        } catch (JsonSyntaxException e) {
            try {
                com.google.gson.stream.JsonReader r = new com.google.gson.stream.JsonReader(new java.io.StringReader(cleaned));
                r.setLenient(true);
                return JsonParser.parseReader(r).getAsJsonObject();
            } catch (Exception e2) {
                System.err.println("[LLM_DeepSeek] JSON parse error: " + e.getMessage());
                return new JsonObject();
            }
        }
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("PathFinding", PathFinding.class, new AStarPathFinding()));
        return parameters;
    }

    private UnitType stringToUnitType(String string) {
        string = string.toLowerCase();
        switch (string) {
            case "worker": return workerType;
            case "light": return lightType;
            case "heavy": return heavyType;
            case "ranged": return rangedType;
            case "base": return baseType;
            case "barracks": return barracksType;
            default: return workerType;
        }
    }

    private String unitActionToString(UnitAction action) {
        if (action == null) return "idling";
        switch (action.getType()) {
            case UnitAction.TYPE_MOVE: return String.format("moving to (%d,%d)", action.getLocationX(), action.getLocationY());
            case UnitAction.TYPE_HARVEST: return String.format("harvesting from (%d,%d)", action.getLocationX(), action.getLocationY());
            case UnitAction.TYPE_RETURN: return String.format("returning resources to (%d,%d)", action.getLocationX(), action.getLocationY());
            case UnitAction.TYPE_PRODUCE: return String.format("producing unit at (%d,%d)", action.getLocationX(), action.getLocationY());
            case UnitAction.TYPE_ATTACK_LOCATION: return String.format("attacking location (%d,%d)", action.getLocationX(), action.getLocationY());
            case UnitAction.TYPE_NONE: return "idling";
            default: return "unknown action";
        }
    }
}