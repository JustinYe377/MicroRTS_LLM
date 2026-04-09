package ai.abstraction.submissions.jmurrllm;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.HybridLLMRush;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.List;

import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

import java.time.Instant;
import java.util.*;
import java.util.regex.*;
import java.io.*;
import java.net.*;
import com.google.gson.*;
import rts.PhysicalGameState;
import gui.PhysicalGameStatePanel;
import rts.UnitAction;
import rts.Player;
import rts.units.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.text.SimpleDateFormat;
import java.util.stream.Collectors;

public class JMurrAgent extends AbstractionLayerAI {

   protected UnitTypeTable utt;

   private CopyCat copyCatAI;
   private HybridLLMRush hybridRushAI;

   private static final int COPY_TIME = 200;

   public JMurrAgent(UnitTypeTable a_utt) {
      this(a_utt, new AStarPathFinding());
   }

   public JMurrAgent(UnitTypeTable a_utt, PathFinding a_pf) {
      super(a_pf);
      reset(a_utt);
   }

   @Override
   public void reset() {
      super.reset();
      if (copyCatAI != null) copyCatAI.reset();
      if (hybridRushAI != null) hybridRushAI.reset();
   }

   public void reset(UnitTypeTable a_utt) {
      copyCatAI = new CopyCat(a_utt, pf);
      hybridRushAI = new HybridLLMRush(a_utt, pf);
   }

   // Start by prompting an LLM to mirror the opponent's strategy. Then, after a set number of ticks, switch to the HybridLLMRush AI.
   @Override
   public PlayerAction getAction(int player, GameState gs) throws Exception {
      int currentTime = gs.getTime();

      if (currentTime < COPY_TIME) {
         return copyCatAI.getAction(player, gs);
      }

      return hybridRushAI.getAction(player, gs);
   }

   @Override
   public AI clone() {
      JMurrAgent clone = new JMurrAgent(utt, pf);
      return clone;
   }

   @Override
   public List<ParameterSpecification> getParameters() {
      List<ParameterSpecification> parameters = new ArrayList<>();
      parameters.add(new ParameterSpecification("PathFinding", PathFinding.class, new AStarPathFinding()));
      return parameters;
   }   
}





/**
 *
 * @author Mukesh
 */
final class CopyCat extends AbstractionLayerAI {

    /**
     * Static & non-static variables
     * connected to 2 classes  unitTypeTable & unitType
     */

    // NOTE: TESTING ONLY gmu3r2g need to remove it are better ways to handile it in github
    //static final String API_KEY =   "   "; /// remove it


    // gemini-1.5-flash (15 req/min)
    // gemini-2.0-flash (15 req/min)
    // gemini-2.5-flash
    //String MODEL = "gemini-2.5-flash";
    String fileName = "Response_format.csv";
    static final String ENDPOINT_URL = "https://generativelanguage.googleapis.com/v1beta/models/";
    static final JsonObject MOVE_RESPONSE_SCHEMA;
    // How often the LLM should act on the game state
    // NOTE: Fairness is now handled at the game level via ai_decision_interval in config.properties
    // This should be set to 1 so the LLM responds whenever the game asks
    private static final int LLM_INTERVAL =
            Integer.parseInt(System.getenv().getOrDefault("HYBRID_LLM_INTERVAL", "5"));
    LocalDateTime now = LocalDateTime.now();
    String timestamp = now.format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss"));

    JsonObject wrapper = new JsonObject();
    //String FilenameXXXten ="";
    Random r = new Random();
    protected UnitTypeTable utt; // different class
    UnitType resourceType;
    UnitType workerType;
    UnitType lightType;
    UnitType heavyType;
    UnitType rangedType;
    UnitType baseType;
    UnitType barracksType;
    int totalMovesGenerated = 0;
    int totalMovesAccepted = 0;
    int totalMovesRejected = 0;
    int promptstart =0;
    FileWriter writer;
    int jj =0;
    Instant promptTime;
    Instant responseTime;
    int TotalTokens = 0;
    int requestTokens =0;
    int responseTokens=0;
    long Latency =0;
    int totalTokens =0;
    String num_shot ="One-Shot";
    String aiName1= "";
    String aiName2="";
    String fileName01 ="";
    String value_TimestampandScore = "";
    private boolean logsInitialized = false;
    private boolean logsInitializedone = false;

    // ==== OLLAMA CONFIG ====
  //  static final String OLLAMA_HOST =
          //  System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
   // static String MODEL = "llama3.1:8b"; // gpt-oss  "llama3.1:8b" gpt-oss:20b
   // static final boolean OLLAMA_STREAM = false;
    static final String OLLAMA_FORMAT = "json";

    // Keep your file names/logs using MODEL:
    // String fileName = "Response_format.csv";

    /// --

    // ==== OLLAMA CONFIG ====
// Env takes precedence: export OLLAMA_MODEL=llama3.1:8b (or gpt-oss:20b, mistral:latest, ...)
    static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");

    static String MODEL =
            System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b"); // smollm2:135m is bad real bad
    // deepseek-r1:14b
     //

    // Models that usually honor "format":"json" (add more as you verify)
    static final Set<String> JSON_FRIENDLY = Set.of(
            "llama3.1:8b", "mistral:latest", "mistral:7b","llama3-gradient:8b","deepseek-r1:14b",
            "qwen2.5:7b", "qwen2.5:14b", "qwen3:latest", "qwen3-coder:30b", "deepseek-r1:8b"
    );

    // Optional switch to use chat endpoint (recommended)
    static final boolean USE_CHAT = true;

    // Keep stream false until you implement incremental parsing
    static final boolean OLLAMA_STREAM = false;

    String PROMPT = """
You are an AI playing a real-time strategy game. You control ALLY units only.

CRITICAL RULES:
1. You can ONLY command units marked as "Ally" - NEVER command "Enemy" or "Neutral" units
2. Each move MUST be a JSON object with ALL four required fields
3. The unit_position MUST match an Ally unit's position exactly from the game state

ACTIONS (use exact format):
- move((x, y)) - Move to position
- harvest((resource_x, resource_y), (base_x, base_y)) - Worker gathers resources
- train(unit_type) - Base trains: worker | Barracks trains: light, heavy, ranged
- build((x, y), building_type) - Worker builds: base, barracks
- attack((enemy_x, enemy_y)) - Attack enemy at position

REQUIRED JSON FORMAT:
{
  "thinking": "Brief strategy",
  "moves": [
    {
      "raw_move": "(x, y): unit_type action((args))",
      "unit_position": [x, y],
      "unit_type": "worker",
      "action_type": "harvest"
    }
  ]
}

EVERY MOVE OBJECT MUST HAVE ALL 4 FIELDS:
- raw_move: string like "(2, 0): worker harvest((0, 0), (2, 1))"
- unit_position: [x, y] array matching YOUR Ally unit position
- unit_type: "worker", "base", "barracks", "light", "heavy", or "ranged"
- action_type: "move", "harvest", "train", "build", or "attack"

VALID MOVE EXAMPLES:
{"raw_move": "(1, 1): worker harvest((0, 0), (2, 1))", "unit_position": [1, 1], "unit_type": "worker", "action_type": "harvest"}
{"raw_move": "(2, 1): base train(worker)", "unit_position": [2, 1], "unit_type": "base", "action_type": "train"}
{"raw_move": "(1, 1): worker attack((5, 6))", "unit_position": [1, 1], "unit_type": "worker", "action_type": "attack"}
{"raw_move": "(1, 1): worker move((3, 3))", "unit_position": [1, 1], "unit_type": "worker", "action_type": "move"}
{"raw_move": "(1, 1): worker build((3, 3), barracks)", "unit_position": [1, 1], "unit_type": "worker", "action_type": "build"}

STRATEGY: Copy the opponent strategy by mirroring their moves closely. If the opponent is building a unit, you build a unit of the same type of ally unit. Have the same number of ally workers harvesting for your base as the opponent does.
""";

    /**
     *
     *
     * Json retalted static block like structure and elements are over hear.
     * */
    static { // first priority when calling a class before main() & constructor
        MOVE_RESPONSE_SCHEMA = new JsonObject();


        String schemaJson = """
                {
                  "type": "object",
                  "properties": {
                    "thinking": {
                      "type": "string",
                      "description": "Plan out what moves you should take you can do multiple moves at a times"
                    },
                    "moves": {
                      "type": "array",
                      "items": {
                        "type": "object",
                        "properties": {
                          "raw_move": {
                            "type": "string"
                          },
                          "unit_position": {
                            "type": "array",
                            "items": {
                              "type": "integer"
                            },
                            "minItems": 2,
                            "maxItems": 2
                          },
                          "unit_type": {
                            "type": "string",
                            "enum": [
                              "worker",
                              "light",
                              "heavy",
                              "ranged",
                              "base",
                              "barracks"
                            ]
                          },
                          "action_type": {
                            "type": "string",
                            "enum": [
                              "move",
                              "train",
                              "build",
                              "harvest",
                              "attack"
                            ]
                          }
                        },
                        "required": [
                          "raw_move",
                          "unit_position",
                          "unit_type",
                          "action_type"
                        ]
                      }
                    }
                  },
                  "required": [
                    "moves",
                    "thinking"
                  ],
                  "propertyOrdering": [
                    "thinking",
                    "moves"
                  ]
                }
      """; // "thinking",

        JsonParser parser = new JsonParser();   /// if any format of json issue take a look and any modifications have a look
        JsonObject responseSchema = parser.parse(schemaJson).getAsJsonObject();
        MOVE_RESPONSE_SCHEMA.add("response_schema", responseSchema);

        System.out.println("responseSchema  270 :  gmu3r2g  -> "+responseSchema);

    }

    // is there any other way to give prompt in a better way to give Free to it ?


    /**
     * constructors
     */

    /**
     *
     * @param a_utt
     *
     */
    public CopyCat(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());

        //
        System.out.println(" in this 1 st nd mg546924 288   ----- >  Y / N ");
        String timestamp1 = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
       //  FilenameXXXten = "LLmGemini_"+timestamp1+".json";
        // this is also good i gess
        // String fileName01 = "Response_format_02"+".csv";
    }

    public CopyCat(UnitTypeTable a_utt,String aiName1, String aiName2){
        this(a_utt, new AStarPathFinding());
        if((aiName1 != null && aiName2 != null) && (!(aiName1.isEmpty())  || !(aiName2.isEmpty())) && logsInitializedone != true) {
            this.aiName1 = aiName1;
            this.aiName2 = aiName2;
            System.out.println("aiName1 - 23 " + aiName1);
            System.out.println("aiName2 - 23" + aiName2);
            logsInitializedone = true;
        }
    }

    /**
     *
     * @param a_utt = ?
     * @param a_pf = ?
     */
    public CopyCat(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf); //
        System.out.println(" in this 2 nd mg546924 180 "+ a_utt);
        reset(a_utt); // method call
    }


    /**
     *
     */
    public String createFileName() throws Exception {
        System.out.println("aiName1 "+this.aiName1+" "+this.aiName2+" "+aiName2+" "+aiName1);
        return "Response"+timestamp+"_"+this.aiName1+"_"+num_shot+"_"+this.aiName2+"_"+MODEL+".csv";
    }

    /**
     * Methods
     */

    /**
     * @method :  logEndGameMetrics
     *
     * head a issue check it will work properly are not write now we are adding json responses
     * this format is bad need to update
     *
     */
    private void logEndGameMetrics() {
        System.out.println(" 158 check gmu3r2g");

        JsonObject metrics = new JsonObject();
        metrics.addProperty("moves_generated", totalMovesGenerated);
        metrics.addProperty("moves_accepted", totalMovesAccepted);
        metrics.addProperty("moves_rejected", totalMovesRejected);

        JsonObject totals = new JsonObject();
        totals.addProperty("total_moves_generated", totalMovesGenerated);
        totals.addProperty("total_moves_accepted", totalMovesAccepted);
        totals.addProperty("total_moves_rejected", totalMovesRejected);

        JsonObject wrapper = new JsonObject();
        wrapper.add("player_final_stats", metrics);
        wrapper.add("game_metrics", totals);


        String timestamp = java.time.LocalDateTime.now()
                .format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSSSSS"));
        wrapper.addProperty("end_time", timestamp);

        try (FileWriter writer = new FileWriter("game_summary.json", true)) {
            System.out.println(" am i in : 245 LLM _gemini ");
            writer.write(new GsonBuilder().setPrettyPrinting().create().toJson(wrapper));
            writer.write(System.lineSeparator());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }





    private void initLogsIfNeeded() {
        System.out.println(" initLogsIfNeeded  ~~~~  ~~ ~ ~");
        if (logsInitialized) return;

        // Fallback labels if names weren’t injected
        String a = (aiName1 == null || aiName1.isEmpty()) ? "LLM_Gemini" : aiName1;
        String b = (aiName2 == null || aiName2.isEmpty()) ? "RandomBiasedAI" : aiName2;


        // Build names once
        String ts = new java.text.SimpleDateFormat("yyyy-MM-dd_HH-mm-ss").format(new java.util.Date());
        fileName01 = "Response" + ts + "_" + a + "_" + num_shot + "_" + b + "_" + MODEL + ".csv";
       // FilenameXXXten = "LLM_Gemini_" + ts + ".json";

        try (FileWriter writer = new FileWriter(fileName01)) {
            writer.append("Thinking,Moves,Feature locations,Request Time, Request / Prompt Tokens, response Time, Response Tokens, Latency(milliseconds),Total Tokens,Score_in_every_run\n");
        } catch (IOException e) {
            e.printStackTrace();
        }

        logsInitialized = true;
    }





    /**
     *
     * reset function is reseting they Time budget & Iteration budget
     * going to they reset of abstract layer ai to reset are clearing they data from
     * the hashMap   HashMap<Unit, AbstractAction>
     *
     *
     * TIME_BUDGET  in  aiwithComputationbudget
     * ITERATIONS_BUDGET  in  aiwithComputationbudget
     */
    public void reset() {
        super.reset();
        TIME_BUDGET = -1;
        ITERATIONS_BUDGET = -1;
    }

    /**
     *
     * @param a_utt
     */
    public void reset(UnitTypeTable a_utt)
    {
        utt = a_utt;
        resourceType = utt.getUnitType("Resource");
        workerType = utt.getUnitType("Worker");
        lightType = utt.getUnitType("Light");
        heavyType = utt.getUnitType("Heavy");
        rangedType = utt.getUnitType("Ranged");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
    }

    /**
     * utt passing with a_utt
     * pf from abstract layer ai
     *
     *
     * @return
     */

    @Override
    public AI clone() {
        return new CopyCat(utt, pf);
    }


    /**
     *
     * @param player ID of the player to move. Use it to check whether units are yours or enemy's
     * @param gs the game state where the action should be performed
     * @return
     */



    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        // make sure filenames/log headers exist
        initLogsIfNeeded();

        String finalPrompt;

        System.out.println(" [CopyCat.getAction] start ");

        // If we're NOT on an LLM turn, just keep executing the abstract actions already assigned
        if (gs.getTime() % LLM_INTERVAL != 0) {
            if (gs.gameover()) {
                logEndGameMetrics();
            }

            PlayerAction pa = translateActions(player, gs);
            System.out.println("🎯 (LLM_INTERVAL skip) translateActions() generated PlayerAction:");
            System.out.println(pa);
            return pa;
        }

        // ===== Gather game context =====
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int width = pgs.getWidth();
        int height = pgs.getHeight();
        Player p = gs.getPlayer(player);

        ArrayList<String> features = new ArrayList<>();
        int maxActions = 0;

        // Build feature list for the prompt and help us count how many units we can legally command
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                maxActions++;
            }

            String unitStats;
            UnitAction unitAction = gs.getUnitAction(u);
            String unitActionString = unitActionToString(unitAction);

            String unitType;
            if (u.getType() == resourceType) {
                unitType = "Resource Node";
                unitStats = "{resources=" + u.getResources() + "}";
            } else if (u.getType() == baseType) {
                unitType = "Base Unit";
                unitStats = "{resources=" + p.getResources() +
                        ", current_action=\"" + unitActionString +
                        "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == barracksType) {
                unitType = "Barracks Unit";
                unitStats = "{current_action=\"" + unitActionString +
                        "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == workerType) {
                unitType = "Worker Unit";
                unitStats = "{current_action=\"" + unitActionString +
                        "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == lightType) {
                unitType = "Light Unit";
                unitStats = "{current_action=\"" + unitActionString +
                        "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == heavyType) {
                unitType = "Heavy Unit";
                unitStats = "{current_action=\"" + unitActionString +
                        "\", HP=" + u.getHitPoints() + "}";
            } else if (u.getType() == rangedType) {
                unitType = "Ranged Unit";
                unitStats = "{current_action=\"" + unitActionString +
                        "\", HP=" + u.getHitPoints() + "}";
            } else {
                unitType = "Unknown";
                unitStats = "{}";
            }

            String unitPos = "(" + u.getX() + ", " + u.getY() + ")";
            String team = (u.getPlayer() == player) ? "Ally" :
                    (u.getType() == resourceType ? "Neutral" : "Enemy");

            features.add(unitPos + " " + team + " " + unitType + " " + unitStats);
        }

        // Map summary for the LLM
        String mapPrompt         = "Map size: " + width + "x" + height;
        String turnPrompt        = "Turn: " + gs.getTime() + "/" + 5000;
        String maxActionsPrompt  = "Max actions: " + maxActions;
        value_TimestampandScore  = PhysicalGameStatePanel.info1;

        String featuresPrompt = "Feature locations:\n" + String.join("\n", features);

        // Also build a JSON-ish array string version of features for CSV logging later
        String[] lines = featuresPrompt.split("\n");
        String arrayFormat = Arrays.stream(lines)
                .map(s -> "\"" + s.replace("\"", "\\\"") + "\"")
                .collect(Collectors.joining(", ", "[", "]"));

        // Final LLM prompt
        finalPrompt = PROMPT + "\n\n" +
                mapPrompt + "\n" +
                turnPrompt + "\n" +
                maxActionsPrompt + "\n\n" +
                featuresPrompt + "\n";

       // System.out.println("=== Prompt to LLM ===");
       // System.out.println(finalPrompt);
       // System.out.println("=====================");LLM returned String as expected

        // ===== Call the model (Ollama in your current version) =====
        String response = prompt(finalPrompt);
        System.out.println("=== Raw LLM Response ===");
        System.out.println(response);
        System.out.println("========================");

        if (response instanceof String) {
            System.out.println("LLM returned String as expected.");
        }
        // ===== Parse model JSON safely & log pretty copy =====
        JsonObject jsonResponse = parseJsonStrictThenLenient(response);

        Gson gson = new GsonBuilder().setPrettyPrinting().create();
        String prettyJson = gson.toJson(jsonResponse);

        /**
        try (FileWriter file = new FileWriter(FilenameXXXten, true)) {
            String tsNow = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date());
            file.write("[" + tsNow + "]\n");
            file.write(prettyJson);
            file.write(System.lineSeparator());
        } catch (IOException e) {
            e.printStackTrace();
        } */

        // ===== Extract "moves" array from model output =====
        JsonArray moveElements = jsonResponse.getAsJsonArray("moves");

        if (moveElements == null || moveElements.size() == 0) {
            System.out.println("[LLM] No moves[] in response. Falling back to translateActions.");
            if (gs.gameover()) {
                logEndGameMetrics();
            }
            PlayerAction fallbackPA = translateActions(player, gs);
            System.out.println("🎯 translateActions() (no LLM moves) =>");
            System.out.println(fallbackPA);
            return fallbackPA;
        }

        // ===== Try to apply each move from the LLM safely =====
        for (JsonElement moveElement : moveElements) {
            try {
                if (!moveElement.isJsonObject()) {
                    System.out.println("[LLM] skipping non-object move element: " + moveElement);
                    continue;
                }

                JsonObject move = moveElement.getAsJsonObject();

                // --- Validate unit_position ---
                if (!move.has("unit_position") || !move.get("unit_position").isJsonArray()) {
                    System.out.println("[LLM] skipping move, missing unit_position: " + move);
                    continue;
                }

                JsonArray unitPosition = move.getAsJsonArray("unit_position");
                if (unitPosition == null ||
                        unitPosition.size() < 2 ||
                        unitPosition.get(0).isJsonNull() ||
                        unitPosition.get(1).isJsonNull()) {

                    System.out.println("[LLM] skipping move, bad unit_position: " + unitPosition);
                    continue;
                }

                int unitX = unitPosition.get(0).getAsInt();
                int unitY = unitPosition.get(1).getAsInt();

                // --- Look up the unit in the game ---
                Unit unit = pgs.getUnitAt(unitX, unitY);
                if (unit == null) {
                    System.out.println("❌ No unit at (" + unitX + ", " + unitY + ") - skipping move.");
                    continue;
                }

                // cannot command enemy / neutral units
                if (unit.getPlayer() != player) {
                    System.out.println("  ----->   Can't command non-owned unit at (" + unitX + ", " + unitY + ") - skipping.");
                    continue;
                }

                // --- Required action fields ---
                if (!move.has("action_type") || !move.has("raw_move")) {
                    System.out.println("[LLM] skipping move, missing action_type/raw_move: " + move);
                    continue;
                }

                String actionType = move.get("action_type").getAsString();
                String rawMove    = move.get("raw_move").getAsString();
                String unitType   = move.has("unit_type")
                        ? move.get("unit_type").getAsString()
                        : "unknown";

                System.out.println(
                        " Applying LLM move: " + rawMove +
                                " | action_type=" + actionType +
                                " | unit=(" + unitX + "," + unitY + ") type=" + unitType
                );

                // We'll parse text like "(2,1): worker move((3,1))"
                // using regex per action type, then call the abstraction-layer helpers
                switch (actionType) {
                    case "move": {
                        // structures can't move
                        if (unit.getType() == baseType || unit.getType() == barracksType) {
                            System.out.println("'move' failed: structure at ("+unitX+", "+unitY+")");
                            break;
                        }

                        Pattern pattern = Pattern.compile(
                                "\\(\\s*\\d+,\\s*\\d+\\):.*?move\\(\\(\\s*(\\d+),\\s*(\\d+)\\s*\\)\\)"
                        );
                        Matcher matcher = pattern.matcher(rawMove);

                        if (matcher.find()) {
                            int targetX = Integer.parseInt(matcher.group(1));
                            int targetY = Integer.parseInt(matcher.group(2));
                            move(unit, targetX, targetY);
                        } else {
                            System.out.println("'move' regex no match for: " + rawMove);
                        }

                        if (gs.gameover()) logEndGameMetrics();
                        break;
                    }

                    case "harvest": {
                        // workers only
                        if (unit.getType() != workerType) {
                            System.out.println("'harvest' failed: non-worker at ("+unitX+", "+unitY+")");
                            break;
                        }

                        Pattern pattern = Pattern.compile(
                                "\\(\\s*\\d+,\\s*\\d+\\):.*?harvest\\(\\((\\d+),\\s*(\\d+)\\),\\s*\\((\\d+),\\s*(\\d+)\\)\\)"
                        );
                        Matcher matcher = pattern.matcher(rawMove);

                        if (matcher.find()) {
                            int resourceX = Integer.parseInt(matcher.group(1));
                            int resourceY = Integer.parseInt(matcher.group(2));
                            int baseX     = Integer.parseInt(matcher.group(3));
                            int baseY     = Integer.parseInt(matcher.group(4));

                            Unit resourceUnit = pgs.getUnitAt(resourceX, resourceY);
                            Unit baseUnit     = pgs.getUnitAt(baseX, baseY);

                            if (resourceUnit != null && baseUnit != null) {
                                harvest(unit, resourceUnit, baseUnit);
                            } else {
                                System.out.println("'harvest' failed: couldn't resolve resource/base units");
                            }
                        } else {
                            System.out.println("'harvest' regex no match for: " + rawMove);
                        }

                        if (gs.gameover()) logEndGameMetrics();
                        break;
                    }

                    case "train": {
                        // only base or barracks can train
                        if ((unit.getType() != baseType) && (unit.getType() != barracksType)) {
                            System.out.println("'train' failed: not base/barracks at ("+unitX+", "+unitY+")");
                            break;
                        }

                        Pattern pattern = Pattern.compile(
                                "\\(\\s*\\d+,\\s*\\d+\\):.*?train\\(\\s*['\"]?(\\w+)['\"]?\\s*\\)"
                        );
                        Matcher matcher = pattern.matcher(rawMove);

                        if (matcher.find()) {
                            String stringTrainUnitType = matcher.group(1);
                            UnitType trainUnitType = stringToUnitType(stringTrainUnitType);
                            train(unit, trainUnitType);
                        } else {
                            System.out.println("'train' regex no match for: " + rawMove);
                        }

                        if (gs.gameover()) logEndGameMetrics();
                        break;
                    }

                    case "build": {
                        // only workers can build
                        if (unit.getType() != workerType) {
                            System.out.println("'build' failed: non-worker at ("+unitX+", "+unitY+")");
                            break;
                        }

                        Pattern pattern = Pattern.compile(
                                "\\(\\s*\\d+,\\s*\\d+\\):.*?build\\(\\s*\\(\\s*(\\d+),\\s*(\\d+)\\s*\\),\\s*['\"]?(\\w+)['\"]?\\s*\\)"
                        );
                        Matcher matcher = pattern.matcher(rawMove);

                        if (matcher.find()) {
                            int buildX = Integer.parseInt(matcher.group(1));
                            int buildY = Integer.parseInt(matcher.group(2));
                            String stringBuildUnitType = matcher.group(3);
                            UnitType unitBuildType = stringToUnitType(stringBuildUnitType);
                            build(unit, unitBuildType, buildX, buildY);
                        } else {
                            System.out.println("'build' regex no match for: " + rawMove);
                        }

                        if (gs.gameover()) logEndGameMetrics();
                        break;
                    }

                    case "attack": {
                        Pattern pattern = Pattern.compile(
                                "\\(\\s*\\d+,\\s*\\d+\\):.*?attack\\(\\s*\\(\\s*(\\d+),\\s*(\\d+)\\s*\\)\\s*\\)"
                        );
                        Matcher matcher = pattern.matcher(rawMove);

                        if (matcher.find()) {
                            int enemyX = Integer.parseInt(matcher.group(1));
                            int enemyY = Integer.parseInt(matcher.group(2));
                            Unit enemyUnit = pgs.getUnitAt(enemyX, enemyY);

                            if (enemyUnit != null) {
                                attack(unit, enemyUnit);
                            } else {
                                System.out.println("'attack' failed: no enemy at ("+enemyX+","+enemyY+")");
                            }
                        } else {
                            System.out.println("'attack' regex no match for: " + rawMove);
                        }

                        if (gs.gameover()) logEndGameMetrics();
                        break;
                    }

                    case "idle": {
                        idle(unit);
                        if (gs.gameover()) logEndGameMetrics();
                        break;
                    }

                    default: {
                        System.out.println("Unknown action_type '" + actionType + "', skipping.");
                        if (gs.gameover()) logEndGameMetrics();
                        break;
                    }
                }

            } catch (Exception ex) {
                // CRITICAL: swallow bad move so AI doesn't crash the whole game
                System.out.println("[LLM] Exception applying a move: " + ex.getMessage());
                ex.printStackTrace();
                // continue to next move
            }
        }

        // ===== Auto-defense override (only if unit has no current abstract action) =====
        // If an allied combat unit is standing next to an enemy, let it attack,
        // but DON'T override if LLM already gave that unit an action.
        for (Unit u1 : pgs.getUnits()) {
            // only consider our units that can attack
            if (u1.getPlayer() != player || !u1.getType().canAttack) {
                continue;
            }

            Unit closestEnemy = null;
            int closestDistance = 0;

            for (Unit u2 : pgs.getUnits()) {
                if (u2.getPlayer() == player) continue; // skip allies

                int d = Math.abs(u2.getX() - u1.getX()) + Math.abs(u2.getY() - u1.getY());
                if (closestEnemy == null || d < closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }

            if (closestEnemy != null && closestDistance == 1) {
                if (getAbstractAction(u1) == null) {
                    System.out.println("Auto-attack: " + u1 + " -> " + closestEnemy);
                    attack(u1, closestEnemy);
                } else {
                    System.out.println("⚠️ Skipping auto-override for " + u1 +
                            " (already has LLM action)");
                }
            }
        }

        // ===== Logging & CSV append =====
        totalMovesGenerated++;
        totalMovesAccepted++;

        System.out.println("gs.gameover() = " + gs.gameover());
        Player p0 = gs.getPlayer(0);
        Player p1 = gs.getPlayer(1);
        int currentTime = gs.getTime();

        System.out.println("Running getAction for Player: " + player);
        System.out.println(" current time " + currentTime + " p0 " + p0 + " p1 " + p1);

        String combinestring = "T : " + currentTime + "," + p0 + "," + p1;

        try (FileWriter writer = new FileWriter(fileName01, true)) {
            if (response.indexOf("\"thinking\"") != -1 &&
                    response.indexOf("\"moves\"")    != -1) {

                String beforeMoves = response.substring(
                        response.indexOf("\"thinking\"") + 11,
                        response.indexOf("\"moves\"")
                );
                String fromMoves   = response.substring(
                        response.indexOf("\"moves\"")
                );

                String csvSafeThinking   = escapeForCSV(beforeMoves);
                String csvSafeMoves      = escapeForCSV(fromMoves);
                String csvSafeFeatures   = escapeForCSV(arrayFormat);
                String csvSafeTimestamp  = escapeForCSV(combinestring);
                String csvSafeScoreBlock = escapeForCSV(value_TimestampandScore);

                writer.append(csvSafeThinking).append(",")
                        .append(csvSafeMoves).append(",")
                        .append(csvSafeFeatures).append(",")
                        .append(promptTime != null ? promptTime.toString() : "")
                        .append(",")
                        .append(String.valueOf(requestTokens)).append(",")
                        .append(responseTime != null ? responseTime.toString() : "")
                        .append(",")
                        .append(String.valueOf(responseTokens)).append(",")
                        .append(String.valueOf(Latency)).append(",")
                        .append(String.valueOf(totalTokens)).append(",")
                        .append(csvSafeScoreBlock)
                        .append("\n");
            } else {
                System.out.println("CSV logging: keywords not found in response.");
            }
        } catch (IOException e) {
            e.printStackTrace();
            System.err.println("error writing CSV row: " + e.getMessage());
        }

        System.out.printf(
                "T: %d, P0: %d (%s), P1: %d (%s)%n",
                currentTime,
                p0.getID(), p0.getResources(),
                p1.getID(), p1.getResources()
        );

        if (gs.gameover()) {
            logEndGameMetrics();
        }

        // ===== Return the final PlayerAction for this frame =====
        PlayerAction pa = translateActions(player, gs);
        System.out.println("🎯 Final translateActions() PlayerAction:");
        System.out.println(pa);
        return pa;
    }

    /**
     *
     * @param value = "String format that contain a lot of , . \n "
     *
     * @return = " plain string which is not goint to split in any 2 are 3 individual bxes in excel are google sheets "
     */
    public static String escapeForCSV(String value) {
        if (value == null) return "\"\"";
        // Step 1: Escape internal quotes
        String escaped = value.replace("\"", "\"\"");
        // Step 2: Escape newlines (convert to literal \n so Excel doesn’t break cell)
        escaped = escaped.replace("\n", "\\n").replace("\r", "\\r");
        // Step 3: Wrap in quotes
        return "\"" + escaped + "\"";
    }


    static String sanitizeModelJson(String s) {
        if (s == null) return "";
        s = s.trim();

        // Strip Markdown code fences if model adds them
        if (s.startsWith("```")) {
            int first = s.indexOf('\n');
            if (first >= 0) s = s.substring(first + 1);
            int close = s.lastIndexOf("```");
            if (close > 0) s = s.substring(0, close);
            s = s.trim();
        }

        // If the model prepended text, jump to first JSON object/array
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
                com.google.gson.stream.JsonReader r =
                        new com.google.gson.stream.JsonReader(new java.io.StringReader(cleaned));
                r.setLenient(true);
                return JsonParser.parseReader(r).getAsJsonObject();
            } catch (Exception e2) {
                throw e; // bubble up the original strict error
            }
        }
    }



    public String prompt(String finalPrompt) {
        try {
            // Build Ollama request body
            JsonObject body = new JsonObject();
            body.addProperty("model", MODEL);
            // Prepend /no_think to disable qwen3 thinking mode for faster responses
            body.addProperty("prompt", "/no_think " + finalPrompt);
            body.addProperty("stream", OLLAMA_STREAM);   // false -> single JSON
            body.addProperty("format", OLLAMA_FORMAT);   // "json" -> enforce JSON output

            // Optional generation knobs (tweak as needed):
            // body.addProperty("temperature", 0.4);
            // body.addProperty("num_ctx", 8192);

            URL url = new URL(OLLAMA_HOST + "/api/generate");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);

            // record request time for latency
            promptTime = Instant.now();

            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = body.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
                os.write(input);
            }

            int code = conn.getResponseCode();
            InputStream is = (code == HttpURLConnection.HTTP_OK)
                    ? conn.getInputStream()
                    : conn.getErrorStream();

            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(is, java.nio.charset.StandardCharsets.UTF_8))) {
                for (String line; (line = br.readLine()) != null; ) sb.append(line);
            }

            responseTime = Instant.now();
            Latency = responseTime.toEpochMilli() - promptTime.toEpochMilli();

            if (code != HttpURLConnection.HTTP_OK) {
                System.err.println("❌ Ollama error (" + code + "): " + sb);
                return "{\"thinking\":\"error\",\"moves\":[]}";
            }

            // Ollama /api/generate returns JSON like:
            // {"model":"...","created_at":"...","response":"...TEXT...","done":true,...}
            // Note: qwen3 "thinking" models put output in "thinking" field instead of "response"
            JsonObject top = JsonParser.parseString(sb.toString()).getAsJsonObject();

            String modelText = "";

            // First try "response" field (standard models like llama3.1)
            if (top.has("response") && !top.get("response").getAsString().isEmpty()) {
                modelText = top.get("response").getAsString();
            }
            // Fall back to "thinking" field (qwen3 thinking models)
            else if (top.has("thinking") && !top.get("thinking").isJsonNull()) {
                modelText = top.get("thinking").getAsString();
                System.out.println("📝 Using 'thinking' field from qwen3 model");
            }
            else {
                System.err.println("❌ Unexpected Ollama payload (no response or thinking): " + sb);
                return "{\"thinking\":\"invalid_response\",\"moves\":[]}";
            }

            // (Optional) log the raw text for debugging
            // System.out.println("OLLAMA raw response:\n" + modelText);

            // Return the text **as-is** — your caller will parse to JSON later
            return modelText;

        } catch (Exception e) {
            e.printStackTrace();
            return "{\"thinking\":\"exception\",\"moves\":[]}";
        }
    }



    @Override
    public List<ParameterSpecification> getParameters()
    {
        List<ParameterSpecification> parameters = new ArrayList<>();

        parameters.add(new ParameterSpecification("PathFinding", PathFinding.class, new AStarPathFinding()));

        return parameters;
    }


    private UnitType stringToUnitType(String string) {
        string = string.toLowerCase();
        switch (string) {
            case "worker":
                return workerType;
            case "light":
                return lightType;
            case "heavy":
                return heavyType;
            case "ranged":
                return rangedType;
            case "base":
                return baseType;
            case "barracks":
                return barracksType;
            default:
                System.out.println("Unknown unit type: " + string);
                return workerType;
        }
    }

    private String unitActionToString(UnitAction action) {
        if (action == null) { return "idling"; }

        String description;
        switch (action.getType()) {
            case UnitAction.TYPE_MOVE:
                description = String.format("moving to (%d,%d)", action.getLocationX(), action.getLocationY());
                break;
            case UnitAction.TYPE_HARVEST:
                description = String.format("harvesting from (%d,%d)", action.getLocationX(), action.getLocationY());
                break;
            case UnitAction.TYPE_RETURN:
                description = String.format("returning resources to (%d,%d)", action.getLocationX(), action.getLocationY());
                break;
            case UnitAction.TYPE_PRODUCE:
                description = String.format("producing unit at (%d,%d)", action.getLocationX(), action.getLocationY());
                break;
            case UnitAction.TYPE_ATTACK_LOCATION:
                description = String.format("attacking location (%d,%d)", action.getLocationX(), action.getLocationY());
                break;
            case UnitAction.TYPE_NONE:
                description = "idling";
                break;
            default:
                description = "unknown action";
                break;
        }
        return description;
    }

}