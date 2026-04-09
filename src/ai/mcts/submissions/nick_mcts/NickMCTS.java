package ai.mcts.submissions.nick_mcts;

import ai.abstraction.LightRush;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.RandomBiasedAI;
import ai.evaluation.LanchesterEvaluationFunction;
import ai.mcts.naivemcts.NaiveMCTS;
import java.util.ArrayList;
import java.util.List;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;
import rts.units.Unit;
import rts.PhysicalGameState;

public class NickMCTS extends NaiveMCTS {
    private UnitTypeTable utt;
    private final StrategyController controller = new StrategyController();
    private int lastUpdateFrame = -1;

    public NickMCTS(UnitTypeTable utt) {
        // Initialize super with the Safe wrapper to catch library bugs during MCTS simulations
        super(160, -1, 100, 25, 0.02f, 0.0f, 0.4f,
              new SafeLightRush(utt), 
              new MyEvaluation(utt, null), 
              true);
        
        this.utt = utt;
        
        // Link the controller to the evaluation function
        if (this.ef instanceof MyEvaluation) {
            ((MyEvaluation)this.ef).setController(this.controller);
        }
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        // Strategic update every 200 frames via LLM
        if (gs != null && gs.getTime() % 200 == 0 && gs.getTime() != lastUpdateFrame) {
            lastUpdateFrame = gs.getTime();
            controller.updateStrategy(gs, player);
        }
        return super.getAction(player, gs);
    }

    @Override
    public AI clone() {
        return new NickMCTS(this.utt);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}

/**
 * Safe Wrapper for LightRush.
 * Catches NullPointerExceptions in the underlying Harvest abstraction.
 */
class SafeLightRush extends LightRush {
    private AI fallback;

    public SafeLightRush(UnitTypeTable utt) {
        super(utt);
        // RandomBiasedAI is robust and doesn't use the buggy abstraction layer
        this.fallback = new RandomBiasedAI();
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) {
        try {
            return super.getAction(player, gs);
        } catch (Exception e) {
            // If Harvest crashes (target null), fallback to RandomBiasedAI 
            // so the MCTS simulation can actually progress.
            try {
                return fallback.getAction(player, gs);
            } catch (Exception e2) {
                return new PlayerAction(); 
            }
        }
    }

    @Override
    public AI clone() {
        return new SafeLightRush(this.utt);
    }
}

class StrategyController {
    private static final String OLLAMA_HOST = System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String OLLAMA_MODEL = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    
    public volatile float aggression = 1.2f;
    public volatile float threatWeight = 0.8f; 
    public volatile float resourceWeight = 0.15f;
    public volatile float offensiveWeight = 1.0f; 

    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofMillis(800))
            .build();
    private final Gson gson = new Gson();

    public void updateStrategy(GameState gs, int player) {
        if (gs == null) return;
        
        String stateSummary = summarizeState(gs, player);
        String prompt = "MicroRTS Context: " + stateSummary + 
                        ". Task: Break draw. Respond JSON ONLY: " +
                        "{\"agg\":float(0.5-3), \"thr\":float(0-5), \"res\":float(0-1), \"off\":float(0-2)}";

        Map<String, Object> payload = new HashMap<>();
        payload.put("model", OLLAMA_MODEL);
        payload.put("prompt", prompt);
        payload.put("stream", false);
        payload.put("format", "json");
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OLLAMA_HOST + "/api/generate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(payload)))
                .build();

        client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
              .thenApply(HttpResponse::body)
              .thenAccept(this::parseAndApply)
              .exceptionally(e -> null);
    }

    private String summarizeState(GameState gs, int player) {
        int[] my = countUnits(gs, player);
        int[] en = countUnits(gs, 1 - player);
        return String.format("Me[U:%d,B:%d,R:%d] En[U:%d,B:%d,R:%d]",
            (my[0]+my[1]), my[2], gs.getPlayer(player).getResources(),
            (en[0]+en[1]), en[2], gs.getPlayer(1-player).getResources());
    }

    private int[] countUnits(GameState gs, int p) {
        int w=0, c=0, b=0, br=0;
        for (Unit u : gs.getUnits()) {
            if (u != null && u.getPlayer() == p && u.getType() != null) {
                String n = u.getType().name;
                if (n.equals("Worker")) w++;
                else if (n.equals("Base")) b++;
                else if (n.equals("Barracks")) br++;
                else c++;
            }
        }
        return new int[]{w, c, b, br};
    }

    private void parseAndApply(String responseBody) {
        try {
            JsonObject topObj = JsonParser.parseString(responseBody).getAsJsonObject();
            String modelOutput = topObj.get("response").getAsString();
            JsonObject strategy = JsonParser.parseString(modelOutput).getAsJsonObject();

            if (strategy.has("agg")) this.aggression = strategy.get("agg").getAsFloat();
            if (strategy.has("thr")) this.threatWeight = strategy.get("thr").getAsFloat();
            if (strategy.has("res")) this.resourceWeight = strategy.get("res").getAsFloat();
            if (strategy.has("off")) this.offensiveWeight = strategy.get("off").getAsFloat();
        } catch (Exception e) { }
    }
}

class MyEvaluation extends LanchesterEvaluationFunction {
    private StrategyController sc;

    public MyEvaluation(UnitTypeTable utt, StrategyController sc) {
        this.sc = sc;
    }

    public void setController(StrategyController sc) { this.sc = sc; }

    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        if (gs == null) return 0;

        float agg = (sc != null) ? sc.aggression : 1.2f;
        float thr = (sc != null) ? sc.threatWeight : 0.8f;
        float res = (sc != null) ? sc.resourceWeight : 0.15f;
        float off = (sc != null) ? sc.offensiveWeight : 1.0f;

        float baseScore = super.evaluate(maxplayer, minplayer, gs);
        if (baseScore > 0) baseScore *= agg; 

        float offensiveBonus = calculateGlobalOffensiveBonus(maxplayer, gs) * off;
        float threatPenalty = calculateThreat(maxplayer, gs, 0.02f) * thr;
        
        float carryingBonus = 0;
        for (Unit u : gs.getUnits()) {
            if (u != null && u.getPlayer() == maxplayer && u.getResources() > 0) {
                carryingBonus += res;
            }
        }

        return baseScore - threatPenalty + carryingBonus + offensiveBonus;
    }

    private float calculateThreat(int player, GameState gs, float weight) {
        float threatPenalty = 0.0f;
        PhysicalGameState pgs = gs.getPhysicalGameState();
        for (Unit u : pgs.getUnits()) {
            if (u != null && u.getPlayer() == player && u.getType().name.equals("Base")) {
                for (Unit e : pgs.getUnits()) {
                    if (e != null && e.getPlayer() == 1 - player) {
                        int dist = Math.abs(u.getX() - e.getX()) + Math.abs(u.getY() - e.getY());
                        if (dist < 7) threatPenalty += (7 - dist) * weight;
                    }
                }
            }
        }
        return threatPenalty;
    }

    private float calculateGlobalOffensiveBonus(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int mapDim = pgs.getWidth() + pgs.getHeight();
        Unit enemyTarget = null;

        for (Unit u : pgs.getUnits()) {
            if (u != null && u.getPlayer() == 1 - player) {
                if (enemyTarget == null || u.getType().name.equals("Base")) enemyTarget = u;
            }
        }

        if (enemyTarget == null) return 0;

        float bonus = 0;
        for (Unit u : pgs.getUnits()) {
            if (u != null && u.getPlayer() == player && u.getType().canAttack) {
                int dist = Math.abs(u.getX() - enemyTarget.getX()) + Math.abs(u.getY() - enemyTarget.getY());
                bonus += (mapDim - dist) * 0.15f; 
            }
        }
        return bonus;
    }
}

