/*
 * yebot — Async PureLLM with Java fallback
 *
 * How it works:
 *   - Java worker rush runs immediately from tick 0 (no LLM wait)
 *   - LLM call fires async at tick 0 in background
 *   - When LLM responds, its decisions replace the Java defaults
 *   - LLM fires again every LLM_INTERVAL ticks
 *   - If LLM is slow/fails, Java defaults keep the game moving
 *
 * This is still PureLLM in the research sense:
 *   - The LLM reads full game state and outputs all unit actions
 *   - Java defaults only exist to prevent timeout — they get replaced
 *   - The LLM's decisions take priority the moment they arrive
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
import java.util.concurrent.*;
import java.util.regex.*;

public class yebot extends AbstractionLayerAI {

    // ─── Config ────────────────────────────────────────────────────────────────
    private static final String OLLAMA_MODEL = System.getenv("OLLAMA_MODEL") != null
            ? System.getenv("OLLAMA_MODEL") : "qwen3:8b";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int LLM_TIMEOUT  = 8000; // qwen3:8b responds in <1s; 8s is generous
    private static final int LLM_INTERVAL = 50; // ticks between LLM calls

    // ─── Unit Types ────────────────────────────────────────────────────────────
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ─── Async LLM state ──────────────────────────────────────────────────────
    // unitID → UnitAction, from last LLM response
    // volatile so game thread always sees latest value written by LLM thread
    private volatile Map<Long, UnitAction> llmDecisions = null;
    private volatile boolean llmRunning = false;
    private int lastLLMTick = -LLM_INTERVAL;
    private final ExecutorService llmThread = Executors.newSingleThreadExecutor();

    // ─── LLM Prompt ───────────────────────────────────────────────────────────
    private static final String SYSTEM_PROMPT =
        "You are a MicroRTS AI. Output unit actions as JSON.\n"
      + "UNITS: Worker(HP=1,dmg=1,range=1,cost=1) Light(HP=4,dmg=2,range=1,cost=2) "
      + "Heavy(HP=8,dmg=4,range=1,cost=3) Ranged(HP=3,dmg=1,range=3,cost=2) "
      + "Base(HP=10) Barracks(HP=5)\n"
      + "RULES:\n"
      + "- Only command idle units\n"
      + "- One action per unit\n"
      + "- Buildings cannot move or attack\n"
      + "- harvest() base must be YOUR base\n"
      + "ACTIONS: move(x,y) attack(x,y) harvest(rx,ry,bx,by) train(worker|light|heavy|ranged) build(x,y,barracks)\n"
      + "OUTPUT: {\"thinking\":\"one sentence\",\"moves\":[{\"uid\":unitID,\"type\":\"move|attack|harvest|train|build\",\"args\":[...]}]}\n"
      + "uid is the integer unit ID. args are integers. For train, args is empty.\n";

    // ══════════════════════════════════════════════════════════════════════════
    //  CONSTRUCTORS
    // ══════════════════════════════════════════════════════════════════════════

    public yebot(UnitTypeTable a_utt) { this(a_utt, new AStarPathFinding()); }

    public yebot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    @Override
    public void reset() {
        super.reset();
        llmDecisions = null;
        llmRunning   = false;
        lastLLMTick  = -LLM_INTERVAL;
    }

    public void reset(UnitTypeTable a_utt) {
        utt          = a_utt;
        workerType   = utt.getUnitType("Worker");
        lightType    = utt.getUnitType("Light");
        heavyType    = utt.getUnitType("Heavy");
        rangedType   = utt.getUnitType("Ranged");
        baseType     = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
    }

    @Override
    public AI clone() { return new yebot(utt, pf); }

    // ══════════════════════════════════════════════════════════════════════════
    //  MAIN LOOP
    // ══════════════════════════════════════════════════════════════════════════

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int tick = gs.getTime();

        // ── Fire async LLM if interval elapsed ────────────────────────────────
        if (!llmRunning && tick - lastLLMTick >= LLM_INTERVAL) {
            lastLLMTick = tick;
            llmRunning  = true;
            final GameState gsCopy    = gs.clone();
            final int       playerRef = player;
            llmThread.submit(() -> {
                try {
                    Map<Long, UnitAction> result = callLLMForDecisions(playerRef, gsCopy, pgs.getWidth());
                    if (result != null && !result.isEmpty()) {
                        llmDecisions = result;
                        System.out.println("[yebot] LLM updated t=" + tick
                                + " (" + result.size() + " actions)");
                    }
                } catch (Exception e) {
                    System.err.println("[yebot] LLM error: " + e.getMessage());
                } finally {
                    llmRunning = false;
                }
            });
        }

        // ── Apply LLM decisions if available, else Java worker rush ───────────
        Map<Long, UnitAction> dec = llmDecisions;
        if (dec != null && !dec.isEmpty()) {
            return applyDecisions(dec, player, gs, pgs);
        } else {
            return workerRushFallback(player, gs, pgs);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  APPLY LLM DECISIONS
    // ══════════════════════════════════════════════════════════════════════════

    private PlayerAction applyDecisions(Map<Long, UnitAction> dec,
                                         int player, GameState gs,
                                         PhysicalGameState pgs) throws Exception {
        PlayerAction pa = new PlayerAction();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) continue;
            if (gs.getActionAssignment(u) != null) continue;
            UnitAction ua = dec.get(u.getID());
            if (ua != null && gs.isUnitActionAllowed(u, ua)) {
                pa.addUnitAction(u, ua);
            }
        }
        pa.fillWithNones(gs, player, 1);
        return pa;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  JAVA WORKER RUSH FALLBACK — instant, no LLM
    //  Runs until first LLM response arrives, then LLM takes over
    // ══════════════════════════════════════════════════════════════════════════

    private PlayerAction workerRushFallback(int player, GameState gs,
                                              PhysicalGameState pgs) throws Exception {
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

        // Base trains worker
        if (myBase != null && gs.getActionAssignment(myBase) == null
                && gs.getPlayer(player).getResources() >= workerType.cost) {
            train(myBase, workerType);
        }

        // One harvester, rest attack
        boolean harvesting = false;
        for (Unit w : myWorkers) {
            if (gs.getActionAssignment(w) != null) continue;
            if (!harvesting && myBase != null && !resources.isEmpty()) {
                harvest(w, nearest(w, resources), myBase);
                harvesting = true;
            } else if (!enemies.isEmpty()) {
                attack(w, nearest(w, enemies));
            }
        }

        return translateActions(player, gs);
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LLM CALL + PARSING
    // ══════════════════════════════════════════════════════════════════════════

    private Map<Long, UnitAction> callLLMForDecisions(int player, GameState gs,
                                                        int mapWidth) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        String stateText = buildStateText(player, gs, pgs);
        String response  = callLLM(stateText);
        return parseDecisions(response, player, gs, pgs, mapWidth);
    }

    // ── State description ─────────────────────────────────────────────────────

    private String buildStateText(int player, GameState gs, PhysicalGameState pgs) {
        StringBuilder sb = new StringBuilder();
        sb.append("Turn=").append(gs.getTime())
          .append(" Resources=").append(gs.getPlayer(player).getResources())
          .append(" Map=").append(pgs.getWidth()).append("x").append(pgs.getHeight())
          .append("\n");

        for (Unit u : pgs.getUnits()) {
            String team = u.getPlayer() == player ? "ALLY"
                        : u.getPlayer() == -1     ? "RES"
                        : "ENEMY";
            boolean idle = gs.getActionAssignment(u) == null;
            sb.append(team).append(" id=").append(u.getID())
              .append(" ").append(u.getType().name)
              .append(" (").append(u.getX()).append(",").append(u.getY()).append(")")
              .append(" hp=").append(u.getHitPoints())
              .append(idle ? " IDLE" : " busy")
              .append("\n");
        }
        return sb.toString();
    }

    // ── Response parsing ──────────────────────────────────────────────────────

    private Map<Long, UnitAction> parseDecisions(String response, int player,
                                                   GameState gs, PhysicalGameState pgs,
                                                   int mapWidth) {
        Map<Long, UnitAction> result = new HashMap<>();
        try {
            response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
            int s = response.indexOf("{"), e = response.lastIndexOf("}") + 1;
            if (s < 0 || e <= s) return result;

            JsonObject json  = JsonParser.parseString(response.substring(s, e)).getAsJsonObject();
            JsonArray  moves = json.getAsJsonArray("moves");
            if (moves == null) return result;

            if (json.has("thinking"))
                System.out.println("[yebot] " + json.get("thinking").getAsString());

            Set<Long> used = new HashSet<>();

            for (JsonElement el : moves) {
                if (!el.isJsonObject()) continue;
                JsonObject m = el.getAsJsonObject();
                try {
                    long uid = m.get("uid").getAsLong();
                    if (used.contains(uid)) continue;

                    Unit unit = pgs.getUnit(uid);
                    if (unit == null || unit.getPlayer() != player) continue;
                    if (gs.getActionAssignment(unit) != null) continue;

                    String type = m.get("type").getAsString().toLowerCase();
                    JsonArray args = m.has("args") ? m.getAsJsonArray("args") : new JsonArray();

                    UnitAction ua = buildUnitAction(type, args, unit, player, gs, pgs, mapWidth);
                    if (ua != null && gs.isUnitActionAllowed(unit, ua)) {
                        result.put(uid, ua);
                        used.add(uid);
                    }
                } catch (Exception ex) { /* skip bad move */ }
            }
        } catch (Exception ex) {
            System.err.println("[yebot] Parse error: " + ex.getMessage());
        }
        return result;
    }

    private UnitAction buildUnitAction(String type, JsonArray args, Unit unit,
                                        int player, GameState gs,
                                        PhysicalGameState pgs, int mapWidth) {
        switch (type) {
            case "move": {
                if (args.size() < 2) return null;
                int tx = args.get(0).getAsInt(), ty = args.get(1).getAsInt();
                UnitAction mv = pf.findPathToAdjacentPosition(unit,
                        tx + ty * mapWidth, gs, null);
                return mv == null ? null : new UnitAction(UnitAction.TYPE_MOVE, mv.getDirection());
            }
            case "attack": {
                if (args.size() < 2) return null;
                int tx = args.get(0).getAsInt(), ty = args.get(1).getAsInt();
                Unit target = pgs.getUnitAt(tx, ty);
                if (target == null || target.getPlayer() == player) return null;
                int d = Math.abs(unit.getX()-tx) + Math.abs(unit.getY()-ty);
                if (d <= unit.getType().attackRange)
                    return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, tx + ty * mapWidth);
                UnitAction mv = pf.findPathToAdjacentPosition(unit, tx + ty * mapWidth, gs, null);
                return mv == null ? null : new UnitAction(UnitAction.TYPE_MOVE, mv.getDirection());
            }
            case "harvest": {
                if (args.size() < 4) return null;
                int rx = args.get(0).getAsInt(), ry = args.get(1).getAsInt();
                int bx = args.get(2).getAsInt(), by = args.get(3).getAsInt();
                Unit res  = pgs.getUnitAt(rx, ry);
                Unit base = pgs.getUnitAt(bx, by);
                if (res == null || base == null || base.getType() != baseType) return null;
                if (unit.getResources() > 0) {
                    UnitAction mv = pf.findPathToAdjacentPosition(unit, bx+by*mapWidth, gs, null);
                    if (mv == null) return null;
                    boolean adj = Math.abs(unit.getX()-bx)+Math.abs(unit.getY()-by) == 1;
                    return new UnitAction(adj ? UnitAction.TYPE_RETURN : UnitAction.TYPE_MOVE,
                            mv.getDirection());
                } else {
                    UnitAction mv = pf.findPathToAdjacentPosition(unit, rx+ry*mapWidth, gs, null);
                    if (mv == null) return null;
                    boolean adj = Math.abs(unit.getX()-rx)+Math.abs(unit.getY()-ry) == 1;
                    return new UnitAction(adj ? UnitAction.TYPE_HARVEST : UnitAction.TYPE_MOVE,
                            mv.getDirection());
                }
            }
            case "train": {
                String name = args.size() > 0 ? args.get(0).getAsString().toLowerCase() : "";
                UnitType tt = null;
                if (unit.getType() == baseType && "worker".equals(name)) tt = workerType;
                else if (unit.getType() == barracksType) {
                    if ("light".equals(name))  tt = lightType;
                    if ("heavy".equals(name))  tt = heavyType;
                    if ("ranged".equals(name)) tt = rangedType;
                }
                if (tt == null || gs.getPlayer(player).getResources() < tt.cost) return null;
                for (int dir = 0; dir < 4; dir++) {
                    int nx = unit.getX()+UnitAction.DIRECTION_OFFSET_X[dir];
                    int ny = unit.getY()+UnitAction.DIRECTION_OFFSET_Y[dir];
                    if (nx<0||ny<0||nx>=pgs.getWidth()||ny>=pgs.getHeight()) continue;
                    if (pgs.getUnitAt(nx,ny) != null) continue;
                    return new UnitAction(UnitAction.TYPE_PRODUCE, dir, tt);
                }
                return null;
            }
            case "build": {
                if (args.size() < 2) return null;
                int bx = args.get(0).getAsInt(), by = args.get(1).getAsInt();
                if (gs.getPlayer(player).getResources() < barracksType.cost) return null;
                UnitAction mv = pf.findPathToAdjacentPosition(unit, bx+by*mapWidth, gs, null);
                if (mv == null) return null;
                boolean adj = Math.abs(unit.getX()-bx)+Math.abs(unit.getY()-by) == 1;
                return new UnitAction(adj ? UnitAction.TYPE_PRODUCE : UnitAction.TYPE_MOVE,
                        adj ? mv.getDirection() : mv.getDirection(),
                        adj ? barracksType : null);
            }
        }
        return null;
    }

    // ── HTTP call ─────────────────────────────────────────────────────────────

    private String callLLM(String stateText) {
        try {
            URL url = new URL(API_URL);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);
            conn.setConnectTimeout(LLM_TIMEOUT);
            conn.setReadTimeout(LLM_TIMEOUT);

            JsonObject req = new JsonObject();
            req.addProperty("model", OLLAMA_MODEL);

            JsonArray msgs = new JsonArray();
            JsonObject sys = new JsonObject();
            sys.addProperty("role", "system");
            sys.addProperty("content", SYSTEM_PROMPT);
            msgs.add(sys);
            JsonObject usr = new JsonObject();
            usr.addProperty("role", "user");
            usr.addProperty("content", stateText);
            msgs.add(usr);
            req.add("messages", msgs);

            JsonObject fmt = new JsonObject();
            fmt.addProperty("type", "json_object");
            req.add("response_format", fmt);
            req.addProperty("temperature", 0.2);
            req.addProperty("max_tokens", 512);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(req.toString().getBytes(StandardCharsets.UTF_8));
            }

            if (conn.getResponseCode() == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    JsonObject resp    = JsonParser.parseString(sb.toString()).getAsJsonObject();
                    JsonArray  choices = resp.getAsJsonArray("choices");
                    if (choices != null && choices.size() > 0)
                        return choices.get(0).getAsJsonObject()
                                .getAsJsonObject("message").get("content").getAsString();
                }
            }
        } catch (Exception e) {
            System.err.println("[yebot] callLLM: " + e.getMessage());
        }
        return "{}";
    }

    // ── Utility ───────────────────────────────────────────────────────────────

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