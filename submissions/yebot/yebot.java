/*
 * yebot — Minimal Async PureLLM Worker Rush
 *
 * ONE LLM call at tick 0 with an explicit scripted prompt.
 * The prompt tells the model exactly what to output.
 * Everything else is just applying the response and filling nones.
 *
 * This is a diagnostic build to verify:
 *   1. The LLM call actually completes
 *   2. The response is parsed correctly
 *   3. Units act on the decisions
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
            ? System.getenv("OLLAMA_MODEL") : "llama4:latest";
    private static final String API_URL = System.getenv("OLLAMA_URL") != null
            ? System.getenv("OLLAMA_URL") : "http://localhost:11434/v1/chat/completions";
    private static final int LLM_TIMEOUT = 60000;

    // ─── Unit Types ────────────────────────────────────────────────────────────
    private UnitTypeTable utt;
    private UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType;

    // ─── State ─────────────────────────────────────────────────────────────────
    // unitID → UnitAction decided by LLM
    private volatile Map<Long, UnitAction> decisions = new HashMap<>();
    private volatile boolean llmDone = false;
    private final ExecutorService llmThread = Executors.newSingleThreadExecutor();

    // ══════════════════════════════════════════════════════════════════════════
    //  CONSTRUCTORS
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
        decisions = new HashMap<>();
        llmDone   = false;
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

        // ── Tick 0: fire LLM async, build pre-validated vocab first ───────────
        if (tick == 0 && !llmDone) {
            // Build vocab now so we have valid action IDs to give the LLM
            Map<String, Long> idToUID   = new LinkedHashMap<>(); // actionId → unitUID
            Map<String, UnitAction> idToUA = new LinkedHashMap<>(); // actionId → UnitAction
            String vocabText = buildVocab(player, gs, pgs, idToUID, idToUA);

            System.out.println("[yebot] Firing opening LLM call...");
            llmThread.submit(() -> {
                try {
                    String prompt   = buildRushPrompt(vocabText);
                    String response = callLLM(prompt);
                    System.out.println("[yebot] LLM raw response: " + response.substring(0, Math.min(200, response.length())));
                    Map<Long, UnitAction> parsed = parseResponse(response, idToUID, idToUA);
                    System.out.println("[yebot] Parsed " + parsed.size() + " actions");
                    decisions = parsed;
                } catch (Exception e) {
                    System.err.println("[yebot] LLM failed: " + e.getMessage());
                    // Use vocab's best action per unit as fallback
                    decisions = buildFallback(idToUID, idToUA);
                    System.out.println("[yebot] Fallback: " + decisions.size() + " actions");
                } finally {
                    llmDone = true;
                }
            });
        }

        // ── Every tick: apply decisions to idle units ─────────────────────────
        PlayerAction pa = new PlayerAction();
        Map<Long, UnitAction> dec = decisions;

        if (!dec.isEmpty()) {
            for (Unit u : pgs.getUnits()) {
                if (u.getPlayer() != player) continue;
                if (gs.getActionAssignment(u) != null) continue;
                UnitAction ua = dec.get(u.getID());
                if (ua != null && gs.isUnitActionAllowed(u, ua)) {
                    pa.addUnitAction(u, ua);
                }
            }
        }

        pa.fillWithNones(gs, player, 1);
        return pa;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  VOCAB BUILDER — pre-validated actions, keyed by human-readable ID
    // ══════════════════════════════════════════════════════════════════════════

    private String buildVocab(int player, GameState gs, PhysicalGameState pgs,
                               Map<String, Long> idToUID,
                               Map<String, UnitAction> idToUA) {
        StringBuilder sb = new StringBuilder();
        int wIdx = 0;

        List<Unit> enemies   = new ArrayList<>();
        List<Unit> resources = new ArrayList<>();
        Unit myBase = null;

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) { resources.add(u); continue; }
            if (u.getPlayer() == player && u.getType() == baseType) myBase = u;
            if (u.getPlayer() != player && u.getPlayer() != -1) enemies.add(u);
        }

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() != player) continue;
            UnitType t = u.getType();

            if (t == workerType) {
                String wid = "W" + wIdx++;
                sb.append(wid).append(" [Worker at ").append(u.getX()).append(",").append(u.getY()).append("]:\n");

                // Harvest option
                if (myBase != null && !resources.isEmpty() && u.getResources() == 0) {
                    Unit res = nearestUnit(u, resources);
                    UnitAction ua = pf.findPathToAdjacentPosition(u,
                            res.getX() + res.getY() * pgs.getWidth(), gs, null);
                    if (ua != null) {
                        boolean adj = dist(u, res) == 1;
                        UnitAction actual = adj
                                ? new UnitAction(UnitAction.TYPE_HARVEST, ua.getDirection())
                                : new UnitAction(UnitAction.TYPE_MOVE, ua.getDirection());
                        if (gs.isUnitActionAllowed(u, actual)) {
                            String aid = wid + "_harvest";
                            idToUID.put(aid, u.getID());
                            idToUA.put(aid, actual);
                            sb.append("  ").append(aid).append("\n");
                        }
                    }
                }

                // Attack options — nearest enemy and enemy base
                if (!enemies.isEmpty()) {
                    Unit nearest = nearestUnit(u, enemies);
                    addAttackAction(wid, u, nearest, gs, pgs, sb, idToUID, idToUA, "_attack_nearest");

                    enemies.stream()
                            .filter(e -> e.getType() == baseType)
                            .findFirst()
                            .ifPresent(base -> addAttackAction(wid, u, base, gs, pgs,
                                    sb, idToUID, idToUA, "_attack_base"));
                }

            } else if (t == baseType) {
                String bid = "BASE";
                sb.append(bid).append(" [Base at ").append(u.getX()).append(",").append(u.getY()).append("]:\n");
                int myRes = gs.getPlayer(player).getResources();
                if (myRes >= workerType.cost) {
                    for (int dir = 0; dir < 4; dir++) {
                        int nx = u.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
                        int ny = u.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
                        if (nx < 0 || ny < 0 || nx >= pgs.getWidth() || ny >= pgs.getHeight()) continue;
                        if (pgs.getUnitAt(nx, ny) != null) continue;
                        UnitAction ua = new UnitAction(UnitAction.TYPE_PRODUCE, dir, workerType);
                        if (gs.isUnitActionAllowed(u, ua)) {
                            String aid = bid + "_train_worker";
                            idToUID.put(aid, u.getID());
                            idToUA.put(aid, ua);
                            sb.append("  ").append(aid).append("\n");
                            break;
                        }
                    }
                }
            }
        }

        return sb.toString();
    }

    private void addAttackAction(String uid, Unit unit, Unit target,
                                  GameState gs, PhysicalGameState pgs,
                                  StringBuilder sb,
                                  Map<String, Long> idToUID,
                                  Map<String, UnitAction> idToUA,
                                  String suffix) {
        if (target == null) return;
        UnitAction ua;
        int d = dist(unit, target);
        if (d <= unit.getType().attackRange) {
            ua = new UnitAction(UnitAction.TYPE_ATTACK_LOCATION,
                    target.getX() + target.getY() * pgs.getWidth());
        } else {
            UnitAction move = pf.findPathToAdjacentPosition(unit,
                    target.getX() + target.getY() * pgs.getWidth(), gs, null);
            if (move == null) return;
            ua = new UnitAction(UnitAction.TYPE_MOVE, move.getDirection());
        }
        if (!gs.isUnitActionAllowed(unit, ua)) return;
        String aid = uid + suffix;
        idToUID.put(aid, unit.getID());
        idToUA.put(aid, ua);
        sb.append("  ").append(aid).append("\n");
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  RUSH PROMPT — explicit, scripted, tells model exactly what to output
    // ══════════════════════════════════════════════════════════════════════════

    private String buildRushPrompt(String vocabText) {
        return "You are commanding a MicroRTS agent doing a WORKER RUSH.\n"
             + "Pick one action per unit from the list below. Output JSON only.\n\n"
             + "RULES:\n"
             + "- BASE: always assign BASE_train_worker\n"
             + "- W0: assign W0_harvest (keep economy alive)\n"
             + "- W1, W2, ...: assign their _attack_nearest or _attack_base action\n"
             + "- If a unit has no attack action, assign its harvest action\n\n"
             + "AVAILABLE ACTIONS:\n"
             + vocabText
             + "\nOUTPUT FORMAT:\n"
             + "{\"actions\": [\n"
             + "  {\"id\": \"BASE_train_worker\"},\n"
             + "  {\"id\": \"W0_harvest\"},\n"
             + "  {\"id\": \"W1_attack_nearest\"}\n"
             + "]}\n"
             + "Use only IDs from the list above. No other text.";
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  RESPONSE PARSER
    // ══════════════════════════════════════════════════════════════════════════

    private Map<Long, UnitAction> parseResponse(String response,
                                                  Map<String, Long> idToUID,
                                                  Map<String, UnitAction> idToUA) {
        Map<Long, UnitAction> result = new HashMap<>();
        try {
            // Strip think tags
            response = response.replaceAll("(?s)<think>.*?</think>", "").trim();
            // Extract JSON
            int s = response.indexOf("{"), e = response.lastIndexOf("}") + 1;
            if (s < 0 || e <= s) return result;
            response = response.substring(s, e);

            JsonObject json = JsonParser.parseString(response).getAsJsonObject();
            JsonArray actions = json.getAsJsonArray("actions");
            if (actions == null) return result;

            Set<Long> usedUIDs = new HashSet<>();
            for (JsonElement el : actions) {
                if (!el.isJsonObject()) continue;
                String aid = el.getAsJsonObject().get("id").getAsString();
                Long uid = idToUID.get(aid);
                UnitAction ua = idToUA.get(aid);
                if (uid != null && ua != null && !usedUIDs.contains(uid)) {
                    result.put(uid, ua);
                    usedUIDs.add(uid);
                    System.out.println("[yebot] Assigned: " + aid);
                } else {
                    System.out.println("[yebot] Unknown/dup action id: " + aid);
                }
            }
        } catch (Exception ex) {
            System.err.println("[yebot] Parse error: " + ex.getMessage());
        }
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  FALLBACK — used if LLM fails, ensures units never permanently idle
    // ══════════════════════════════════════════════════════════════════════════

    private Map<Long, UnitAction> buildFallback(Map<String, Long> idToUID,
                                                  Map<String, UnitAction> idToUA) {
        Map<Long, UnitAction> result = new HashMap<>();
        Set<Long> assigned = new HashSet<>();

        // Priority order: attack_base > attack_nearest > harvest > train_worker
        String[] priority = {"_attack_base", "_attack_nearest", "_harvest", "_train_worker"};

        for (String suffix : priority) {
            for (Map.Entry<String, Long> entry : idToUID.entrySet()) {
                String aid = entry.getKey();
                Long uid   = entry.getValue();
                if (!aid.endsWith(suffix)) continue;
                if (assigned.contains(uid)) continue;
                result.put(uid, idToUA.get(aid));
                assigned.add(uid);
            }
        }
        return result;
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  LLM HTTP CALL
    // ══════════════════════════════════════════════════════════════════════════

    private String callLLM(String prompt) {
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
            JsonObject msg = new JsonObject();
            msg.addProperty("role", "user");
            msg.addProperty("content", prompt);
            msgs.add(msg);
            req.add("messages", msgs);

            JsonObject fmt = new JsonObject();
            fmt.addProperty("type", "json_object");
            req.add("response_format", fmt);
            req.addProperty("temperature", 0.0); // deterministic
            req.addProperty("max_tokens", 256);  // small — just a list of IDs

            try (OutputStream os = conn.getOutputStream()) {
                os.write(req.toString().getBytes(StandardCharsets.UTF_8));
            }

            int code = conn.getResponseCode();
            System.out.println("[yebot] HTTP status: " + code);

            if (code == 200) {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getInputStream(), StandardCharsets.UTF_8))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    JsonObject resp = JsonParser.parseString(sb.toString()).getAsJsonObject();
                    JsonArray choices = resp.getAsJsonArray("choices");
                    if (choices != null && choices.size() > 0) {
                        return choices.get(0).getAsJsonObject()
                                .getAsJsonObject("message")
                                .get("content").getAsString();
                    }
                }
            } else {
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(conn.getErrorStream(), StandardCharsets.UTF_8))) {
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = br.readLine()) != null) sb.append(line);
                    System.err.println("[yebot] API error body: " + sb);
                }
            }
        } catch (Exception e) {
            System.err.println("[yebot] callLLM exception: " + e.getClass().getSimpleName()
                    + ": " + e.getMessage());
        }
        return "{\"actions\":[]}";
    }

    // ══════════════════════════════════════════════════════════════════════════
    //  UTILITIES
    // ══════════════════════════════════════════════════════════════════════════

    private int dist(Unit a, Unit b) {
        return Math.abs(a.getX()-b.getX()) + Math.abs(a.getY()-b.getY());
    }

    private Unit nearestUnit(Unit src, List<Unit> units) {
        Unit best = null; int bestD = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = dist(src, u);
            if (d < bestD) { bestD = d; best = u; }
        }
        return best;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}