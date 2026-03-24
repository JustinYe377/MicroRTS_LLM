package ai.mcts.submissions.xiebot;

import ai.RandomBiasedAI;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import ai.mcts.naivemcts.NaiveMCTS;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

/**

XieBot v3

Practical tournament design for small aggressive maps:

Deterministic opening/macro/execution is the default (fast and reliable).
LLM has a narrow role: choose one of three safe macro modes periodically.
MCTS is used only in dense tactical fights where lookahead is valuable.
*/
public class XieBot extends AbstractionLayerAI {
// ===== Core unit types =====
private UnitTypeTable utt;
private UnitType workerType;
private UnitType baseType;
private UnitType barracksType;
private UnitType lightType;
private UnitType rangedType;
private UnitType heavyType;

// ===== Safe macro modes (bounded strategy space) =====
private enum MacroPlan {
AGGRO_LIGHT, // Fast pressure, especially on 8x8
STABLE_RANGED, // Mix in ranged once stable
DEFENSIVE_HOLD // Survive rush first, then counter
}

// ===== LLM config =====
private static final String OLLAMA_ENDPOINT =
System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434") + "/api/generate";
private static final String OLLAMA_MODEL =
System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
private static final int LLM_CONNECT_TIMEOUT_MS = 70;
private static final int LLM_READ_TIMEOUT_MS = 140;
private static final int LLM_DECISION_INTERVAL = 160;
private static final int LLM_FAILURE_COOLDOWN = 4;
private static final int LLM_START_AFTER_TICK = 400;

// ===== Tactical MCTS config =====
private final NaiveMCTS tacticalMCTS;
private static final int MCTS_TIME_BUDGET_MS = 35;
private static final int MCTS_LOOKAHEAD = 80;
private static final int MCTS_MAX_DEPTH = 8;
private static final int DEFENSE_RADIUS = 6;

// ===== Runtime state =====
private MacroPlan macroPlan = MacroPlan.AGGRO_LIGHT;
private int lastMacroDecisionTime = -9999;
private int llmFailures = 0;

public XieBot(UnitTypeTable a_utt) {
super(new AStarPathFinding());
tacticalMCTS = new NaiveMCTS(
MCTS_TIME_BUDGET_MS,
-1,
MCTS_LOOKAHEAD,
MCTS_MAX_DEPTH,
0.25f,
0.0f,
0.35f,
new RandomBiasedAI(),
new SimpleSqrtEvaluationFunction3(),
true
);
reset(a_utt);
}

@Override
public void reset(UnitTypeTable a_utt) {
utt = a_utt;
workerType = utt.getUnitType("Worker");
baseType = utt.getUnitType("Base");
barracksType = utt.getUnitType("Barracks");
lightType = utt.getUnitType("Light");
rangedType = utt.getUnitType("Ranged");
heavyType = utt.getUnitType("Heavy");
reset();
}

@Override
public void reset() {
super.reset();
macroPlan = MacroPlan.AGGRO_LIGHT;
lastMacroDecisionTime = -9999;
llmFailures = 0;
tacticalMCTS.reset();
}

@Override
public AI clone() {
return new XieBot(utt);
}

@Override
public PlayerAction getAction(int player, GameState gs) {
if (!gs.canExecuteAnyAction(player)) {
return new PlayerAction();
}

 PhysicalGameState pgs = gs.getPhysicalGameState();
 int enemy = 1 - player;

 Counts my = countUnits(pgs, player);
 Counts opp = countUnits(pgs, enemy);

 boolean smallMap = pgs.getWidth() * pgs.getHeight() <= 100;
 boolean opening = gs.getTime() < (smallMap ? 320 : 500);
 boolean enemyNearBase = isEnemyNearBase(pgs, my, enemy, 4);

 maybeUpdateMacroPlan(gs, my, opp, smallMap);

 // Tactical MCTS only in mid-fight states where it pays off.
 if (shouldUseTacticalMCTS(gs, my, opp, opening, enemyNearBase, smallMap)) {
     try {
         PlayerAction mctsAction = tacticalMCTS.getAction(player, gs);
         if (mctsAction != null && !mctsAction.isEmpty()) {
             return mctsAction;
         }
     } catch (Exception ignored) {
         // Fall back to deterministic policy immediately.
     }
 }

 // Deterministic macro + execution policy:
 // 1) production, 2) worker economy/defense, 3) military pressure.
 Unit myBase = my.bases.isEmpty() ? null : my.bases.get(0);
 Unit myBarracks = my.barracks.isEmpty() ? null : my.barracks.get(0);
 Unit enemyBase = opp.bases.isEmpty() ? null : opp.bases.get(0);

 handleBaseProduction(gs, player, my, enemyNearBase, smallMap, opening);
 handleBarracksProduction(gs, player, my, opp, enemyNearBase, smallMap, opening);
 handleWorkers(gs, player, my, opp, myBase, myBarracks, enemyBase, enemyNearBase, smallMap, opening);
 handleMilitary(gs, player, enemy, my, opp, myBase, enemyBase, enemyNearBase);

 return translateActions(player, gs);
}

// ===========================
// Deterministic production
// ===========================
private void handleBaseProduction(GameState gs, int player, Counts my, boolean enemyNearBase, boolean smallMap, boolean opening) {
Player me = gs.getPlayer(player);

 int desiredWorkers;
 if (smallMap) {
     desiredWorkers = opening ? 4 : 5;
 } else {
     desiredWorkers = opening ? 6 : 8;
 }

 // Under immediate threat with low army, keep worker count a little higher for emergency defense.
 if (enemyNearBase && my.militaryCount() <= 1) {
     desiredWorkers = Math.max(desiredWorkers, 6);
 }

 for (Unit base : my.bases) {
     if (!isIdle(gs, base)) continue;
     if (my.workers.size() < desiredWorkers && me.getResources() >= workerType.cost) {
         train(base, workerType);
         // Conservative accounting to avoid over-issuing train orders in same cycle.
         my.workers.add(new Unit(-1, workerType, 0, 0));
     }
 }
}

private void handleBarracksProduction(
GameState gs,
int player,
Counts my,
Counts opp,
boolean enemyNearBase,
boolean smallMap,
boolean opening) {

 Player me = gs.getPlayer(player);
 for (Unit barracks : my.barracks) {
     if (!isIdle(gs, barracks)) continue;

     UnitType chosen = chooseBarracksUnit(my, opp, enemyNearBase, smallMap, opening);
     if (chosen != null && me.getResources() >= chosen.cost) {
         train(barracks, chosen);
     }
 }
}

private UnitType chooseBarracksUnit(Counts my, Counts opp, boolean enemyNearBase, boolean smallMap, boolean opening) {
int oppCombatWithWorkers = opp.militaryCount() + opp.workers.size();

 if (enemyNearBase && my.militaryCount() + 1 < oppCombatWithWorkers) {
     return lightType;
 }

 if (macroPlan == MacroPlan.DEFENSIVE_HOLD) {
     return lightType;
 }

 if (smallMap) {
     // 8x8 emphasis: tempo and collision-fighting are king.
     return lightType;
 }

 // Larger maps: transition to ranged only after stability.
 if (macroPlan == MacroPlan.STABLE_RANGED && !opening) {
     if (my.ranged.size() * 2 < my.light.size() + 1) {
         return rangedType;
     }
 }

 if (my.ranged.size() < my.light.size() / 2 && my.light.size() >= 4) {
     return rangedType;
 }

 return lightType;
}

// ===========================
// Worker manager (economy + anti-rush + build timing)
// ===========================
private void handleWorkers(
GameState gs,
int player,
Counts my,
Counts opp,
Unit myBase,
Unit myBarracks,
Unit enemyBase,
boolean enemyNearBase,
boolean smallMap,
boolean opening) {

 Player me = gs.getPlayer(player);
 int enemy = 1 - player;

 List<Unit> idleWorkers = new ArrayList<>();
 for (Unit w : my.workers) {
     if (isIdle(gs, w)) idleWorkers.add(w);
 }

 // Emergency anti-rush: if no army and enemy close, pull idle workers to fight now.
 if (enemyNearBase && my.militaryCount() <= 1) {
     for (Unit w : idleWorkers) {
         Unit threat = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
         if (threat != null) attack(w, threat);
     }
     return;
 }

 // Barracks timing: first barracks ASAP, second barracks later on larger maps.
 if (shouldBuildBarracks(gs, my, smallMap, opening, enemyNearBase)
         && me.getResources() >= barracksType.cost
         && !idleWorkers.isEmpty()) {
     Unit builder = selectBuilder(idleWorkers, myBase);
     int[] site = chooseBarracksSite(gs.getPhysicalGameState(), myBase != null ? myBase : builder, enemy);
     if (builder != null && site != null) {
         build(builder, barracksType, site[0], site[1]);
         idleWorkers.remove(builder);
     }
 }

 int harvestersWanted;
 if (smallMap) {
     harvestersWanted = opening ? 2 : 2;
 } else {
     harvestersWanted = opening ? 4 : 3;
 }

 if (enemyNearBase) {
     harvestersWanted = Math.max(1, harvestersWanted - 1);
 }
 if (my.barracks.size() >= 2 && !smallMap) {
     harvestersWanted = Math.min(harvestersWanted + 1, 5);
 }

 // Keep at least one non-harvesting worker for flexibility.
 harvestersWanted = Math.min(harvestersWanted, Math.max(0, my.workers.size() - 1));

 List<Unit> resources = listResources(gs.getPhysicalGameState());
 int assigned = 0;

 for (Unit w : idleWorkers) {
     if (assigned >= harvestersWanted) break;
     Unit nearestRes = nearestResource(gs.getPhysicalGameState(), w, resources);
     if (nearestRes != null && myBase != null) {
         harvest(w, nearestRes, myBase);
         assigned++;
     }
 }

 // Remaining idle workers become tactical workers: defend while weak, pressure once stable.
 for (Unit w : idleWorkers) {
     if (!isIdle(gs, w)) continue;
     Unit target;
     if (enemyNearBase) {
         target = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
     } else if (my.militaryCount() < 3 || myBase == null) {
         target = nearestEnemyWithin(gs.getPhysicalGameState(), w, enemy, DEFENSE_RADIUS);
     } else if (enemyBase != null) {
         target = enemyBase;
     } else {
         target = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
     }
     if (target != null) attack(w, target);
 }
}

private boolean shouldBuildBarracks(GameState gs, Counts my, boolean smallMap, boolean opening, boolean enemyNearBase) {
 if (my.barracks.isEmpty()) {
     return true;
 }

 if (smallMap || opening || enemyNearBase) {
     return false;
 }

 if (my.barracks.size() >= 2) {
     return false;
 }

 Player p0 = gs.getPlayer(0);
 Player p1 = gs.getPlayer(1);
 int maxKnownResources = Math.max(p0.getResources(), p1.getResources());
 return my.workers.size() >= 5 && my.militaryCount() >= 4 && maxKnownResources >= 7;
}

// ===========================
// Military execution
// ===========================
private void handleMilitary(
GameState gs,
int player,
int enemy,
Counts my,
Counts opp,
Unit myBase,
Unit enemyBase,
boolean enemyNearBase) {

 List<Unit> army = new ArrayList<>();
 army.addAll(my.light);
 army.addAll(my.ranged);
 army.addAll(my.heavy);

 for (Unit unit : army) {
     if (!isIdle(gs, unit)) continue;

     Unit target = selectCombatTarget(gs.getPhysicalGameState(), unit, enemy, myBase, enemyNearBase, opp);
     if (target == null && enemyBase != null) {
         target = enemyBase;
     }
     if (target != null) {
         attack(unit, target);
     }
 }
}

private Unit selectCombatTarget(PhysicalGameState pgs, Unit attacker, int enemy, Unit myBase, boolean urgentDefense, Counts opp) {
List<Unit> enemies = new ArrayList<>();
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() == enemy) enemies.add(u);
}
if (enemies.isEmpty()) return null;

 enemies.sort(Comparator.comparingInt(e -> targetScore(attacker, e, myBase, urgentDefense, opp)));
 return enemies.get(0);
}

private int targetScore(Unit attacker, Unit target, Unit myBase, boolean urgentDefense, Counts opp) {
int d = manhattan(attacker, target);
int typeBias;

 if (target.getType() == workerType) {
     // If enemy has low workers, each kill is very high value.
     typeBias = (opp.workers.size() <= 2) ? 1 : 5;
 } else if (target.getType() == lightType || target.getType() == rangedType || target.getType() == heavyType) {
     typeBias = 3;
 } else if (target.getType() == barracksType) {
     typeBias = 7;
 } else if (target.getType() == baseType) {
     typeBias = 9;
 } else {
     typeBias = 6;
 }

 if (urgentDefense && target.getType().canAttack) {
     typeBias -= 2;
 }

 int baseProximity = 0;
 if (myBase != null) {
     baseProximity = manhattan(myBase, target);
 }

 return d * 3 + typeBias + (urgentDefense ? baseProximity * 2 : 0);
}

// ===========================
// Macro controller (heuristic first, bounded LLM optional)
// ===========================
private void maybeUpdateMacroPlan(GameState gs, Counts my, Counts opp, boolean smallMap) {
if (gs.getTime() - lastMacroDecisionTime < LLM_DECISION_INTERVAL) {
return;
}
lastMacroDecisionTime = gs.getTime();

 MacroPlan heuristic = heuristicMacro(my, opp, smallMap);

 if (smallMap || gs.getTime() < LLM_START_AFTER_TICK) {
     macroPlan = heuristic;
     return;
 }

 // If LLM has recently failed repeatedly, skip call this cycle.
 if (llmFailures >= LLM_FAILURE_COOLDOWN) {
     llmFailures--;
     macroPlan = heuristic;
     return;
 }

 String llmRaw = queryMacroLLM(gs.getTime(), my, opp, smallMap);
 if (llmRaw == null) {
     llmFailures++;
     macroPlan = heuristic;
     return;
 }

 MacroPlan parsed = parseMacroToken(llmRaw);
 if (parsed == null) {
     llmFailures++;
     macroPlan = heuristic;
     return;
 }

 llmFailures = 0;
 macroPlan = parsed;
}

private MacroPlan heuristicMacro(Counts my, Counts opp, boolean smallMap) {
int myArmy = my.militaryCount();
int oppArmy = opp.militaryCount();

 if (smallMap) {
     if (myArmy + 1 < oppArmy) return MacroPlan.DEFENSIVE_HOLD;
     return MacroPlan.AGGRO_LIGHT;
 }

 if (myArmy < oppArmy) return MacroPlan.DEFENSIVE_HOLD;
 if (my.ranged.size() < my.light.size()) return MacroPlan.STABLE_RANGED;
 return MacroPlan.AGGRO_LIGHT;
}

private String queryMacroLLM(int time, Counts my, Counts opp, boolean smallMap) {
String prompt = "Choose exactly ONE token from {AGGRO_LIGHT, STABLE_RANGED, DEFENSIVE_HOLD}."
+ " Return only the token."
+ " time=" + time
+ " smallMap=" + smallMap
+ " my(workers=" + my.workers.size() + ",bases=" + my.bases.size() + ",barracks=" + my.barracks.size()
+ ",light=" + my.light.size() + ",ranged=" + my.ranged.size() + ",heavy=" + my.heavy.size() + ")"
+ " opp(workers=" + opp.workers.size() + ",bases=" + opp.bases.size() + ",barracks=" + opp.barracks.size()
+ ",light=" + opp.light.size() + ",ranged=" + opp.ranged.size() + ",heavy=" + opp.heavy.size() + ").";

 HttpURLConnection conn = null;
 try {
     URL url = new URL(OLLAMA_ENDPOINT);
     conn = (HttpURLConnection) url.openConnection();
     conn.setRequestMethod("POST");
     conn.setDoOutput(true);
     conn.setConnectTimeout(LLM_CONNECT_TIMEOUT_MS);
     conn.setReadTimeout(LLM_READ_TIMEOUT_MS);
     conn.setRequestProperty("Content-Type", "application/json");

     String body = "{"
             + "\"model\":\"" + jsonEscape(OLLAMA_MODEL) + "\","
             + "\"stream\":false,"
             + "\"prompt\":\"" + jsonEscape(prompt) + "\""
             + "}";

     try (OutputStream os = conn.getOutputStream()) {
         os.write(body.getBytes(StandardCharsets.UTF_8));
     }

     int code = conn.getResponseCode();
     if (code < 200 || code >= 300) return null;

     String response = readAll(conn.getInputStream());
     return extractResponseText(response);
 } catch (Exception e) {
     return null;
 } finally {
     if (conn != null) conn.disconnect();
 }
}

private MacroPlan parseMacroToken(String text) {
String t = text == null ? "" : text.toUpperCase();
if (t.contains("AGGRO_LIGHT")) return MacroPlan.AGGRO_LIGHT;
if (t.contains("STABLE_RANGED")) return MacroPlan.STABLE_RANGED;
if (t.contains("DEFENSIVE_HOLD")) return MacroPlan.DEFENSIVE_HOLD;
return null;
}

// ===========================
// Tactical MCTS gating
// ===========================
private boolean shouldUseTacticalMCTS(
GameState gs,
Counts my,
Counts opp,
boolean opening,
boolean enemyNearBase,
boolean smallMap) {

 if (opening) return false; // deterministic opening is stronger and cheaper.
 if (enemyNearBase && my.militaryCount() <= 1) return false; // anti-rush worker logic should trigger.

 int myArmy = my.militaryCount();
 int oppArmy = opp.militaryCount();
 if (myArmy < 4 || oppArmy < 4) return false;

 // Use MCTS mainly when forces are comparable (high tactical branching).
 int diff = Math.abs(myArmy - oppArmy);
 if (diff > 3) return false;

 // On very small maps, keep MCTS usage conservative to avoid latency spikes.
 if (smallMap && gs.getTime() < 700) return false;

 // Do not invoke tactical MCTS when economy is fragile.
 if (my.workers.size() < 2 || my.bases.isEmpty()) return false;

 return true;
}

// ===========================
// Geometry/helpers
// ===========================
private boolean isIdle(GameState gs, Unit u) {
return gs.getActionAssignment(u) == null;
}

private boolean isEnemyNearBase(PhysicalGameState pgs, Counts my, int enemy, int radius) {
if (my.bases.isEmpty()) return false;
Unit base = my.bases.get(0);
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != enemy) continue;
if (manhattan(base, u) <= radius) return true;
}
return false;
}

private int[] chooseBarracksSite(PhysicalGameState pgs, Unit anchor, int enemy) {
int[][] ring = {
{1,0}, {-1,0}, {0,1}, {0,-1},
{2,0}, {-2,0}, {0,2}, {0,-2},
{1,1}, {-1,1}, {1,-1}, {-1,-1}
};

 int bestScore = Integer.MIN_VALUE;
 int[] best = null;

 for (int[] d : ring) {
     int x = anchor.getX() + d[0];
     int y = anchor.getY() + d[1];
     if (x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight()) continue;
     if (pgs.getTerrain(x, y) != PhysicalGameState.TERRAIN_NONE) continue;
     if (pgs.getUnitAt(x, y) != null) continue;

     // Prefer sites that are not too far from base, and not too exposed to nearest enemy.
     int baseDist = Math.abs(anchor.getX() - x) + Math.abs(anchor.getY() - y);
     int enemyDist = nearestEnemyDistanceFromCell(pgs, x, y, enemy);
     int score = enemyDist * 2 - baseDist;

     if (score > bestScore) {
         bestScore = score;
         best = new int[] {x, y};
     }
 }
 return best;
}

private Unit selectBuilder(List<Unit> workers, Unit base) {
if (workers.isEmpty()) return null;
if (base == null) return workers.get(0);

 Unit best = null;
 int bestD = Integer.MAX_VALUE;
 for (Unit w : workers) {
     int d = manhattan(w, base);
     if (d < bestD) {
         bestD = d;
         best = w;
     }
 }
 return best;
}

private Unit nearestEnemy(PhysicalGameState pgs, Unit from, int enemy) {
Unit best = null;
int bestD = Integer.MAX_VALUE;
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != enemy) continue;
int d = manhattan(from, u);
if (d < bestD) {
bestD = d;
best = u;
}
}
return best;
}

private Unit nearestEnemyWithin(PhysicalGameState pgs, Unit from, int enemy, int maxDistance) {
 Unit best = null;
 int bestD = Integer.MAX_VALUE;
 for (Unit u : pgs.getUnits()) {
     if (u.getPlayer() != enemy) continue;
     int d = manhattan(from, u);
     if (d <= maxDistance && d < bestD) {
         bestD = d;
         best = u;
     }
 }
 return best;
}

private int nearestEnemyDistanceFromCell(PhysicalGameState pgs, int x, int y, int enemy) {
int best = Integer.MAX_VALUE;
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != enemy) continue;
int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
if (d < best) best = d;
}
return best;
}

private Unit nearestResource(PhysicalGameState pgs, Unit from, List<Unit> resources) {
Unit best = null;
int bestD = Integer.MAX_VALUE;
for (Unit r : resources) {
int d = manhattan(from, r);
if (d < bestD) {
bestD = d;
best = r;
}
}
return best;
}

private List<Unit> listResources(PhysicalGameState pgs) {
List<Unit> out = new ArrayList<>();
for (Unit u : pgs.getUnits()) {
if (u.getType().isResource) out.add(u);
}
return out;
}

private int manhattan(Unit a, Unit b) {
return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
}

private Counts countUnits(PhysicalGameState pgs, int player) {
Counts c = new Counts();
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != player) continue;
if (u.getType() == workerType) c.workers.add(u);
else if (u.getType() == baseType) c.bases.add(u);
else if (u.getType() == barracksType) c.barracks.add(u);
else if (u.getType() == lightType) c.light.add(u);
else if (u.getType() == rangedType) c.ranged.add(u);
else if (u.getType() == heavyType) c.heavy.add(u);
}
return c;
}

// ===========================
// Lightweight HTTP/JSON helpers
// ===========================
private String readAll(InputStream in) throws IOException {
try (BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
StringBuilder sb = new StringBuilder();
String line;
while ((line = br.readLine()) != null) sb.append(line);
return sb.toString();
}
}

private String extractResponseText(String json) {
if (json == null) return null;
int key = json.indexOf("\"response\"");
if (key < 0) return null;
int colon = json.indexOf(':', key);
if (colon < 0) return null;
int start = json.indexOf('"', colon + 1);
if (start < 0) return null;

 StringBuilder out = new StringBuilder();
 boolean escaped = false;
 for (int i = start + 1; i < json.length(); i++) {
     char ch = json.charAt(i);
     if (escaped) {
         out.append(ch);
         escaped = false;
         continue;
     }
     if (ch == '\\') {
         escaped = true;
         continue;
     }
     if (ch == '"') {
         break;
     }
     out.append(ch);
 }
 return out.toString().trim();
}

private String jsonEscape(String s) {
return s.replace("\\", "\\\\")
.replace("\"", "\\\"")
.replace("\n", " ");
}

private static final class Counts {
final List<Unit> workers = new ArrayList<>();
final List<Unit> bases = new ArrayList<>();
final List<Unit> barracks = new ArrayList<>();
final List<Unit> light = new ArrayList<>();
final List<Unit> ranged = new ArrayList<>();
final List<Unit> heavy = new ArrayList<>();

 int militaryCount() {
     return light.size() + ranged.size() + heavy.size();
 }
}

@Override
public List<ParameterSpecification> getParameters() {
return new ArrayList<>();
}
}