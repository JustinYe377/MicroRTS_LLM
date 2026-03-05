package ai.mcts.submissions.xiebot;
import ai.mcts.naivemcts.NaiveMCTSNode;
import ai.*;
import ai.abstraction.BoomEconomy;
import ai.abstraction.LightRush;
import ai.abstraction.RangedRush;
import ai.abstraction.TurtleDefense;
import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import ai.evaluation.EvaluationFunction;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import rts.GameState;
import rts.PartiallyObservableGameState;
import rts.PhysicalGameState;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitTypeTable;
import ai.core.InterruptibleAI;

/**
 *
 * @author shaun xie
 */
public class XieBot extends AIWithComputationBudget implements InterruptibleAI {
    public static int DEBUG = 0;
    public EvaluationFunction ef;
       
    Random r = new Random();
    public AI playoutPolicy = new LightRush(new UnitTypeTable());
    private final LightRush openingBookPolicy;
    protected long max_actions_so_far = 0;
    
    protected GameState gs_to_start_from;
    protected NaiveMCTSNode tree;
    protected int current_iteration = 0;
            
    public int MAXSIMULATIONTIME = 1024;
    public int MAX_TREE_DEPTH = 10;
    
    protected int player;
    
    public float epsilon_0 = 0.2f;
    public float epsilon_l = 0.25f;
    public float epsilon_g = 0.0f;

    // these variables are for using a discount factor on the epsilon values above. My experiments indicate that things work better without discount
    // So, they are just maintained here for completeness:
    public float initial_epsilon_0 = 0.2f;
    public float initial_epsilon_l = 0.25f;
    public float initial_epsilon_g = 0.0f;
    public float discount_0 = 0.999f;
    public float discount_l = 0.999f;
    public float discount_g = 0.999f;
    
    public int global_strategy = NaiveMCTSNode.E_GREEDY;
    public boolean forceExplorationOfNonSampledActions = true;
    
    // statistics:
    public long total_runs = 0;
    public long total_cycles_executed = 0;
    public long total_actions_issued = 0;
    public long total_time = 0;

    private enum MacroStrategy {
        LIGHT_RUSH,
        WORKER_RUSH,
        BOOM_ECONOMY,
        TURTLE_DEFENSE,
        COUNTER_ATTACK
    }

    private static final class StrategyDecision {
        MacroStrategy strategy;
        float aggression;
        int timeBudget;
        boolean fromLLM;

        StrategyDecision(MacroStrategy strategy, float aggression, int timeBudget, boolean fromLLM) {
            this.strategy = strategy;
            this.aggression = aggression;
            this.timeBudget = timeBudget;
            this.fromLLM = fromLLM;
        }
    }

    private static final class UnitCounts {
        int workers;
        int bases;
        int barracks;
        int lights;
        int ranged;
        int heavy;

        int militaryCount() {
            return lights + ranged + heavy;
        }
    }

    private static final String OLLAMA_ENDPOINT = "http://localhost:11434/api/chat";
    private static final String OLLAMA_MODEL = "llama3.1:8b";
    private static final int OLLAMA_CONNECT_TIMEOUT_MS = 150;
    private static final int OLLAMA_READ_TIMEOUT_MS = 300;
    private static final int STRATEGY_START_TICK = 300;
    private static final int STRATEGY_INTERVAL = 200;
    private static final int ENEMY_NEAR_BASE_RADIUS = 6;
    private static final int MIN_TIME_BUDGET = 50;
    private static final int MAX_TIME_BUDGET = 200;
    private static final boolean SILENCE_PLAYOUT_STDOUT = true;
    private static final Pattern STRATEGY_JSON_PATTERN = Pattern.compile("\"strategy\"\\s*:\\s*\"([A-Z_]+)\"");
    private static final Pattern AGGRESSION_JSON_PATTERN = Pattern.compile("\"aggression\"\\s*:\\s*(-?\\d+(?:\\.\\d+)?)");
    private static final Pattern TIME_BUDGET_JSON_PATTERN = Pattern.compile("\"timeBudget\"\\s*:\\s*(-?\\d+)");

    private AI lightRushPolicy;
    private AI workerRushPolicy;
    private AI boomEconomyPolicy;
    private AI turtleDefensePolicy;
    private AI counterAttackPolicy;
    private UnitTypeTable strategyUTT;
    private StrategyDecision currentDecision = new StrategyDecision(MacroStrategy.LIGHT_RUSH, 0.6f, 120, false);
    private int lastStrategyDecisionTime = -9999;
    private int lastDecisionBaseCount = -1;
    private int lastDecisionBarracksCount = -1;
    private boolean previousEmergencyState = false;
    
    
    public XieBot(UnitTypeTable utt) {
        this(100,-1,100,10,
             0.3f, 0.0f, 0.4f,
             new LightRush(utt),
             new SimpleSqrtEvaluationFunction3(), true);
        openingBookPolicy.reset(utt);
    }    
    
    
    public XieBot(int available_time, int max_playouts, int lookahead, int max_depth, 
                               float e_l, float discout_l,
                               float e_g, float discout_g, 
                               float e_0, float discout_0, 
                               AI policy, EvaluationFunction a_ef,
                               boolean fensa) {
        super(available_time, max_playouts);
        MAXSIMULATIONTIME = lookahead;
        playoutPolicy = policy;
        openingBookPolicy = new LightRush(new UnitTypeTable());
        MAX_TREE_DEPTH = max_depth;
        initial_epsilon_l = epsilon_l = e_l;
        initial_epsilon_g = epsilon_g = e_g;
        initial_epsilon_0 = epsilon_0 = e_0;
        discount_l = discout_l;
        discount_g = discout_g;
        discount_0 = discout_0;
        ef = a_ef;
        forceExplorationOfNonSampledActions = fensa;
    }    

    public XieBot(int available_time, int max_playouts, int lookahead, int max_depth, float e_l, float e_g, float e_0, AI policy, EvaluationFunction a_ef, boolean fensa) {
        super(available_time, max_playouts);
        MAXSIMULATIONTIME = lookahead;
        playoutPolicy = policy;
        openingBookPolicy = new LightRush(new UnitTypeTable());
        MAX_TREE_DEPTH = max_depth;
        initial_epsilon_l = epsilon_l = e_l;
        initial_epsilon_g = epsilon_g = e_g;
        initial_epsilon_0 = epsilon_0 = e_0;
        discount_l = 1.0f;
        discount_g = 1.0f;
        discount_0 = 1.0f;
        ef = a_ef;
        forceExplorationOfNonSampledActions = fensa;
    }    
    
    public XieBot(int available_time, int max_playouts, int lookahead, int max_depth, float e_l, float e_g, float e_0, int a_global_strategy, AI policy, EvaluationFunction a_ef, boolean fensa) {
        super(available_time, max_playouts);
        MAXSIMULATIONTIME = lookahead;
        playoutPolicy = policy;
        openingBookPolicy = new LightRush(new UnitTypeTable());
        MAX_TREE_DEPTH = max_depth;
        initial_epsilon_l = epsilon_l = e_l;
        initial_epsilon_g = epsilon_g = e_g;
        initial_epsilon_0 = epsilon_0 = e_0;
        discount_l = 1.0f;
        discount_g = 1.0f;
        discount_0 = 1.0f;
        global_strategy = a_global_strategy;
        ef = a_ef;
        forceExplorationOfNonSampledActions = fensa;
    }        
    
    public void reset() {
        tree = null;
        gs_to_start_from = null;
        total_runs = 0;
        total_cycles_executed = 0;
        total_actions_issued = 0;
        total_time = 0;
        current_iteration = 0;
        currentDecision = new StrategyDecision(MacroStrategy.LIGHT_RUSH, 0.6f, 120, false);
        lastStrategyDecisionTime = -9999;
        lastDecisionBaseCount = -1;
        lastDecisionBarracksCount = -1;
        previousEmergencyState = false;
        if (lightRushPolicy != null) lightRushPolicy.reset();
        if (workerRushPolicy != null) workerRushPolicy.reset();
        if (boomEconomyPolicy != null) boomEconomyPolicy.reset();
        if (turtleDefensePolicy != null) turtleDefensePolicy.reset();
        if (counterAttackPolicy != null) counterAttackPolicy.reset();
    }    
        
    
    public AI clone() {
    return new XieBot(TIME_BUDGET, ITERATIONS_BUDGET, MAXSIMULATIONTIME, MAX_TREE_DEPTH,
                     epsilon_l, discount_l, epsilon_g, discount_g, epsilon_0, discount_0,
                     playoutPolicy, ef, forceExplorationOfNonSampledActions);
} 
    
    
    public PlayerAction getAction(int player, GameState gs) throws Exception
    {
        ensureStrategyPolicies(gs.getUnitTypeTable());
        refreshStrategicDecision(player, gs);
        applyDecisionBudgetsAndPolicy(gs.getTime());

        if (gs.getTime() < STRATEGY_START_TICK) {
            openingBookPolicy.reset(gs.getUnitTypeTable());
            return openingBookPolicy.getAction(player, gs);
        }

        if (gs.canExecuteAnyAction(player)) {
            startNewComputation(player,gs.clone());
            computeDuringOneGameFrame();
            return getBestActionSoFar();
        } else {
            return new PlayerAction();        
        }       
    }
    
    
    public void startNewComputation(int a_player, GameState gs) throws Exception {
        player = a_player;
        current_iteration = 0;
        tree = new NaiveMCTSNode(player, 1-player, gs, null, ef.upperBound(gs), current_iteration++, forceExplorationOfNonSampledActions);
        
        if (tree.moveGenerator==null) {
            max_actions_so_far = 0;
        } else {
            max_actions_so_far = Math.max(tree.moveGenerator.getSize(),max_actions_so_far);        
        }
        gs_to_start_from = gs;
        
        epsilon_l = initial_epsilon_l;
        epsilon_g = initial_epsilon_g;
        epsilon_0 = initial_epsilon_0;        
    }    


    private void applyDecisionBudgetsAndPolicy(int time) {
        if (time < STRATEGY_START_TICK) {
            TIME_BUDGET = 150;
            MAX_TREE_DEPTH = 12;
        } else {
            TIME_BUDGET = clampInt(currentDecision.timeBudget, MIN_TIME_BUDGET, MAX_TIME_BUDGET);
            if (currentDecision.strategy == MacroStrategy.TURTLE_DEFENSE || currentDecision.strategy == MacroStrategy.BOOM_ECONOMY) {
                MAX_TREE_DEPTH = 11;
            } else if (time < 1500) {
                MAX_TREE_DEPTH = 10;
            } else {
                MAX_TREE_DEPTH = 8;
            }
        }
    }


    private void ensureStrategyPolicies(UnitTypeTable utt) {
        if (strategyUTT == utt && lightRushPolicy != null && workerRushPolicy != null
                && boomEconomyPolicy != null && turtleDefensePolicy != null && counterAttackPolicy != null) {
            return;
        }
        strategyUTT = utt;
        lightRushPolicy = new LightRush(utt);
        workerRushPolicy = new WorkerRush(utt);
        boomEconomyPolicy = new BoomEconomy(utt);
        turtleDefensePolicy = new TurtleDefense(utt);
        counterAttackPolicy = new RangedRush(utt);
        playoutPolicy = lightRushPolicy;
    }


    private void refreshStrategicDecision(int player, GameState gs) {
        if (gs.getTime() < STRATEGY_START_TICK) return;

        UnitCounts myCounts = countUnits(gs, player);
        UnitCounts enemyCounts = countEnemyUnitsForPrompt(gs, player);
        boolean enemyCountsKnown = !(gs instanceof PartiallyObservableGameState);
        boolean enemyNearMyBase = isEnemyNearMyBase(gs, player, ENEMY_NEAR_BASE_RADIUS);
        boolean lostCriticalStructure = (lastDecisionBaseCount >= 0 && myCounts.bases < lastDecisionBaseCount)
                || (lastDecisionBarracksCount >= 0 && myCounts.barracks < lastDecisionBarracksCount);
        boolean enemyNearTrigger = enemyNearMyBase && !previousEmergencyState;
        boolean emergencyTrigger = lostCriticalStructure || enemyNearTrigger;
        previousEmergencyState = enemyNearMyBase;

        boolean intervalElapsed = (gs.getTime() - lastStrategyDecisionTime) >= STRATEGY_INTERVAL;
        if (!intervalElapsed && !emergencyTrigger) return;

        List<MacroStrategy> shortlist = buildStrategyShortlist(gs, player, myCounts, enemyCounts, enemyCountsKnown, enemyNearMyBase);
        StrategyDecision newDecision = null;
        if (!shortlist.isEmpty()) {
            newDecision = requestLLMDecision(gs, player, shortlist, myCounts, enemyNearMyBase);
        }
        if (newDecision == null) {
            newDecision = getHeuristicFallbackDecision(gs.getTime(), enemyNearMyBase);
        }

        applyStrategicPolicy(newDecision);
        currentDecision = newDecision;
        lastStrategyDecisionTime = gs.getTime();
        lastDecisionBaseCount = myCounts.bases;
        lastDecisionBarracksCount = myCounts.barracks;
        System.out.println("XieBot strategy=" + currentDecision.strategy + " source=" + (currentDecision.fromLLM ? "LLM" : "fallback"));
    }


    private void applyStrategicPolicy(StrategyDecision decision) {
        AI nextPolicy = playoutPolicy;
        switch (decision.strategy) {
            case LIGHT_RUSH:
                nextPolicy = lightRushPolicy;
                break;
            case WORKER_RUSH:
                nextPolicy = workerRushPolicy;
                break;
            case BOOM_ECONOMY:
                nextPolicy = boomEconomyPolicy;
                break;
            case TURTLE_DEFENSE:
                nextPolicy = turtleDefensePolicy;
                break;
            case COUNTER_ATTACK:
                nextPolicy = counterAttackPolicy != null ? counterAttackPolicy : playoutPolicy;
                break;
            default:
                break;
        }
        if (nextPolicy != null && nextPolicy != playoutPolicy) {
            nextPolicy.reset();
            playoutPolicy = nextPolicy;
        }
    }


    private StrategyDecision requestLLMDecision(GameState gs, int player, List<MacroStrategy> shortlist, UnitCounts myCounts, boolean enemyNearMyBase) {
        try {
            String prompt = buildPrompt(gs, player, shortlist, myCounts, enemyNearMyBase);
            String llmRaw = callOllamaChat(prompt);
            if (llmRaw == null) return null;
            StrategyDecision parsed = parseDecisionJson(llmRaw);
            if (parsed == null) return null;
            if (!shortlist.contains(parsed.strategy)) return null;
            parsed.fromLLM = true;
            return parsed;
        } catch (Exception e) {
            return null;
        }
    }


    private StrategyDecision getHeuristicFallbackDecision(int time, boolean enemyNearMyBase) {
        if (enemyNearMyBase) {
            return new StrategyDecision(MacroStrategy.TURTLE_DEFENSE, 0.2f, 170, false);
        }
        if (time < 1200) {
            return new StrategyDecision(MacroStrategy.LIGHT_RUSH, 0.7f, 120, false);
        }
        return new StrategyDecision(MacroStrategy.COUNTER_ATTACK, 0.55f, 100, false);
    }


    private List<MacroStrategy> buildStrategyShortlist(GameState gs, int player, UnitCounts myCounts, UnitCounts enemyCounts, boolean enemyCountsKnown, boolean enemyNearMyBase) {
        LinkedHashSet<MacroStrategy> shortlist = new LinkedHashSet<>();
        int time = gs.getTime();
        int resources = gs.getPlayer(player).getResources();
        int military = myCounts.militaryCount();
        int enemyMilitary = enemyCounts.militaryCount();

        if (enemyNearMyBase) {
            shortlist.add(MacroStrategy.TURTLE_DEFENSE);
            shortlist.add(MacroStrategy.COUNTER_ATTACK);
            if (myCounts.barracks > 0) shortlist.add(MacroStrategy.LIGHT_RUSH);
        } else if (time < 1200) {
            shortlist.add(MacroStrategy.LIGHT_RUSH);
            if (myCounts.barracks == 0 || myCounts.workers <= 2) {
                shortlist.add(MacroStrategy.WORKER_RUSH);
            } else {
                shortlist.add(MacroStrategy.COUNTER_ATTACK);
            }
            if (resources >= 8 && myCounts.workers >= 3) shortlist.add(MacroStrategy.BOOM_ECONOMY);
        } else if (time < 2200) {
            shortlist.add(MacroStrategy.COUNTER_ATTACK);
            shortlist.add(MacroStrategy.LIGHT_RUSH);
            if (resources >= 8 && myCounts.workers >= 3) shortlist.add(MacroStrategy.BOOM_ECONOMY);
        } else {
            shortlist.add(MacroStrategy.COUNTER_ATTACK);
            shortlist.add(MacroStrategy.TURTLE_DEFENSE);
            if (military <= 2) shortlist.add(MacroStrategy.BOOM_ECONOMY);
        }

        if (enemyCountsKnown && enemyMilitary > military + 1) {
            shortlist.add(MacroStrategy.TURTLE_DEFENSE);
            shortlist.add(MacroStrategy.COUNTER_ATTACK);
        }
        if (enemyCountsKnown && enemyCounts.bases == 0) {
            shortlist.add(MacroStrategy.COUNTER_ATTACK);
        }

        if (shortlist.size() < 2) {
            shortlist.add(MacroStrategy.LIGHT_RUSH);
            shortlist.add(MacroStrategy.COUNTER_ATTACK);
        }

        List<MacroStrategy> result = new ArrayList<>();
        for (MacroStrategy s : shortlist) {
            result.add(s);
            if (result.size() == 3) break;
        }
        return result;
    }


    private String buildPrompt(GameState gs, int player, List<MacroStrategy> shortlist, UnitCounts myCounts, boolean enemyNearMyBase) {
        UnitCounts enemyCounts = countEnemyUnitsForPrompt(gs, player);
        boolean enemyCountsKnown = !(gs instanceof PartiallyObservableGameState);
        PhysicalGameState pgs = gs.getPhysicalGameState();

        String timeBucket;
        if (gs.getTime() < 1200) {
            timeBucket = "early";
        } else if (gs.getTime() < 2200) {
            timeBucket = "mid";
        } else {
            timeBucket = "late";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("You are macro strategist for microRTS.\n");
        sb.append("Pick ONLY one strategy from candidates: ");
        sb.append(shortlist.toString());
        sb.append("\n");
        sb.append("Output STRICT JSON only with EXACT fields and no extra text:\n");
        sb.append("{\"strategy\":\"LIGHT_RUSH\",\"aggression\":0.6,\"timeBudget\":120}\n");
        sb.append("strategy must be one of candidates.\n");
        sb.append("aggression must be float in [0,1]. timeBudget must be int in [50,200].\n");
        sb.append("Game summary:\n");
        sb.append("time=").append(gs.getTime()).append(", timeBucket=").append(timeBucket).append("\n");
        sb.append("map=").append(pgs.getWidth()).append("x").append(pgs.getHeight()).append("\n");
        sb.append("myResources=").append(gs.getPlayer(player).getResources()).append("\n");
        sb.append("myCounts workers=").append(myCounts.workers)
                .append(", bases=").append(myCounts.bases)
                .append(", barracks=").append(myCounts.barracks)
                .append(", light=").append(myCounts.lights)
                .append(", ranged=").append(myCounts.ranged)
                .append(", heavy=").append(myCounts.heavy).append("\n");
        if (enemyCountsKnown) {
            sb.append("enemyCounts workers=").append(enemyCounts.workers)
                    .append(", bases=").append(enemyCounts.bases)
                    .append(", barracks=").append(enemyCounts.barracks)
                    .append(", light=").append(enemyCounts.lights)
                    .append(", ranged=").append(enemyCounts.ranged)
                    .append(", heavy=").append(enemyCounts.heavy).append("\n");
        } else {
            sb.append("enemyCounts=unknown\n");
        }
        sb.append("enemyNearMyBase=").append(enemyNearMyBase).append("\n");
        return sb.toString();
    }


    private String callOllamaChat(String prompt) throws IOException {
        HttpURLConnection connection = null;
        try {
            URL url = new URL(OLLAMA_ENDPOINT);
            connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setConnectTimeout(OLLAMA_CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(OLLAMA_READ_TIMEOUT_MS);
            connection.setDoOutput(true);
            connection.setRequestProperty("Content-Type", "application/json");

            String body = "{\"model\":\"" + OLLAMA_MODEL + "\",\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\""
                    + escapeJson(prompt) + "\"}]}";
            try (OutputStream os = connection.getOutputStream()) {
                os.write(body.getBytes(StandardCharsets.UTF_8));
            }

            int code = connection.getResponseCode();
            InputStream stream = (code >= 200 && code < 300) ? connection.getInputStream() : connection.getErrorStream();
            if (stream == null) return null;
            String response = readAll(stream);
            if (code < 200 || code >= 300) return null;
            String assistantContent = extractAssistantContent(response);
            return assistantContent != null ? assistantContent : response;
        } finally {
            if (connection != null) connection.disconnect();
        }
    }


    private StrategyDecision parseDecisionJson(String text) {
        if (text == null) return null;
        String json = extractFirstJsonObject(text.trim());
        if (json == null) return null;

        Matcher strategyMatcher = STRATEGY_JSON_PATTERN.matcher(json);
        Matcher aggressionMatcher = AGGRESSION_JSON_PATTERN.matcher(json);
        Matcher timeMatcher = TIME_BUDGET_JSON_PATTERN.matcher(json);
        if (!strategyMatcher.find() || !aggressionMatcher.find() || !timeMatcher.find()) return null;

        MacroStrategy parsedStrategy;
        try {
            parsedStrategy = MacroStrategy.valueOf(strategyMatcher.group(1));
        } catch (IllegalArgumentException e) {
            return null;
        }

        float aggressionValue;
        int budgetValue;
        try {
            aggressionValue = Float.parseFloat(aggressionMatcher.group(1));
            budgetValue = Integer.parseInt(timeMatcher.group(1));
        } catch (NumberFormatException e) {
            return null;
        }

        aggressionValue = clampFloat(aggressionValue, 0f, 1f);
        budgetValue = clampInt(budgetValue, MIN_TIME_BUDGET, MAX_TIME_BUDGET);
        return new StrategyDecision(parsedStrategy, aggressionValue, budgetValue, true);
    }


    private String extractAssistantContent(String response) {
        int messageIdx = response.indexOf("\"message\"");
        if (messageIdx < 0) return null;
        int contentIdx = response.indexOf("\"content\"", messageIdx);
        if (contentIdx < 0) return null;
        int colonIdx = response.indexOf(':', contentIdx);
        if (colonIdx < 0) return null;
        int quoteStart = response.indexOf('"', colonIdx + 1);
        if (quoteStart < 0) return null;

        StringBuilder sb = new StringBuilder();
        boolean escaped = false;
        for (int i = quoteStart + 1; i < response.length(); i++) {
            char c = response.charAt(i);
            if (escaped) {
                switch (c) {
                    case '"':
                        sb.append('"');
                        break;
                    case '\\':
                        sb.append('\\');
                        break;
                    case 'n':
                        sb.append('\n');
                        break;
                    case 'r':
                        sb.append('\r');
                        break;
                    case 't':
                        sb.append('\t');
                        break;
                    default:
                        sb.append(c);
                        break;
                }
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                return sb.toString();
            } else {
                sb.append(c);
            }
        }
        return null;
    }


    private String extractFirstJsonObject(String text) {
        int start = text.indexOf('{');
        if (start < 0) return null;
        int depth = 0;
        boolean inString = false;
        boolean escaped = false;
        for (int i = start; i < text.length(); i++) {
            char c = text.charAt(i);
            if (inString) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    inString = false;
                }
            } else {
                if (c == '"') {
                    inString = true;
                } else if (c == '{') {
                    depth++;
                } else if (c == '}') {
                    depth--;
                    if (depth == 0) {
                        return text.substring(start, i + 1);
                    }
                }
            }
        }
        return null;
    }


    private String escapeJson(String text) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            switch (c) {
                case '\\':
                    sb.append("\\\\");
                    break;
                case '"':
                    sb.append("\\\"");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    sb.append(c);
                    break;
            }
        }
        return sb.toString();
    }


    private String readAll(InputStream stream) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                sb.append(line);
            }
        }
        return sb.toString();
    }


    private UnitCounts countUnits(GameState gs, int owner) {
        UnitCounts counts = new UnitCounts();
        UnitTypeTable utt = gs.getUnitTypeTable();
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() != owner) continue;
            if (u.getType() == utt.getUnitType("Worker")) counts.workers++;
            else if (u.getType() == utt.getUnitType("Base")) counts.bases++;
            else if (u.getType() == utt.getUnitType("Barracks")) counts.barracks++;
            else if (u.getType() == utt.getUnitType("Light")) counts.lights++;
            else if (u.getType() == utt.getUnitType("Ranged")) counts.ranged++;
            else if (u.getType() == utt.getUnitType("Heavy")) counts.heavy++;
        }
        return counts;
    }


    private UnitCounts countEnemyUnitsForPrompt(GameState gs, int player) {
        if (gs instanceof PartiallyObservableGameState) {
            return new UnitCounts();
        }
        return countUnits(gs, 1 - player);
    }


    private boolean isEnemyNearMyBase(GameState gs, int player, int radius) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        List<Unit> anchors = new ArrayList<>();
        UnitTypeTable utt = gs.getUnitTypeTable();
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player && (u.getType() == utt.getUnitType("Base") || u.getType() == utt.getUnitType("Barracks"))) {
                anchors.add(u);
            }
        }
        if (anchors.isEmpty()) return false;

        for (Unit enemy : pgs.getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == player) continue;
            for (Unit anchor : anchors) {
                int dist = Math.abs(enemy.getX() - anchor.getX()) + Math.abs(enemy.getY() - anchor.getY());
                if (dist <= radius) return true;
            }
        }
        return false;
    }


    private int clampInt(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }


    private float clampFloat(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }
    
    
    public void resetSearch() {
        if (DEBUG>=2) System.out.println("Resetting search...");
        tree = null;
        gs_to_start_from = null;
    }
    

    public void computeDuringOneGameFrame() throws Exception {        
        if (DEBUG>=2) System.out.println("Search...");
        long start = System.currentTimeMillis();
        long end = start;
        long count = 0;
        while(true) {
            if (!iteration(player)) break;
            count++;
            end = System.currentTimeMillis();
            if (TIME_BUDGET>=0 && (end - start)>=TIME_BUDGET) break; 
            if (ITERATIONS_BUDGET>=0 && count>=ITERATIONS_BUDGET) break;             
        }
//        System.out.println("HL: " + count + " time: " + (System.currentTimeMillis() - start) + " (" + available_time + "," + max_playouts + ")");
        total_time += (end - start);
        total_cycles_executed++;
    }
    
    
    public boolean iteration(int player) throws Exception {
        
        NaiveMCTSNode leaf = tree.selectLeaf(player, 1-player, epsilon_l, epsilon_g, epsilon_0, global_strategy, MAX_TREE_DEPTH, current_iteration++);

        if (leaf!=null) {            
            GameState gs2 = leaf.gs.clone();
            simulate(gs2, gs2.getTime() + MAXSIMULATIONTIME);

            int time = gs2.getTime() - gs_to_start_from.getTime();
            double evaluation = ef.evaluate(player, 1-player, gs2)*Math.pow(0.99,time/10.0);

            leaf.propagateEvaluation(evaluation,null);            

            // update the epsilon values:
            epsilon_0*=discount_0;
            epsilon_l*=discount_l;
            epsilon_g*=discount_g;
            total_runs++;
            
//            System.out.println(total_runs + " - " + epsilon_0 + ", " + epsilon_l + ", " + epsilon_g);
            
        } else {
            // no actions to choose from :)
            System.err.println(this.getClass().getSimpleName() + ": claims there are no more leafs to explore...");
            return false;
        }
        return true;
    }
    
    public PlayerAction getBestActionSoFar() {
        int idx = getAggressionBiasedActionIdx();
        if (idx==-1) {
            if (DEBUG>=1) System.out.println("XieBot no children selected. Returning an empty action");
            return new PlayerAction();
        }
        if (DEBUG>=2) tree.showNode(0,1,ef);
        if (DEBUG>=1) {
            NaiveMCTSNode best = (NaiveMCTSNode) tree.children.get(idx);
            System.out.println("NaiveMCTS selected children " + tree.actions.get(idx) + " explored " + best.visit_count + " Avg evaluation: " + (best.accum_evaluation/((double)best.visit_count)));
        }
        return tree.actions.get(idx);
    }
    
    
    private int getAggressionBiasedActionIdx() {
        int mostVisited = getMostVisitedActionIdx();
        if (mostVisited == -1 || tree.children == null || tree.children.isEmpty()) return mostVisited;
        float aggression = currentDecision != null ? currentDecision.aggression : 0.5f;
        if (aggression >= 0.67f) {
            return selectTopVisitedByPressure(3, true);
        }
        if (aggression <= 0.33f) {
            return selectTopVisitedByPressure(3, false);
        }
        return mostVisited;
    }


    private int selectTopVisitedByPressure(int topN, boolean aggressiveMode) {
        List<Integer> topIndices = new ArrayList<>();
        for (int i = 0; i < tree.children.size(); i++) {
            insertByVisitCount(topIndices, i, topN);
        }

        int bestIdx = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int idx : topIndices) {
            NaiveMCTSNode child = (NaiveMCTSNode) tree.children.get(idx);
            double score = aggressiveMode ? pressureEnemyBaseScore(child.gs) : defenseStabilityScore(child.gs);
            if (bestIdx == -1 || score > bestScore) {
                bestIdx = idx;
                bestScore = score;
            }
        }
        return bestIdx == -1 ? getMostVisitedActionIdx() : bestIdx;
    }


    private void insertByVisitCount(List<Integer> topIndices, int idx, int topN) {
        NaiveMCTSNode node = (NaiveMCTSNode) tree.children.get(idx);
        int insertPos = 0;
        while (insertPos < topIndices.size()) {
            NaiveMCTSNode current = (NaiveMCTSNode) tree.children.get(topIndices.get(insertPos));
            if (node.visit_count > current.visit_count) break;
            insertPos++;
        }
        topIndices.add(insertPos, idx);
        if (topIndices.size() > topN) {
            topIndices.remove(topIndices.size() - 1);
        }
    }


    private double pressureEnemyBaseScore(GameState gs) {
        UnitTypeTable utt = gs.getUnitTypeTable();
        Unit enemyBase = null;
        int myMilitary = 0;
        int closestDist = Integer.MAX_VALUE;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == 1 - player && u.getType() == utt.getUnitType("Base")) {
                enemyBase = u;
                break;
            }
        }
        if (enemyBase == null) return 0.0;

        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() != player) continue;
            if (!u.getType().canAttack) continue;
            myMilitary++;
            int d = Math.abs(u.getX() - enemyBase.getX()) + Math.abs(u.getY() - enemyBase.getY());
            if (d < closestDist) closestDist = d;
        }
        if (myMilitary == 0) return 0.0;
        return (1.0 / (1.0 + closestDist)) + (0.04 * myMilitary);
    }


    private double defenseStabilityScore(GameState gs) {
        UnitTypeTable utt = gs.getUnitTypeTable();
        Unit myBase = null;
        int defendersNearBase = 0;
        int enemyNearBase = 0;
        int nearestEnemy = Integer.MAX_VALUE;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && u.getType() == utt.getUnitType("Base")) {
                myBase = u;
                break;
            }
        }
        if (myBase == null) return 0.0;

        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            int dist = Math.abs(u.getX() - myBase.getX()) + Math.abs(u.getY() - myBase.getY());
            if (u.getPlayer() == player && u.getType().canAttack && dist <= ENEMY_NEAR_BASE_RADIUS) {
                defendersNearBase++;
            } else if (u.getPlayer() >= 0 && u.getPlayer() != player) {
                if (dist <= ENEMY_NEAR_BASE_RADIUS) enemyNearBase++;
                if (dist < nearestEnemy) nearestEnemy = dist;
            }
        }
        double enemyDistanceTerm = nearestEnemy == Integer.MAX_VALUE ? 0.5 : (nearestEnemy / 20.0);
        return defendersNearBase * 0.08 - enemyNearBase * 0.2 + enemyDistanceTerm;
    }


    public int getMostVisitedActionIdx() {
        total_actions_issued++;
            
        int bestIdx = -1;
        NaiveMCTSNode best = null;
        if (DEBUG>=2) {
//            for(Player p:gs_to_start_from.getPlayers()) {
//                System.out.println("Resources P" + p.getID() + ": " + p.getResources());
//            }
            System.out.println("Number of playouts: " + tree.visit_count);
            tree.printUnitActionTable();
        }
        if (tree.children==null) return -1;
        for(int i = 0;i<tree.children.size();i++) {
            NaiveMCTSNode child = (NaiveMCTSNode)tree.children.get(i);
            if (DEBUG>=2) {
                System.out.println("child " + tree.actions.get(i) + " explored " + child.visit_count + " Avg evaluation: " + (child.accum_evaluation/((double)child.visit_count)));
            }
//            if (best == null || (child.accum_evaluation/child.visit_count)>(best.accum_evaluation/best.visit_count)) {
            if (best == null || child.visit_count>best.visit_count) {
                best = child;
                bestIdx = i;
            }
        }
        
        return bestIdx;
    }
    
    
    public int getHighestEvaluationActionIdx() {
        total_actions_issued++;
            
        int bestIdx = -1;
        NaiveMCTSNode best = null;
        if (DEBUG>=2) {
//            for(Player p:gs_to_start_from.getPlayers()) {
//                System.out.println("Resources P" + p.getID() + ": " + p.getResources());
//            }
            System.out.println("Number of playouts: " + tree.visit_count);
            tree.printUnitActionTable();
        }
        for(int i = 0;i<tree.children.size();i++) {
            NaiveMCTSNode child = (NaiveMCTSNode)tree.children.get(i);
            if (DEBUG>=2) {
                System.out.println("child " + tree.actions.get(i) + " explored " + child.visit_count + " Avg evaluation: " + (child.accum_evaluation/((double)child.visit_count)));
            }
//            if (best == null || (child.accum_evaluation/child.visit_count)>(best.accum_evaluation/best.visit_count)) {
            if (best == null || (child.accum_evaluation/((double)child.visit_count))>(best.accum_evaluation/((double)best.visit_count))) {
                best = child;
                bestIdx = i;
            }
        }
        
        return bestIdx;
    }
    
        
    public void simulate(GameState gs, int time) throws Exception {
        PrintStream originalOut = null;
        PrintStream mutedOut = null;
        if (SILENCE_PLAYOUT_STDOUT) {
            originalOut = System.out;
            mutedOut = new PrintStream(OutputStream.nullOutputStream());
            System.setOut(mutedOut);
        }

        boolean gameover = false;
        AI simPolicyP0 = playoutPolicy != null ? playoutPolicy.clone() : null;
        AI simPolicyP1 = playoutPolicy != null ? playoutPolicy.clone() : null;
        AI fallbackP0 = strategyUTT != null ? new LightRush(strategyUTT) : new LightRush(gs.getUnitTypeTable());
        AI fallbackP1 = strategyUTT != null ? new LightRush(strategyUTT) : new LightRush(gs.getUnitTypeTable());

        try {
            int noProgressFrames = 0;
            int lastTime = gs.getTime();
            do{
                if (!gs.isComplete()) {
                    PlayerAction p0Action = safePlayoutAction(simPolicyP0, fallbackP0, 0, gs);
                    PlayerAction p1Action = safePlayoutAction(simPolicyP1, fallbackP1, 1, gs);
                    p0Action.fillWithNones(gs, 0, 1);
                    p1Action.fillWithNones(gs, 1, 1);
                    gs.issue(p0Action);
                    gs.issue(p1Action);
                }
                if (gs.isComplete()) {
                    gameover = gs.cycle();
                } else {
                    // Prevent rare bad-script states from stalling simulation forever.
                    PlayerAction forceP0 = new PlayerAction();
                    PlayerAction forceP1 = new PlayerAction();
                    forceP0.fillWithNones(gs, 0, 1);
                    forceP1.fillWithNones(gs, 1, 1);
                    gs.issue(forceP0);
                    gs.issue(forceP1);
                }

                if (gs.getTime() == lastTime) {
                    noProgressFrames++;
                    if (noProgressFrames > 8) break;
                } else {
                    noProgressFrames = 0;
                    lastTime = gs.getTime();
                }
            }while(!gameover && gs.getTime()<time);
        } finally {
            if (SILENCE_PLAYOUT_STDOUT) {
                System.setOut(originalOut);
                if (mutedOut != null) mutedOut.close();
            }
        }
    }


    private PlayerAction safePlayoutAction(AI primary, AI fallback, int simPlayer, GameState gs) {
        try {
            if (primary != null) {
                PlayerAction pa = primary.getAction(simPlayer, gs);
                if (pa != null) return pa;
            }
        } catch (Exception ignored) {
            // Keep simulation alive even if a rollout script throws.
        }
        try {
            PlayerAction pa = fallback.getAction(simPlayer, gs);
            return pa != null ? pa : new PlayerAction();
        } catch (Exception ignored) {
            return new PlayerAction();
        }
    }
    
    public NaiveMCTSNode getTree() {
        return tree;
    }
    
    public GameState getGameStateToStartFrom() {
        return gs_to_start_from;
    }
    
    
    @Override
    public String toString() {
        return getClass().getSimpleName() + "(" + TIME_BUDGET + ", " + ITERATIONS_BUDGET + ", " + MAXSIMULATIONTIME + "," + MAX_TREE_DEPTH + "," + epsilon_l + ", " + discount_l + ", " + epsilon_g + ", " + discount_g + ", " + epsilon_0 + ", " + discount_0 + ", " + playoutPolicy + ", " + ef + ")";
    }
    
    @Override
    public String statisticsString() {
        return "Total runs: " + total_runs + 
               ", runs per action: " + (total_runs/(float)total_actions_issued) + 
               ", runs per cycle: " + (total_runs/(float)total_cycles_executed) + 
               ", average time per cycle: " + (total_time/(float)total_cycles_executed) + 
               ", max branching factor: " + max_actions_so_far;
    }
    
    
    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        
        parameters.add(new ParameterSpecification("TimeBudget",int.class,100));
        parameters.add(new ParameterSpecification("IterationsBudget",int.class,-1));
        parameters.add(new ParameterSpecification("PlayoutLookahead",int.class,100));
        parameters.add(new ParameterSpecification("MaxTreeDepth",int.class,10));
        
        parameters.add(new ParameterSpecification("E_l",float.class,0.3));
        parameters.add(new ParameterSpecification("Discount_l",float.class,1.0));
        parameters.add(new ParameterSpecification("E_g",float.class,0.0));
        parameters.add(new ParameterSpecification("Discount_g",float.class,1.0));
        parameters.add(new ParameterSpecification("E_0",float.class,0.4));
        parameters.add(new ParameterSpecification("Discount_0",float.class,1.0));
                
        parameters.add(new ParameterSpecification("DefaultPolicy",AI.class, playoutPolicy));
        parameters.add(new ParameterSpecification("EvaluationFunction", EvaluationFunction.class, new SimpleSqrtEvaluationFunction3()));

        parameters.add(new ParameterSpecification("ForceExplorationOfNonSampledActions",boolean.class,true));
        
        return parameters;
    }    
    
    
    public int getPlayoutLookahead() {
        return MAXSIMULATIONTIME;
    }
    
    
    public void setPlayoutLookahead(int a_pola) {
        MAXSIMULATIONTIME = a_pola;
    }


    public int getMaxTreeDepth() {
        return MAX_TREE_DEPTH;
    }
    
    
    public void setMaxTreeDepth(int a_mtd) {
        MAX_TREE_DEPTH = a_mtd;
    }
    
    
    public float getE_l() {
        return epsilon_l;
    }
    
    
    public void setE_l(float a_e_l) {
        epsilon_l = a_e_l;
    }


    public float getDiscount_l() {
        return discount_l;
    }
    
    
    public void setDiscount_l(float a_discount_l) {
        discount_l = a_discount_l;
    }


    public float getE_g() {
        return epsilon_g;
    }
    
    
    public void setE_g(float a_e_g) {
        epsilon_g = a_e_g;
    }


    public float getDiscount_g() {
        return discount_g;
    }
    
    
    public void setDiscount_g(float a_discount_g) {
        discount_g = a_discount_g;
    }


    public float getE_0() {
        return epsilon_0;
    }
    
    
    public void setE_0(float a_e_0) {
        epsilon_0 = a_e_0;
    }


    public float getDiscount_0() {
        return discount_0;
    }
    
    
    public void setDiscount_0(float a_discount_0) {
        discount_0 = a_discount_0;
    }
    
    
    
    public AI getDefaultPolicy() {
        return playoutPolicy;
    }
    
    
    public void setDefaultPolicy(AI a_dp) {
        playoutPolicy = a_dp;
    }
    
    
    public EvaluationFunction getEvaluationFunction() {
        return ef;
    }
    
    
    public void setEvaluationFunction(EvaluationFunction a_ef) {
        ef = a_ef;
    }
    
    public boolean getForceExplorationOfNonSampledActions() {
        return forceExplorationOfNonSampledActions;
    }
    
    public void setForceExplorationOfNonSampledActions(boolean fensa)
    {
        forceExplorationOfNonSampledActions = fensa;
    }    
}
