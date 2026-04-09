/* MicroRTS LLM Competition - Leaderboard App */

(function () {
    'use strict';

    var DATA_URL = 'data/tournament_results.json';
    var ANCHORS = ['RandomBiasedAI', 'HeavyRush', 'LightRush', 'WorkerRush', 'Tiamat', 'CoacAI'];
    var ANCHOR_CLASSES = {
        'RandomBiasedAI': 'ai.RandomBiasedAI',
        'HeavyRush': 'ai.abstraction.HeavyRush',
        'LightRush': 'ai.abstraction.LightRush',
        'WorkerRush': 'ai.abstraction.WorkerRush',
        'Tiamat': 'ai.competition.tiamat.Tiamat',
        'CoacAI': 'ai.coac.CoacAI'
    };
    // Fallback mapping from agent type keywords in display_name to Java classes
    var AGENT_TYPE_CLASSES = {
        'PureLLM': 'ai.abstraction.ollama',
        'Hybrid': 'ai.abstraction.HybridLLMRush',
        'Search+LLM': 'ai.mcts.llmguided.LLMInformedMCTS'
    };
    var GRADE_ORDER = { 'A+': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'F': 5 };

    // State
    var sortState = {};
    var leaderboardEntries = [];
    var benchmarkEntries = [];
    var historyEntries = [];
    var dataLoaded = false;
    var expandedRows = {}; // tableId -> expanded index (or null)

    // --- Utilities ---

    function escapeHtml(str) {
        if (str == null) return '';
        return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function gradeBadge(grade) {
        var cls = 'grade-f';
        if (grade === 'A+') cls = 'grade-aplus';
        else if (grade === 'A') cls = 'grade-a';
        else if (grade === 'B') cls = 'grade-b';
        else if (grade === 'C') cls = 'grade-c';
        else if (grade === 'D') cls = 'grade-d';
        return '<span class="grade ' + cls + '">' + escapeHtml(grade) + '</span>';
    }

    function opponentCell(opponents, anchorName) {
        var data = opponents ? opponents[anchorName] : null;
        if (!data) return '<td class="result-skip">--</td>';
        var w = data.wins || 0;
        var d = data.draws || 0;
        var l = data.losses || 0;
        var pts = data.weighted_points != null ? data.weighted_points : '?';
        var cls = 'result-skip';
        if (w > 0) cls = 'result-win';
        else if (l > 0) cls = 'result-loss';
        else if (d > 0) cls = 'result-draw';
        return '<td class="' + cls + '">' + w + 'W/' + d + 'D/' + l + 'L<br><small>' + pts + ' pts</small></td>';
    }

    function formatDate(dateStr) {
        if (!dateStr) return '--';
        try {
            var d = new Date(dateStr);
            return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
        } catch (e) {
            return dateStr.substring(0, 10);
        }
    }

    function timeAgo(dateStr) {
        if (!dateStr) return '';
        try {
            var d = new Date(dateStr);
            var now = new Date();
            var diffMs = now - d;
            var diffMins = Math.floor(diffMs / 60000);
            var diffHours = Math.floor(diffMs / 3600000);
            var diffDays = Math.floor(diffMs / 86400000);
            if (diffMins < 1) return 'just now';
            if (diffMins < 60) return diffMins + 'm ago';
            if (diffHours < 24) return diffHours + 'h ago';
            if (diffDays === 1) return '1 day ago';
            if (diffDays < 30) return diffDays + ' days ago';
            if (diffDays < 365) return Math.floor(diffDays / 30) + ' months ago';
            return Math.floor(diffDays / 365) + ' years ago';
        } catch (e) {
            return '';
        }
    }

    function formatDateWithAge(dateStr) {
        if (!dateStr) return '--';
        var formatted = formatDate(dateStr);
        var relative = timeAgo(dateStr);
        if (relative) {
            return formatted + '<br><small class="time-ago">' + relative + '</small>';
        }
        return formatted;
    }

    function toDateOnly(dateStr) {
        if (!dateStr) return '';
        try {
            return new Date(dateStr).toISOString().substring(0, 10);
        } catch (e) {
            return '';
        }
    }

    // Object.values polyfill for older browsers
    function objValues(obj) {
        if (Object.values) return Object.values(obj);
        var vals = [];
        for (var k in obj) {
            if (obj.hasOwnProperty(k)) vals.push(obj[k]);
        }
        return vals;
    }

    // --- Sorting ---

    function comparator(type) {
        return function (a, b) {
            if (type === 'number') return (parseFloat(a) || 0) - (parseFloat(b) || 0);
            if (type === 'date') {
                var da = a ? new Date(a).getTime() : 0;
                var db = b ? new Date(b).getTime() : 0;
                return da - db;
            }
            if (type === 'grade') {
                var ga = GRADE_ORDER[a] != null ? GRADE_ORDER[a] : 99;
                var gb = GRADE_ORDER[b] != null ? GRADE_ORDER[b] : 99;
                return ga - gb;
            }
            return String(a || '').localeCompare(String(b || ''));
        };
    }

    function sortEntries(entries, key, type, dir) {
        var cmp = comparator(type);
        var sorted = entries.slice();
        sorted.sort(function (a, b) {
            var result = cmp(a._sortData[key], b._sortData[key]);
            return dir === 'asc' ? result : -result;
        });
        return sorted;
    }

    function updateSortIndicators(tableId, activeKey, dir) {
        var table = document.getElementById(tableId);
        if (!table) return;
        var ths = table.querySelectorAll('th[data-sort]');
        for (var i = 0; i < ths.length; i++) {
            ths[i].classList.remove('sort-asc', 'sort-desc');
            if (ths[i].getAttribute('data-sort') === activeKey) {
                ths[i].classList.add(dir === 'asc' ? 'sort-asc' : 'sort-desc');
            }
        }
    }

    // --- Filtering ---

    function getFilterValues(prefix) {
        var searchEl = document.getElementById(prefix + '-search');
        var fromEl = document.getElementById(prefix + '-date-from');
        var toEl = document.getElementById(prefix + '-date-to');
        return {
            search: searchEl ? searchEl.value.toLowerCase().trim() : '',
            dateFrom: fromEl ? fromEl.value : '',
            dateTo: toEl ? toEl.value : ''
        };
    }

    function matchesSearch(entry, searchStr, fields) {
        if (!searchStr) return true;
        for (var i = 0; i < fields.length; i++) {
            var val = entry._sortData[fields[i]];
            if (val && String(val).toLowerCase().indexOf(searchStr) !== -1) return true;
        }
        return false;
    }

    function matchesDateRange(dateStr, from, to) {
        if (!from && !to) return true;
        var d = toDateOnly(dateStr);
        if (!d) return !from && !to;
        if (from && d < from) return false;
        if (to && d > to) return false;
        return true;
    }

    function updateCount(id, shown, total) {
        var el = document.getElementById(id);
        if (!el) return;
        if (shown === total) {
            el.textContent = total + ' entries';
        } else {
            el.textContent = 'Showing ' + shown + ' of ' + total + ' entries';
        }
    }

    // --- Detail panel ---

    function guessAgentClass(entry) {
        if (entry.agent_class) return entry.agent_class;
        // Fallback: try to match agent type from display_name e.g. "model (Search+LLM)"
        var name = entry.display_name || '';
        for (var key in AGENT_TYPE_CLASSES) {
            if (AGENT_TYPE_CLASSES.hasOwnProperty(key) && name.indexOf(key) !== -1) {
                return AGENT_TYPE_CLASSES[key];
            }
        }
        return '';
    }

    function mapScoreCell(mapScores, label) {
        if (!mapScores || !mapScores[label]) return '<td class="score-cell">--</td>';
        var ms = mapScores[label];
        var score = ms.score != null ? ms.score : '?';
        var grade = ms.grade || '';
        return '<td class="score-cell">' + score + ' ' + gradeBadge(grade) + '</td>';
    }

    function getMapScore(mapScores, label) {
        if (!mapScores || !mapScores[label]) return null;
        return mapScores[label].score != null ? mapScores[label].score : null;
    }

    function renderMapScoresSection(mapScores) {
        if (!mapScores || typeof mapScores !== 'object') return '';
        var labels = Object.keys(mapScores);
        if (labels.length === 0) return '';

        var html = '<div class="map-scores-section">';
        html += '<h5>Per-Map Scores</h5>';
        html += '<div class="map-score-chips">';
        for (var i = 0; i < labels.length; i++) {
            var ml = labels[i];
            var ms = mapScores[ml];
            var score = ms.score != null ? ms.score : '?';
            var grade = ms.grade || '';
            var elim = ms.eliminated_at;
            var elimText = elim ? ' (elim: ' + escapeHtml(elim) + ')' : '';
            html += '<span class="map-score-chip">' + escapeHtml(ml) + ': <strong>' + score + '</strong> ' + gradeBadge(grade) + elimText + '</span>';
        }
        html += '</div></div>';
        return html;
    }

    function renderDetailPanel(entry) {
        var opponents = entry.opponents || {};
        var agentClass = guessAgentClass(entry);
        var map = entry.map || 'maps/8x8/basesWorkers8x8.xml';
        var html = '<div class="detail-panel">';
        html += '<h4>Games for ' + escapeHtml(entry.display_name) + '</h4>';

        // Show per-map scores if available
        html += renderMapScoresSection(entry.map_scores);

        html += '<div class="game-cards">';

        for (var i = 0; i < ANCHORS.length; i++) {
            var anchorName = ANCHORS[i];
            var data = opponents[anchorName];
            var anchorClass = ANCHOR_CLASSES[anchorName] || '';

            html += '<div class="game-card">';
            html += '<div class="opponent-name">vs ' + escapeHtml(anchorName) + '</div>';

            if (data) {
                var w = data.wins || 0;
                var d = data.draws || 0;
                var l = data.losses || 0;
                var badgeCls = 'badge-skip';
                var badgeText = 'N/A';
                if (w > 0) { badgeCls = 'badge-win'; badgeText = w + 'W/' + d + 'D/' + l + 'L'; }
                else if (l > 0) { badgeCls = 'badge-loss'; badgeText = w + 'W/' + d + 'D/' + l + 'L'; }
                else if (d > 0) { badgeCls = 'badge-draw'; badgeText = w + 'W/' + d + 'D/' + l + 'L'; }
                html += '<span class="result-badge ' + badgeCls + '">' + badgeText + '</span>';

                var pts = data.weighted_points != null ? data.weighted_points : '?';
                html += '<div class="game-pts">' + pts + ' pts</div>';
            } else {
                html += '<span class="result-badge badge-skip">--</span>';
                html += '<div class="game-pts">not played</div>';
            }

            if (agentClass && anchorClass) {
                var replayUrl = 'match.html?ai1=' + encodeURIComponent(agentClass) +
                    '&ai2=' + encodeURIComponent(anchorClass) +
                    '&map=' + encodeURIComponent(map);
                html += '<a class="replay-btn" href="' + replayUrl + '">Run Match</a>';
            }

            html += '</div>';
        }

        html += '</div></div>';
        return html;
    }

    function toggleDetail(tableId, idx, filteredEntries) {
        var tbody = document.querySelector('#' + tableId + ' tbody');
        if (!tbody) return;

        // Find existing detail row for this table
        var existing = tbody.querySelector('tr.detail-row');
        var existingIdx = expandedRows[tableId];

        // Remove existing detail row
        if (existing) {
            var prevRow = existing.previousElementSibling;
            if (prevRow) prevRow.classList.remove('active-row');
            existing.parentNode.removeChild(existing);
            expandedRows[tableId] = null;
        }

        // If clicking same row, just collapse
        if (existingIdx === idx) return;

        // Find the clicked row (idx-th data row in tbody)
        var rows = tbody.querySelectorAll('tr:not(.detail-row)');
        if (idx >= rows.length) return;
        var clickedRow = rows[idx];
        clickedRow.classList.add('active-row');

        // Insert detail row after clicked row
        var detailRow = document.createElement('tr');
        detailRow.className = 'detail-row';
        var colCount = clickedRow.children.length;
        var td = document.createElement('td');
        td.setAttribute('colspan', colCount);
        td.innerHTML = renderDetailPanel(filteredEntries[idx].raw);
        detailRow.appendChild(td);

        if (clickedRow.nextSibling) {
            tbody.insertBefore(detailRow, clickedRow.nextSibling);
        } else {
            tbody.appendChild(detailRow);
        }

        expandedRows[tableId] = idx;
    }

    // --- Render functions ---

    function renderLeaderboard() {
        var tbody = document.querySelector('#leaderboard-table tbody');
        if (!tbody || !dataLoaded) return;
        tbody.innerHTML = '';
        expandedRows['leaderboard-table'] = null;

        var filters = getFilterValues('leaderboard');
        var entries = [];
        for (var i = 0; i < leaderboardEntries.length; i++) {
            var e = leaderboardEntries[i];
            if (!matchesSearch(e, filters.search, ['name', 'grade'])) continue;
            if (!matchesDateRange(e._sortData.date, filters.dateFrom, filters.dateTo)) continue;
            entries.push(e);
        }

        var state = sortState['leaderboard-table'];
        if (state) {
            entries = sortEntries(entries, state.key, state.type, state.dir);
        }

        var filteredRef = entries;
        for (var j = 0; j < entries.length; j++) {
            var entry = entries[j].raw;
            var row = document.createElement('tr');
            row.setAttribute('data-idx', j);
            var displayRank = state ? (j + 1) : entries[j]._sortData.rank;
            var html = '<td class="rank-cell">' + displayRank + '</td>';
            html += '<td>' + escapeHtml(entry.display_name) + '</td>';
            html += '<td class="score-cell">' + entry.score + '</td>';
            html += '<td>' + gradeBadge(entry.grade) + '</td>';
            html += '<td class="date-cell">' + formatDateWithAge(entry.last_updated || entry.date) + '</td>';
            html += mapScoreCell(entry.map_scores, '8x8');
            html += mapScoreCell(entry.map_scores, '16x16');
            for (var k = 0; k < ANCHORS.length; k++) {
                html += opponentCell(entry.opponents, ANCHORS[k]);
            }
            row.innerHTML = html;
            (function (idx) {
                row.addEventListener('click', function () {
                    toggleDetail('leaderboard-table', idx, filteredRef);
                });
            })(j);
            tbody.appendChild(row);
        }

        updateCount('leaderboard-count', entries.length, leaderboardEntries.length);
    }

    function renderBenchmarks() {
        var tbody = document.querySelector('#benchmarks-table tbody');
        if (!tbody || !dataLoaded) return;
        tbody.innerHTML = '';
        expandedRows['benchmarks-table'] = null;

        var filters = getFilterValues('benchmarks');
        var entries = [];
        for (var i = 0; i < benchmarkEntries.length; i++) {
            var e = benchmarkEntries[i];
            if (!matchesSearch(e, filters.search, ['name', 'grade', 'source'])) continue;
            if (!matchesDateRange(e._sortData.date, filters.dateFrom, filters.dateTo)) continue;
            entries.push(e);
        }

        var state = sortState['benchmarks-table'];
        if (state) {
            entries = sortEntries(entries, state.key, state.type, state.dir);
        }

        var filteredRef = entries;
        for (var j = 0; j < entries.length; j++) {
            var entry = entries[j].raw;
            var row = document.createElement('tr');
            row.setAttribute('data-idx', j);
            var src = entry.source || 'unknown';
            row.innerHTML =
                '<td>' + escapeHtml(entry.display_name) + '</td>' +
                '<td class="score-cell">' + entry.score + '</td>' +
                '<td>' + gradeBadge(entry.grade) + '</td>' +
                '<td><span class="source-badge">' + escapeHtml(src) + '</span></td>' +
                '<td>' + formatDate(entry.date) + '</td>';
            (function (idx) {
                row.addEventListener('click', function () {
                    toggleDetail('benchmarks-table', idx, filteredRef);
                });
            })(j);
            tbody.appendChild(row);
        }

        updateCount('benchmarks-count', entries.length, benchmarkEntries.length);
    }

    function renderH2H(data) {
        var h2h = data.head_to_head || {};
        var players = [];
        for (var p in h2h) {
            if (h2h.hasOwnProperty(p)) players.push(p);
        }
        players.sort();

        if (players.length === 0) {
            document.getElementById('h2h-empty').style.display = 'block';
            var container = document.querySelector('#h2h .table-container');
            if (container) container.style.display = 'none';
            return;
        }

        var thead = document.querySelector('#h2h-table thead');
        var tbody = document.querySelector('#h2h-table tbody');

        var headerHtml = '<tr><th></th>';
        for (var hi = 0; hi < players.length; hi++) {
            headerHtml += '<th>' + escapeHtml(players[hi].split(' (')[0]) + '</th>';
        }
        headerHtml += '</tr>';
        thead.innerHTML = headerHtml;

        tbody.innerHTML = '';
        for (var ri = 0; ri < players.length; ri++) {
            var p1 = players[ri];
            var row = document.createElement('tr');
            var html = '<td>' + escapeHtml(p1) + '</td>';
            for (var ci = 0; ci < players.length; ci++) {
                var p2 = players[ci];
                if (p1 === p2) {
                    html += '<td class="h2h-self">-</td>';
                } else {
                    var record = h2h[p1] && h2h[p1][p2];
                    if (record) {
                        var cls = 'result-skip';
                        if (record.wins > record.losses) cls = 'result-win';
                        else if (record.losses > record.wins) cls = 'result-loss';
                        else if (record.draws > 0) cls = 'result-draw';
                        html += '<td class="' + cls + '">' + record.wins + 'W/' + record.draws + 'D/' + record.losses + 'L</td>';
                    } else {
                        html += '<td class="result-skip">--</td>';
                    }
                }
            }
            row.innerHTML = html;
            tbody.appendChild(row);
        }
    }

    function renderHistory() {
        var tbody = document.querySelector('#history-table tbody');
        if (!tbody || !dataLoaded) return;
        tbody.innerHTML = '';
        expandedRows['history-table'] = null;

        var filters = getFilterValues('history');
        var entries = [];
        for (var i = 0; i < historyEntries.length; i++) {
            var e = historyEntries[i];
            if (!matchesSearch(e, filters.search, ['name', 'grade', 'source'])) continue;
            if (!matchesDateRange(e._sortData.date, filters.dateFrom, filters.dateTo)) continue;
            entries.push(e);
        }

        var state = sortState['history-table'];
        if (state) {
            entries = sortEntries(entries, state.key, state.type, state.dir);
        }

        var filteredRef = entries;
        for (var j = 0; j < entries.length; j++) {
            var entry = entries[j].raw;
            var row = document.createElement('tr');
            row.setAttribute('data-idx', j);
            var src = entry.source || 'unknown';
            row.innerHTML =
                '<td>' + formatDate(entry.date) + '</td>' +
                '<td>' + escapeHtml(entry.display_name) + '</td>' +
                '<td class="score-cell">' + entry.score + '</td>' +
                '<td>' + gradeBadge(entry.grade) + '</td>' +
                '<td><span class="source-badge">' + escapeHtml(src) + '</span></td>' +
                mapScoreCell(entry.map_scores, '8x8') +
                mapScoreCell(entry.map_scores, '16x16');
            (function (idx) {
                row.addEventListener('click', function () {
                    toggleDetail('history-table', idx, filteredRef);
                });
            })(j);
            tbody.appendChild(row);
        }

        updateCount('history-count', entries.length, historyEntries.length);
    }

    // --- Event binding (runs immediately, before data loads) ---

    // Tab switching
    var tabs = document.querySelectorAll('.tab');
    for (var ti = 0; ti < tabs.length; ti++) {
        tabs[ti].addEventListener('click', function (e) {
            e.preventDefault();
            var target = this.getAttribute('data-tab');
            if (!target) return;
            var allTabs = document.querySelectorAll('.tab');
            for (var i = 0; i < allTabs.length; i++) allTabs[i].classList.remove('active');
            this.classList.add('active');
            var contents = document.querySelectorAll('.tab-content');
            for (var i = 0; i < contents.length; i++) contents[i].classList.remove('active');
            var el = document.getElementById(target);
            if (el) el.classList.add('active');
        });
    }

    // Sort handlers - attach to all th[data-sort] in the page
    function handleSort(e) {
        var th = e.currentTarget;
        var table = th.closest('table');
        if (!table) return;
        var tableId = table.id;
        var key = th.getAttribute('data-sort');
        var type = th.getAttribute('data-type') || 'string';
        var prev = sortState[tableId] || {};
        var dir = (prev.key === key && prev.dir === 'desc') ? 'asc' : 'desc';
        sortState[tableId] = { key: key, type: type, dir: dir };
        updateSortIndicators(tableId, key, dir);

        if (tableId === 'leaderboard-table') renderLeaderboard();
        else if (tableId === 'benchmarks-table') renderBenchmarks();
        else if (tableId === 'history-table') renderHistory();
    }

    var sortHeaders = document.querySelectorAll('th[data-sort]');
    for (var si = 0; si < sortHeaders.length; si++) {
        sortHeaders[si].addEventListener('click', handleSort);
    }

    // Filter handlers - search inputs and date inputs + filter buttons
    function setupFilter(prefix, renderFn) {
        var searchEl = document.getElementById(prefix + '-search');
        var fromEl = document.getElementById(prefix + '-date-from');
        var toEl = document.getElementById(prefix + '-date-to');
        var clearBtn = document.getElementById(prefix + '-date-clear');
        var filterBtn = document.getElementById(prefix + '-filter-btn');

        // Live filtering on every keystroke/change
        if (searchEl) {
            searchEl.addEventListener('input', renderFn);
            searchEl.addEventListener('change', renderFn);
            searchEl.addEventListener('keyup', renderFn);
        }
        if (fromEl) {
            fromEl.addEventListener('input', renderFn);
            fromEl.addEventListener('change', renderFn);
        }
        if (toEl) {
            toEl.addEventListener('input', renderFn);
            toEl.addEventListener('change', renderFn);
        }
        if (clearBtn) {
            clearBtn.addEventListener('click', function () {
                if (fromEl) fromEl.value = '';
                if (toEl) toEl.value = '';
                renderFn();
            });
        }
        if (filterBtn) {
            filterBtn.addEventListener('click', renderFn);
        }
    }

    setupFilter('leaderboard', renderLeaderboard);
    setupFilter('benchmarks', renderBenchmarks);
    setupFilter('history', renderHistory);

    // --- Data loading ---

    fetch(DATA_URL)
        .then(function (response) {
            if (!response.ok) throw new Error('Failed to load data: ' + response.status);
            return response.json();
        })
        .then(function (data) {
            // Prepare leaderboard
            var lb = data.leaderboard || [];
            for (var i = 0; i < lb.length; i++) {
                leaderboardEntries.push({
                    raw: lb[i],
                    _sortData: {
                        rank: i + 1,
                        name: lb[i].display_name || '',
                        score: lb[i].score,
                        grade: lb[i].grade || 'F',
                        date: lb[i].last_updated || lb[i].date || '',
                        score_8x8: getMapScore(lb[i].map_scores, '8x8'),
                        score_16x16: getMapScore(lb[i].map_scores, '16x16')
                    }
                });
            }

            // Prepare benchmarks (best per name)
            var history = data.history || [];
            var seen = {};
            for (var j = 0; j < history.length; j++) {
                var key = history[j].display_name;
                if (!seen[key] || history[j].score > seen[key].score) {
                    seen[key] = history[j];
                }
            }
            var best = objValues(seen);
            best.sort(function (a, b) { return b.score - a.score; });
            for (var bi = 0; bi < best.length; bi++) {
                benchmarkEntries.push({
                    raw: best[bi],
                    _sortData: {
                        name: best[bi].display_name || '',
                        score: best[bi].score,
                        grade: best[bi].grade || 'F',
                        source: best[bi].source || 'unknown',
                        date: best[bi].date || ''
                    }
                });
            }

            // Prepare history
            for (var hi = 0; hi < history.length; hi++) {
                historyEntries.push({
                    raw: history[hi],
                    _sortData: {
                        date: history[hi].date || '',
                        name: history[hi].display_name || '',
                        score: history[hi].score,
                        grade: history[hi].grade || 'F',
                        source: history[hi].source || 'unknown',
                        score_8x8: getMapScore(history[hi].map_scores, '8x8'),
                        score_16x16: getMapScore(history[hi].map_scores, '16x16')
                    }
                });
            }

            dataLoaded = true;

            // Render all
            renderLeaderboard();
            renderBenchmarks();
            renderH2H(data);
            renderHistory();

            var genEl = document.getElementById('generated-at');
            if (genEl && data.generated) {
                genEl.textContent = 'Data updated: ' + formatDate(data.generated);
            }
        })
        .catch(function (err) {
            console.error('Error loading tournament data:', err);
            var main = document.querySelector('main');
            if (main) {
                main.innerHTML = '<p class="empty-msg">Failed to load tournament data. ' +
                    'If viewing locally, serve with a local HTTP server (e.g. <code>python3 -m http.server</code> in the docs/ directory).</p>';
            }
        });
})();
