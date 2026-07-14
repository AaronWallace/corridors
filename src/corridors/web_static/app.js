const $ = selector => document.querySelector(selector);
const api = async (path, options = {}) => {
  const response = await fetch(path, {
    headers: {'Content-Type': 'application/json'},
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw Error(payload.error || 'Request failed');
  return payload;
};
const fmt = value => value == null ? '—' : Number(value).toLocaleString();
async function waitForLiveDelay(readSeconds, token) {
  const started = performance.now();
  while (autoplayRunning && token === autoplayToken) {
    const remaining = readSeconds() * 1000 - (performance.now() - started);
    if (remaining <= 0) return true;
    await new Promise(resolve => setTimeout(resolve, Math.min(remaining, 50)));
  }
  return false;
}

let config = {models: []};
let game = null;
let mode = 'human-ai';
let wallMode = null;
let pawnSelected = false;
let busy = false;
let lastSetup = null;
let rapidMode = false;
let autoplayRunning = false;
let autoplayToken = 0;
let turnDelay = 0.5;
let endgameDelay = 1.0;
let currentGameRecorded = false;

function emptyAutorunStats() {
  return {
    games: 0,
    p1Wins: 0,
    p2Wins: 0,
    draws: 0,
    totalTurns: 0,
    currentTurns: 0,
    shortest: null,
    longest: null,
    moveCount: 0,
    totalMoveSeconds: 0,
    walls: 0,
    paths: {1: Array(99).fill(0), 2: Array(99).fill(0)},
    wallHeat: new Map(),
    drawReasons: {},
  };
}
let autorun = emptyAutorunStats();

function agentEditor(root, player) {
  root.innerHTML = `<legend>Player ${player}</legend>
    <div class="field"><label>Controller</label><select class="kind">
      <option value="classical">Classical solver</option>
      ${config.models.map(model => `<option value="model:${model.name}">${model.name}</option>`).join('')}
    </select></div>
    <div class="classical">
      <div class="field"><label>Time per move</label><input class="time" type="number" value="0.5" min="0" max="600" step="0.1"></div>
      <div class="field"><label>Maximum depth</label><input class="depth" type="number" value="4" min="1" max="8"></div>
    </div><div class="model-note"></div>`;
  const select = root.querySelector('.kind');
  const sync = () => {
    const isModel = select.value.startsWith('model:');
    root.querySelector('.classical').hidden = isModel;
    const name = select.value.slice(6);
    const model = config.models.find(item => item.name === name);
    root.querySelector('.model-note').textContent = model
      ? `${model.architecture.toUpperCase()}${model.elo != null ? ' · Elo ' + model.elo : ''}${model.positions ? ' · ' + fmt(model.positions) + ' positions' : ''}${model.dataset ? ' · ' + model.dataset : ''}${model.loaded ? ' · preloaded' : ''}`
      : 'Iterative deepening with a persistent move cache.';
  };
  select.onchange = sync;
  sync();
}

function spec(root) {
  const value = root.querySelector('.kind').value;
  if (value.startsWith('model:')) return {kind: 'model', checkpoint: value.slice(6)};
  return {
    kind: 'classical',
    timeLimit: +root.querySelector('.time').value,
    depth: +root.querySelector('.depth').value,
  };
}

function syncMode() {
  document.querySelectorAll('.segmented button').forEach(button => {
    button.classList.toggle('active', button.dataset.mode === mode);
  });
  const aiOnly = mode === 'ai-ai' || mode === 'rapid';
  $('#humanOptions').hidden = aiOnly;
  $('#rapidOptions').hidden = mode !== 'rapid';
  $('#p1Setup').hidden = mode === 'human-ai' && $('#humanSide').value === '1';
  $('#p2Setup').hidden = mode === 'human-ai' && $('#humanSide').value === '2';
  $('#startGame').textContent = mode === 'rapid' ? 'Start autoplay' : 'Start match';
}

function randomizedSetup() {
  return {...lastSetup, p1Col: Math.floor(Math.random() * 9), p2Col: Math.floor(Math.random() * 9)};
}

async function launch(body) {
  lastSetup = body;
  game = await api('/api/games', {method: 'POST', body: JSON.stringify(body)});
  wallMode = null;
  pawnSelected = false;
  render();
}

async function start() {
  stopRapidAutoplay();
  const selectedMode = mode;
  const body = {
    mode: selectedMode === 'human-ai' ? 'human-ai' : 'ai-ai',
    p1Col: Math.floor(Math.random() * 9),
    p2Col: Math.floor(Math.random() * 9),
    maxPlies: Math.max(20, +$('#maxPlies').value || 160),
  };
  if (selectedMode === 'human-ai') {
    body.humanSide = $('#humanSide').value;
    body.ai = spec(body.humanSide === '1' ? $('#p2Setup') : $('#p1Setup'));
  } else {
    body.p1 = spec($('#p1Setup'));
    body.p2 = spec($('#p2Setup'));
  }
  rapidMode = selectedMode === 'rapid';
  $('#statsPanel').hidden = !rapidMode;
  if (rapidMode) autorun = emptyAutorunStats();
  await launch(body);
  $('#setup').close();
  if (rapidMode) {
    beginRapidGame();
    resumeRapidAutoplay();
  } else {
    autoMove();
  }
}

async function rematch() {
  if (!lastSetup) return;
  await launch(randomizedSetup());
  autoMove();
}

function humanTurn() {
  return game && !game.gameOver && game.players[String(game.turn)].kind === 'human';
}
function pawnPos(player) {
  const [row, col] = game.pawns[player];
  return {left: (col + 0.5) / 9 * 100, top: (row + 0.5) / 11 * 100};
}
function moveKey(move) { return `${move.kind}:${move.at.join(',')}`; }

async function play(move) {
  if (busy) return;
  busy = true;
  try {
    game = await api(`/api/games/${game.id}/move`, {
      method: 'POST', body: JSON.stringify(move),
    });
    wallMode = null;
    pawnSelected = false;
    render();
  } catch (error) {
    showStatus(error.message, true);
  } finally {
    busy = false;
  }
  await autoMove();
}

async function requestAiMove(token = null) {
  if (!game || game.gameOver || busy) return false;
  const gameId = game.id;
  busy = true;
  const label = agentLabel(game.players[String(game.turn)]);
  const started = performance.now();
  const tick = () => showStatus(`${label} is thinking… ${((performance.now() - started) / 1000).toFixed(1)}s`);
  tick();
  const timer = setInterval(tick, 100);
  try {
    const next = await api(`/api/games/${gameId}/ai-move`, {method: 'POST', body: '{}'});
    if (token != null && (token !== autoplayToken || !autoplayRunning || game.id !== gameId)) return false;
    if (rapidMode) recordRapidMove(next);
    game = next;
    render();
    return true;
  } catch (error) {
    showStatus(error.message, true);
    if (rapidMode) stopRapidAutoplay();
    return false;
  } finally {
    clearInterval(timer);
    busy = false;
  }
}

async function autoMove() {
  if (rapidMode || !game || game.gameOver || humanTurn() || busy) return;
  if (await requestAiMove() && game && !game.gameOver && !humanTurn()) {
    setTimeout(autoMove, 260);
  }
}

function beginRapidGame() {
  currentGameRecorded = false;
  autorun.currentTurns = 0;
  for (const player of ['1', '2']) {
    const [row, col] = game.pawns[player];
    autorun.paths[player][row * 9 + col] += 1;
  }
  renderStats();
}

function recordRapidMove(next) {
  const move = next.history[next.history.length - 1];
  if (!move) return;
  autorun.currentTurns = next.plies;
  autorun.moveCount += 1;
  autorun.totalMoveSeconds += Number(move.elapsed || 0);
  if (move.kind === 'm') {
    const [row, col] = move.at;
    autorun.paths[String(move.player)][row * 9 + col] += 1;
  } else {
    autorun.walls += 1;
    const key = move.at.join(',');
    autorun.wallHeat.set(key, (autorun.wallHeat.get(key) || 0) + 1);
  }
  renderStats();
}

function finishRapidGame() {
  if (!game || !game.gameOver || currentGameRecorded) return;
  currentGameRecorded = true;
  const turns = game.plies;
  autorun.games += 1;
  autorun.totalTurns += turns;
  autorun.shortest = autorun.shortest == null ? turns : Math.min(autorun.shortest, turns);
  autorun.longest = autorun.longest == null ? turns : Math.max(autorun.longest, turns);
  if (game.winner === 1) autorun.p1Wins += 1;
  else if (game.winner === 2) autorun.p2Wins += 1;
  else {
    autorun.draws += 1;
    const reason = game.drawReason || 'draw';
    autorun.drawReasons[reason] = (autorun.drawReasons[reason] || 0) + 1;
  }
  renderStats();
}

async function rapidLoop(token) {
  while (autoplayRunning && token === autoplayToken && rapidMode) {
    if (game.gameOver) {
      finishRapidGame();
      if (!await waitForLiveDelay(() => endgameDelay, token)) return;
      await launch(randomizedSetup());
      if (!autoplayRunning || token !== autoplayToken) return;
      beginRapidGame();
      continue;
    }
    if (!await waitForLiveDelay(() => turnDelay, token)) return;
    if (!await requestAiMove(token)) return;
  }
}

function resumeRapidAutoplay() {
  if (!rapidMode || autoplayRunning) return;
  autoplayRunning = true;
  const token = ++autoplayToken;
  renderStats();
  rapidLoop(token);
}
function stopRapidAutoplay() {
  autoplayRunning = false;
  autoplayToken += 1;
  renderStats();
}
async function resetRapidStats() {
  const wasRunning = autoplayRunning;
  stopRapidAutoplay();
  autorun = emptyAutorunStats();
  if (rapidMode && lastSetup) {
    await launch(randomizedSetup());
    beginRapidGame();
    if (wasRunning) resumeRapidAutoplay();
  } else {
    renderStats();
  }
}

function formatDelay(value) {
  return value === 0 ? 'No delay' : `${value.toFixed(1)}s`;
}
function renderStats() {
  if (!$('#statsPanel')) return;
  $('#runState').textContent = autoplayRunning ? 'Running' : 'Paused';
  $('#runState').classList.toggle('paused', !autoplayRunning);
  $('#toggleAutoplay').textContent = autoplayRunning ? 'Pause' : 'Resume';
  $('#turnDelayValue').textContent = formatDelay(turnDelay);
  $('#endgameDelayValue').textContent = formatDelay(endgameDelay);
  $('#statGames').textContent = autorun.games;
  $('#statCurrentTurns').textContent = autorun.currentTurns;
  $('#statP1Wins').textContent = autorun.p1Wins;
  $('#statP2Wins').textContent = autorun.p2Wins;
  $('#statDraws').textContent = autorun.draws;
  $('#statAvgTurns').textContent = autorun.games ? (autorun.totalTurns / autorun.games).toFixed(1) : '—';
  $('#statShortest').textContent = autorun.shortest ?? '—';
  $('#statLongest').textContent = autorun.longest ?? '—';
  $('#statAvgMove').textContent = autorun.moveCount ? `${(autorun.totalMoveSeconds * 1000 / autorun.moveCount).toFixed(0)}ms` : '—';
  $('#statWalls').textContent = autorun.games ? (autorun.walls / autorun.games).toFixed(1) : '—';
  drawHeatmap();
}

function drawHeatmap() {
  const canvas = $('#heatmap');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const margin = 15;
  const cellW = (width - margin * 2) / 9;
  const cellH = (height - margin * 2) / 11;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#24180f';
  ctx.fillRect(0, 0, width, height);
  for (let row = 0; row < 11; row++) {
    for (let col = 0; col < 9; col++) {
      ctx.fillStyle = row === 0 || row === 10 ? '#473522' : '#77502f';
      ctx.fillRect(margin + col * cellW + 1.5, margin + row * cellH + 1.5, cellW - 3, cellH - 3);
    }
  }
  const pathMax = Math.max(1, ...autorun.paths[1], ...autorun.paths[2]);
  for (let row = 0; row < 11; row++) {
    for (let col = 0; col < 9; col++) {
      const index = row * 9 + col;
      const centerX = margin + (col + 0.5) * cellW;
      const centerY = margin + (row + 0.5) * cellH;
      for (const [player, color, offset] of [[1, '231,169,59', -4], [2, '168,137,235', 4]]) {
        const count = autorun.paths[player][index];
        if (!count) continue;
        ctx.beginPath();
        ctx.arc(centerX + offset, centerY, 3 + 7 * Math.sqrt(count / pathMax), 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${color},${0.22 + 0.7 * count / pathMax})`;
        ctx.fill();
      }
    }
  }
  const wallMax = Math.max(1, ...autorun.wallHeat.values());
  ctx.lineCap = 'round';
  for (const [key, count] of autorun.wallHeat) {
    const [row, col, orientation] = key.split(',');
    const x = margin + (+col + 1) * cellW;
    const y = margin + (+row + 1) * cellH;
    ctx.strokeStyle = `rgba(241,222,180,${0.18 + 0.82 * count / wallMax})`;
    ctx.lineWidth = 2 + 4 * count / wallMax;
    ctx.beginPath();
    if (orientation === 'H') {
      ctx.moveTo(x - cellW, y);
      ctx.lineTo(x + cellW, y);
    } else {
      ctx.moveTo(x, y - cellH);
      ctx.lineTo(x, y + cellH);
    }
    ctx.stroke();
  }
}

function agentLabel(agent) {
  return agent.kind === 'human' ? 'Human' : agent.kind === 'model'
    ? agent.checkpoint : `Classical · ${agent.timeLimit}s`;
}
function showStatus(text, error = false) {
  $('#status').textContent = text;
  $('#status').style.color = error ? '#ef7f78' : '';
}
function renderPlayer(player) {
  const agent = game.players[String(player)];
  const panel = $(`#p${player}Panel`);
  panel.classList.toggle('active', game.turn === player && !game.gameOver);
  const left = game.wallsLeft[String(player)];
  panel.innerHTML = `<span class="eyebrow">PLAYER ${player}</span>
    <h3>${agent.kind === 'human' ? 'Human' : agent.kind === 'model' ? 'Neural model' : 'Classical AI'}</h3>
    <div class="agent-name" title="${agentLabel(agent)}">${agentLabel(agent)}</div>
    <div class="wall-rack">${Array.from({length: 9}, (_, index) => `<button class="rack-wall ${index >= left ? 'used' : ''}" ${agent.kind !== 'human' || index >= left ? 'disabled' : ''} aria-label="Pick up wall"></button>`).join('')}</div>
    <div class="agent-name" style="margin-top:9px">${left} walls remaining</div>`;
  panel.querySelectorAll('.rack-wall:not(.used)').forEach(button => button.onclick = () => {
    if (humanTurn() && game.turn === player) { wallMode = wallMode || 'H'; pawnSelected = false; render(); }
  });
}

function render() {
  if (!game) return;
  renderPlayer(1);
  renderPlayer(2);
  const board = $('#board');
  board.innerHTML = '';
  const legalMoves = new Set(game.legal.filter(move => move.kind === 'm').map(moveKey));
  for (let row = 0; row < 11; row++) for (let col = 0; col < 9; col++) {
    const cell = document.createElement('button');
    cell.className = 'cell' + (row === 0 || row === 10 ? ' endzone' : '');
    if (Object.values(game.goals).some(goal => goal[0] === row && goal[1] === col)) cell.classList.add('goal');
    const key = `m:${row},${col}`;
    if (humanTurn() && pawnSelected && legalMoves.has(key)) {
      cell.classList.add('legal');
      cell.onclick = () => play({kind: 'm', at: [row, col]});
    }
    cell.ondragover = event => { if (humanTurn() && legalMoves.has(key)) event.preventDefault(); };
    cell.ondrop = event => { event.preventDefault(); if (humanTurn() && legalMoves.has(key)) play({kind: 'm', at: [row, col]}); };
    board.appendChild(cell);
  }
  for (const player of ['1', '2']) {
    const position = pawnPos(player);
    const element = document.createElement('div');
    element.className = `pawn p${player}` + (pawnSelected && +player === game.turn ? ' selected' : '');
    element.style.left = position.left + '%';
    element.style.top = position.top + '%';
    element.draggable = humanTurn() && +player === game.turn;
    element.onclick = () => { if (humanTurn() && +player === game.turn) { pawnSelected = !pawnSelected; wallMode = null; render(); } };
    element.ondragstart = () => { pawnSelected = true; };
    board.appendChild(element);
  }
  for (const [row, col, orientation] of game.walls) {
    const wall = document.createElement('div');
    wall.className = `wall ${orientation.toLowerCase()}`;
    wall.style.left = (col + 1) / 9 * 100 + '%';
    wall.style.top = (row + 1) / 11 * 100 + '%';
    board.appendChild(wall);
  }
  if (wallMode && humanTurn()) {
    const legal = new Set(game.legal.filter(move => move.kind === 'w').map(moveKey));
    for (let row = 1; row <= 8; row++) for (let col = 0; col <= 7; col++) {
      const key = `w:${row},${col},${wallMode}`;
      const slot = document.createElement('button');
      slot.className = `wall-slot ${wallMode.toLowerCase()}`;
      slot.style.left = (col + 1) / 9 * 100 + '%';
      slot.style.top = (row + 1) / 11 * 100 + '%';
      slot.disabled = !legal.has(key);
      if (legal.has(key)) slot.onclick = () => play({kind: 'w', at: [row, col, wallMode]});
      board.appendChild(slot);
    }
  }
  $('#playAgain').hidden = rapidMode || !game.gameOver;
  if (game.winner) showStatus(`Player ${game.winner} wins`);
  else if (game.drawReason) showStatus(`Draw · ${game.drawReason}`);
  else if (humanTurn()) showStatus(wallMode ? `Place a ${wallMode === 'H' ? 'horizontal' : 'vertical'} wall` : `Player ${game.turn} · your move`);
  else showStatus(`${agentLabel(game.players[String(game.turn)])} to move`);
  $('#history').innerHTML = game.history.map(move => `<li>P${move.player} · ${move.kind === 'm' ? String.fromCharCode(65 + move.at[1]) + move.at[0] : String.fromCharCode(65 + move.at[1]) + move.at[0] + move.at[2]}</li>`).join('');
  const info = game.lastInfo;
  $('#telemetry').textContent = info ? `${info.solver} · ${(info.elapsed * 1000).toFixed(0)} ms` : 'No moves yet';
  if (rapidMode) renderStats();
}

document.addEventListener('keydown', event => {
  if (event.key.toLowerCase() === 'r' && wallMode) { wallMode = wallMode === 'H' ? 'V' : 'H'; render(); }
  if (event.key === 'Escape' && (wallMode || pawnSelected)) { wallMode = null; pawnSelected = false; render(); }
});
document.querySelectorAll('.segmented button').forEach(button => button.onclick = () => { mode = button.dataset.mode; syncMode(); });
$('#humanSide').onchange = syncMode;
$('#setupForm').onsubmit = event => { event.preventDefault(); start(); };
$('#newGame').onclick = () => { stopRapidAutoplay(); $('#setup').showModal(); };
$('#playAgain').onclick = rematch;
$('#toggleAutoplay').onclick = () => autoplayRunning ? stopRapidAutoplay() : resumeRapidAutoplay();
$('#resetStats').onclick = resetRapidStats;
$('#turnDelay').oninput = event => { turnDelay = +event.target.value; renderStats(); };
$('#endgameDelay').oninput = event => { endgameDelay = +event.target.value; renderStats(); };

(async () => {
  config = await api('/api/config');
  agentEditor($('#p1Setup'), 1);
  agentEditor($('#p2Setup'), 2);
  syncMode();
  renderStats();
  $('#setup').showModal();
})().catch(error => showStatus(error.message, true));
