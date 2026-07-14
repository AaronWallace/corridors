const $ = selector => document.querySelector(selector);
const api = async (path, options = {}) => {
  const response = await fetch(path, {headers: {'Content-Type': 'application/json'}, ...options});
  const body = await response.json();
  if (!response.ok) throw Error(body.error || 'Request failed');
  return body;
};

let catalog = [];
let game = null;
let analysis = null;
let selectedLayer = 'stem';
let selectedChannel = 0;
let selectedPlane = 0;
let busy = false;
let playing = false;
let playToken = 0;
let gameNumber = 1;

const fmt = value => Number(value).toLocaleString();
const compact = value => value >= 1e6 ? `${(value / 1e6).toFixed(2)}M` : value >= 1e3 ? `${(value / 1e3).toFixed(1)}K` : String(value);
const delay = seconds => seconds <= 0 ? Promise.resolve() : new Promise(resolve => setTimeout(resolve, seconds * 1000));
const checkpoint = () => $('#modelSelect').value;

function modelSpec() { return {kind: 'model', checkpoint: checkpoint()}; }

async function newPosition() {
  stopAutoplay();
  const payload = {
    mode: 'ai-ai', p1: modelSpec(), p2: modelSpec(), maxPlies: 160,
    p1Col: Math.floor(Math.random() * 9), p2Col: Math.floor(Math.random() * 9),
  };
  game = await api('/api/games', {method: 'POST', body: JSON.stringify(payload)});
  gameNumber += game ? 1 : 0;
  selectedChannel = 0;
  await analyze();
}

async function analyze() {
  if (!game || busy) return;
  busy = true;
  $('#loading').classList.add('visible');
  try {
    analysis = await api('/api/explorer/analyze', {
      method: 'POST',
      body: JSON.stringify({
        checkpoint: checkpoint(), gameId: game.id, layer: selectedLayer,
        channel: selectedChannel, weightOut: +$('#weightOut').value || 0,
        weightIn: +$('#weightIn').value || 0,
      }),
    });
    game = analysis.game;
    if (analysis.selectedActivation) {
      selectedLayer = analysis.selectedActivation.layer;
      selectedChannel = analysis.selectedActivation.channel;
    }
    render();
  } catch (error) {
    $('#modelFacts').textContent = error.message;
    stopAutoplay();
  } finally {
    busy = false;
    $('#loading').classList.remove('visible');
  }
}

async function stepModel() {
  if (busy) return;
  if (!game || game.gameOver) {
    await newPosition();
    return;
  }
  busy = true;
  $('#loading').classList.add('visible');
  try {
    game = await api(`/api/games/${game.id}/ai-move`, {method: 'POST', body: '{}'});
  } catch (error) {
    $('#modelFacts').textContent = error.message;
    stopAutoplay();
  } finally {
    busy = false;
    $('#loading').classList.remove('visible');
  }
  await analyze();
}

async function autoplayLoop(token) {
  while (playing && token === playToken) {
    if (game && game.gameOver) {
      await delay(0.8);
      if (!playing || token !== playToken) return;
      const payload = {
        mode: 'ai-ai', p1: modelSpec(), p2: modelSpec(), maxPlies: 160,
        p1Col: Math.floor(Math.random() * 9), p2Col: Math.floor(Math.random() * 9),
      };
      game = await api('/api/games', {method: 'POST', body: JSON.stringify(payload)});
      gameNumber += 1;
      await analyze();
      continue;
    }
    await delay(+$(`#playDelay`).value);
    if (!playing || token !== playToken) return;
    await stepModel();
  }
}

function stopAutoplay() {
  playing = false;
  playToken += 1;
  $('#autoplay').classList.remove('active');
  $('#autoplay').textContent = 'Autoplay';
}

function toggleAutoplay() {
  if (playing) return stopAutoplay();
  playing = true;
  const token = ++playToken;
  $('#autoplay').classList.add('active');
  $('#autoplay').textContent = 'Pause';
  autoplayLoop(token);
}

function renderBoard() {
  const board = $('#miniBoard');
  board.innerHTML = '';
  for (let row = 0; row < 11; row++) for (let col = 0; col < 9; col++) {
    const cell = document.createElement('div');
    cell.className = `mini-cell${row === 0 || row === 10 ? ' end' : ''}`;
    if (Object.values(game.goals).some(goal => goal[0] === row && goal[1] === col)) cell.classList.add('goal');
    board.appendChild(cell);
  }
  for (const player of ['1', '2']) {
    const [row, col] = game.pawns[player];
    const pawn = document.createElement('div');
    pawn.className = `mini-pawn p${player}`;
    pawn.style.left = `${(col + .5) / 9 * 100}%`;
    pawn.style.top = `${(row + .5) / 11 * 100}%`;
    board.appendChild(pawn);
  }
  for (const [row, col, orientation] of game.walls) {
    const wall = document.createElement('div');
    wall.className = `mini-wall ${orientation.toLowerCase()}`;
    wall.style.left = `${(col + 1) / 9 * 100}%`;
    wall.style.top = `${(row + 1) / 11 * 100}%`;
    board.appendChild(wall);
  }
  $('#turnBadge').textContent = game.gameOver ? (game.winner ? `P${game.winner} wins` : 'Draw') : `P${game.turn} to move`;
  $('#gameState').textContent = `Game ${gameNumber} · turn ${game.plies}`;
  $('#wallCount').textContent = `${game.wallsLeft['1']} / ${game.wallsLeft['2']} walls`;
}

const planeDescriptions = [
  'One-hot location of Player 1’s pawn.', 'One-hot location of Player 2’s pawn.',
  'Anchors of every horizontal wall.', 'Anchors of every vertical wall.',
  'Player 1 wall inventory, broadcast across the board.', 'Player 2 wall inventory, broadcast across the board.',
  'All ones when Player 1 moves; all zeroes when Player 2 moves.',
  'One-hot goal cell Player 1 must reach.', 'One-hot goal cell Player 2 must reach.',
];

function color(value, signed = false) {
  const amount = Math.min(1, Math.abs(value));
  if (signed && value < 0) return `rgba(119,91,198,${.15 + amount * .85})`;
  return `rgba(67,221,205,${.08 + amount * .92})`;
}

function drawGrid(canvas, values, signed = false) {
  const ctx = canvas.getContext('2d');
  const rows = values.length, cols = values[0].length;
  const w = canvas.width / cols, h = canvas.height / rows;
  const max = Math.max(1e-9, ...values.flat().map(Math.abs));
  ctx.fillStyle = '#070d10'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  values.forEach((line, row) => line.forEach((raw, col) => {
    const value = raw / max;
    ctx.fillStyle = color(value, signed);
    ctx.fillRect(col * w + 1, row * h + 1, w - 2, h - 2);
    if (Math.abs(value) > .2) {
      ctx.fillStyle = '#e8f4f3b8'; ctx.font = `${Math.max(7, Math.min(11, w / 3))}px sans-serif`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(raw.toFixed(2), (col + .5) * w, (row + .5) * h);
    }
  }));
}

function renderInputs() {
  $('#planeTabs').innerHTML = analysis.inputPlanes.map((name, index) =>
    `<button class="${index === selectedPlane ? 'active' : ''}" data-plane="${index}" title="${name}">${index} · ${name}</button>`).join('');
  $('#planeTabs').querySelectorAll('button').forEach(button => button.onclick = () => {
    selectedPlane = +button.dataset.plane;
    renderInputs();
  });
  drawGrid($('#inputMap'), analysis.input[selectedPlane]);
  $('#planeDescription').textContent = planeDescriptions[selectedPlane];
}

function renderArchitecture() {
  const spatial = analysis.layers.filter(layer => layer.map);
  const maxEnergy = Math.max(1e-9, ...analysis.layers.map(layer => layer.stats.mean));
  $('#architecture').innerHTML = analysis.layers.map(layer => {
    const selected = layer.name === selectedLayer ? ' selected' : '';
    const branch = layer.name.includes('policy') || layer.name.includes('value') ? ' branch' : '';
    const width = Math.min(100, Math.abs(layer.stats.mean) / maxEnergy * 100);
    return `<button class="layer-node${selected}${branch}" data-layer="${layer.name}"><strong>${layer.label}</strong><span>${layer.shape.join(' × ')}</span><span>${compact(layer.parameters)} params</span><i><b style="width:${width}%"></b></i></button>`;
  }).join('');
  $('#architecture').querySelectorAll('button').forEach(button => button.onclick = async () => {
    selectedLayer = button.dataset.layer;
    selectedChannel = 0;
    await analyze();
  });
  const maxSpatial = Math.max(1e-9, ...spatial.map(layer => Math.abs(layer.stats.mean)));
  $('#signalPath').innerHTML = spatial.map(layer => `<div class="signal-step" style="--energy:${Math.max(3, Math.abs(layer.stats.mean) / maxSpatial * 100)}%"><i></i><span>${layer.label}</span></div>`).join('');
}

function statCards(stats) {
  return [['mean', stats.mean], ['std dev', stats.std], ['minimum', stats.min], ['maximum', stats.max], ['zero', stats.zeroFraction]].map(([label, value]) =>
    `<div><span>${label}</span><strong>${label === 'zero' ? (value * 100).toFixed(1) + '%' : value.toFixed(4)}</strong></div>`).join('');
}

function renderActivation() {
  const selected = analysis.selectedActivation;
  if (!selected) {
    $('#activationTitle').textContent = 'No spatial activation';
    $('#activationShape').textContent = 'dense output';
    const canvas = $('#activationMap');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    $('#activationStats').innerHTML = '<div><span>Layer type</span><strong>Dense</strong></div>';
    $('#topChannels').innerHTML = '<div class="empty">Choose a spatial layer to inspect individual feature maps.</div>';
    return;
  }
  const layer = analysis.layers.find(item => item.name === selected.layer);
  $('#activationTitle').textContent = layer.label;
  $('#activationShape').textContent = layer.shape.join(' × ');
  $('#channel').max = selected.channels - 1;
  $('#channel').value = selected.channel;
  $('#channelValue').textContent = `${selected.channel} / ${selected.channels - 1}`;
  drawGrid($('#activationMap'), selected.map, true);
  $('#activationStats').innerHTML = statCards(selected.stats);
  const max = Math.max(...selected.topChannels.map(item => item.energy), 1e-9);
  $('#topChannels').innerHTML = selected.topChannels.map(item => `<button data-channel="${item.channel}"><span>CH ${item.channel}</span><i><b style="width:${item.energy / max * 100}%"></b></i><span>${item.energy.toFixed(3)}</span></button>`).join('');
  $('#topChannels').querySelectorAll('button').forEach(button => button.onclick = async () => {
    selectedChannel = +button.dataset.channel;
    await analyze();
  });
}

function moveLabel(move) {
  const [row, col, orientation] = move.at;
  const square = `${String.fromCharCode(65 + col)}${row}`;
  return move.kind === 'm' ? `Pawn → ${square}` : `Wall ${square}${orientation}`;
}

function renderDecision() {
  const value = analysis.value;
  $('#valueScore').textContent = `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
  $('#valuePerspective').textContent = analysis.valuePerspective;
  $('#valueMarker').style.left = `${(value + 1) / 2 * 100}%`;
  $('#policyMoves').innerHTML = analysis.policy.length ? analysis.policy.map((item, index) =>
    `<div class="policy-move"><span>${index + 1}</span><strong>${moveLabel(item.move)}</strong><span class="policy-bar"><b style="width:${item.probability * 100}%"></b></span><span>${(item.probability * 100).toFixed(1)}%</span></div>`).join('') : '<div class="empty">Value-only network — it evaluates child positions instead of producing policy logits.</div>';
}

function drawHistogram() {
  const canvas = $('#weightHistogram'), ctx = canvas.getContext('2d');
  const histogram = analysis.weights.histogram;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!histogram.length) return;
  const max = Math.max(...histogram.map(bin => bin.count));
  const bar = canvas.width / histogram.length;
  histogram.forEach((bin, index) => {
    const height = bin.count / max * (canvas.height - 24);
    const midpoint = (bin.from + bin.to) / 2;
    ctx.fillStyle = midpoint < 0 ? '#765fc4' : '#43d9cd';
    ctx.fillRect(index * bar + 1, canvas.height - height - 14, Math.max(1, bar - 2), height);
  });
  ctx.strokeStyle = '#465b64'; ctx.beginPath(); ctx.moveTo(0, canvas.height - 13); ctx.lineTo(canvas.width, canvas.height - 13); ctx.stroke();
}

function renderWeights() {
  const weights = analysis.weights;
  $('#weightShape').textContent = weights.shape.length ? weights.shape.join(' × ') : 'no weights';
  drawHistogram();
  $('#weightStats').innerHTML = weights.stats ? statCards(weights.stats) : '';
  const kernel = weights.kernel;
  $('#kernel').innerHTML = '';
  if (!kernel) { $('#kernelLabel').textContent = 'This stage has no learned tensor.'; return; }
  const values = kernel.values;
  const flat = values.flat();
  const max = Math.max(1e-9, ...flat.map(Math.abs));
  $('#kernel').style.gridTemplateColumns = `repeat(${values[0].length},1fr)`;
  $('#kernelLabel').textContent = `${weights.tensor} · output ${kernel.out}${kernel.in == null ? '' : ` · input ${kernel.in}`}`;
  $('#kernel').innerHTML = flat.map(value => `<span style="background:${color(value / max, true)}" title="${value.toFixed(6)}">${value.toFixed(2)}</span>`).join('');
}

function renderFacts() {
  const model = catalog.find(item => item.name === checkpoint()) || {};
  $('#modelFacts').innerHTML = `<strong>${analysis.architecture === 'az' ? 'AlphaZero policy + value' : 'Value network'}</strong><br>${fmt(analysis.parameters)} parameters · ${model.blocks || analysis.layers.filter(l => l.name.startsWith('block_')).length} residual blocks · modified ${model.modified}${model.elo != null ? ` · Elo ${Math.round(model.elo)}` : ''}`;
}

function render() {
  renderFacts(); renderBoard(); renderInputs(); renderArchitecture();
  renderActivation(); renderDecision(); renderWeights();
}

$('#newPosition').onclick = newPosition;
$('#stepModel').onclick = stepModel;
$('#autoplay').onclick = toggleAutoplay;
$('#modelSelect').onchange = newPosition;
$('#channel').oninput = event => { selectedChannel = +event.target.value; $('#channelValue').textContent = selectedChannel; };
$('#channel').onchange = analyze;
$('#weightOut').onchange = analyze;
$('#weightIn').onchange = analyze;
$('#playDelay').oninput = event => { $('#delayValue').textContent = +event.target.value === 0 ? 'No delay' : `${(+event.target.value).toFixed(1)}s`; };

(async () => {
  const config = await api('/api/config');
  catalog = config.models.filter(model => !model.loadError);
  if (!catalog.length) throw Error('No loadable neural checkpoints found.');
  $('#modelSelect').innerHTML = catalog.map(model => `<option value="${model.name}">${model.name} · ${model.modified}${model.elo == null ? '' : ` · Elo ${Math.round(model.elo)}`}</option>`).join('');
  gameNumber = 0;
  await newPosition();
})().catch(error => { $('#modelFacts').textContent = error.message; $('#loading').classList.remove('visible'); });
