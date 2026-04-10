cytoscape.use(cytoscapeDagre);

/** 数据集目录名 -> 协议表单值（与后端 _LABELERS / filter 一致） */
const DATASET_NAME_TO_PROTOCOL = {
  MODBUS: 'MODBUS',
  MODBUSTCP: 'MODBUS',
  S7COMM: 'S7COMM',
  S7: 'S7COMM',
  DNP3: 'DNP3',
  IEC104: 'IEC104',
  'IEC60870-104': 'IEC104',
  ETHERNET_IP: 'ETHERNET_IP',
  ETHERNETIP: 'ETHERNET_IP',
  MQTT: 'MQTT',
};

function inferProtocolFromDatasetFolderName(name) {
  const raw = (name || '').trim();
  if (!raw) return null;
  const upper = raw.toUpperCase();
  if (DATASET_NAME_TO_PROTOCOL[upper]) return DATASET_NAME_TO_PROTOCOL[upper];
  const compact = upper.replace(/[-_\s]/g, '');
  if (DATASET_NAME_TO_PROTOCOL[compact]) return DATASET_NAME_TO_PROTOCOL[compact];
  if (compact.includes('S7')) return 'S7COMM';
  if (compact.includes('MODBUS')) return 'MODBUS';
  if (compact.includes('DNP3')) return 'DNP3';
  if (compact.includes('IEC60870') || compact.includes('IEC104')) return 'IEC104';
  if (compact.includes('ETHERNET') && compact.includes('IP')) return 'ETHERNET_IP';
  if (compact.includes('MQTT')) return 'MQTT';
  return null;
}

/** 无 Data 目录时的备选：显示名与常见文件夹名一致，value 为后端协议参数 */
const FALLBACK_PROTOCOL_ROWS = [
  { folderName: 'MODBUS', protocol: 'MODBUS' },
  { folderName: 'S7COMM', protocol: 'S7COMM' },
  { folderName: 'DNP3', protocol: 'DNP3' },
  { folderName: 'IEC60870-104', protocol: 'IEC104' },
  { folderName: 'Ethernet_IP', protocol: 'ETHERNET_IP' },
  { folderName: 'MQTT', protocol: 'MQTT' },
];

function fillProtocolSelect(rows) {
  el.protocolSelect.innerHTML = '';
  rows.forEach((row) => {
    const opt = document.createElement('option');
    opt.value = row.protocol;
    opt.dataset.folderName = row.folderName;
    opt.textContent = row.folderName;
    el.protocolSelect.appendChild(opt);
  });
}

function getRowsFromDatasetItems(items) {
  if (!items || !items.length) return FALLBACK_PROTOCOL_ROWS;
  const seen = new Set();
  const rows = [];
  items.forEach((item) => {
    const proto = item.protocol || inferProtocolFromDatasetFolderName(item.name) || String(item.name).toUpperCase();
    if (!proto || seen.has(proto)) return;
    seen.add(proto);
    rows.push({
      folderName: item.name,
      protocol: proto,
    });
  });
  return rows.length ? rows : FALLBACK_PROTOCOL_ROWS;
}

const state = {
  artifactId: null,
  model: null,
  viewModel: null,
  metrics: null,
  replay: { steps: [], summary: {} },
  currentStep: 0,
  timer: null,
  cy: null,
  view: { level: 'full', percentile: 70 },
};

const el = {
  datasetSelect: document.getElementById('datasetSelect'),
  protocolSelect: document.getElementById('protocolSelect'),
  maxPcapsInput: document.getElementById('maxPcapsInput'),
  maxSessionsInput: document.getElementById('maxSessionsInput'),
  profileSelect: document.getElementById('profileSelect'),
  pruneModeSelect: document.getElementById('pruneModeSelect'),
  prunePercentileInput: document.getElementById('prunePercentileInput'),
  prunePercentileLabel: document.getElementById('prunePercentileLabel'),
  testRatioInput: document.getElementById('testRatioInput'),
  seedInput: document.getElementById('seedInput'),
  viewLevelSelect: document.getElementById('viewLevelSelect'),
  edgePercentileInput: document.getElementById('edgePercentileInput'),
  edgePercentileLabel: document.getElementById('edgePercentileLabel'),
  learnBtn: document.getElementById('learnBtn'),
  uploadBtn: document.getElementById('uploadBtn'),
  pcapFileInput: document.getElementById('pcapFileInput'),
  playBtn: document.getElementById('playBtn'),
  pauseBtn: document.getElementById('pauseBtn'),
  stepBtn: document.getElementById('stepBtn'),
  timelineInput: document.getElementById('timelineInput'),
  speedSelect: document.getElementById('speedSelect'),
  statusBox: document.getElementById('statusBox'),
  summaryList: document.getElementById('summaryList'),
  datasetMetrics: document.getElementById('datasetMetrics'),
  coreMetrics: document.getElementById('coreMetrics'),
  detailPanel: document.getElementById('detailPanel'),
  replayPanel: document.getElementById('replayPanel'),
  replayTable: document.getElementById('replayTable'),
  stepCounter: document.getElementById('stepCounter'),
};

const PROFILE_DEFAULTS = {
  fast: { max_pcaps: 4, max_sessions: 120 },
  balanced: { max_pcaps: 8, max_sessions: 300 },
  thorough: { max_pcaps: 12, max_sessions: 600 },
};

async function fetchJSON(url, options = {}) {
  const res = await fetch(url, options);
  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.detail || 'Request failed');
  }
  return body;
}

async function loadDatasets() {
  const data = await fetchJSON('/api/datasets');
  fillProtocolSelect(getRowsFromDatasetItems(data.items));

  el.datasetSelect.innerHTML = '';
  data.items.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.path;
    option.dataset.folderName = item.name;
    option.dataset.protocol = item.protocol || inferProtocolFromDatasetFolderName(item.name) || String(item.name).toUpperCase();
    option.dataset.mode = item.mode || 'pcap';
    option.textContent = item.label || `${item.name}  (${item.path})`;
    el.datasetSelect.appendChild(option);
  });

  if (!data.items.length) {
    el.protocolSelect.selectedIndex = 0;
  } else {
    for (let i = 0; i < el.datasetSelect.options.length; i++) {
      const opt = el.datasetSelect.options[i];
      if (opt.dataset.protocol === 'MODBUS' && opt.dataset.mode === 'pcap+synthetic') {
        el.datasetSelect.selectedIndex = i;
        break;
      }
    }
    syncProtocolSelectToDataset();
  }
}

/** 与当前选中的数据集目录名对齐协议下拉项 */
function syncProtocolSelectToDataset() {
  const dsOpt = el.datasetSelect.options[el.datasetSelect.selectedIndex];
  if (!dsOpt) return;
  const proto = dsOpt.dataset.protocol || inferProtocolFromDatasetFolderName(dsOpt.dataset.folderName) || el.protocolSelect.value;
  if (proto) el.protocolSelect.value = proto;
  applyDatasetModeDefaults();
}

/** 协议下拉变更时，选中同名数据集目录（若存在） */
function syncDatasetFromProtocol() {
  const pOpt = el.protocolSelect.options[el.protocolSelect.selectedIndex];
  if (!pOpt) return;
  const proto = pOpt.value;
  if (!proto) return;
  const modes = ['pcap+synthetic', 'pcap', 'synthetic'];
  for (let m = 0; m < modes.length; m++) {
    for (let i = 0; i < el.datasetSelect.options.length; i++) {
      const opt = el.datasetSelect.options[i];
      if (opt.dataset.protocol === proto && opt.dataset.mode === modes[m]) {
        el.datasetSelect.selectedIndex = i;
        applyDatasetModeDefaults();
        return;
      }
    }
  }
}

function applyDatasetModeDefaults() {
  const dsOpt = el.datasetSelect.options[el.datasetSelect.selectedIndex];
  if (!dsOpt) return;
  const mode = dsOpt.dataset.mode || 'pcap';
  if (mode !== 'pcap' && el.profileSelect) {
    el.profileSelect.value = 'fast';
    applyProfileDefaults();
  }
  const ms = Number(el.maxSessionsInput.value || 0);
  if (!Number.isFinite(ms) || ms < 120) {
    el.maxSessionsInput.value = '120';
  }
}

function setStatus(text) {
  el.statusBox.textContent = text;
}

function renderSummary(summary = {}) {
  el.summaryList.innerHTML = '';
  Object.entries(summary).forEach(([key, value]) => {
    const item = document.createElement('div');
    item.className = 'kv-item';
    item.innerHTML = `<span>${key}</span><strong>${Array.isArray(value) ? value.join(', ') : value}</strong>`;
    el.summaryList.appendChild(item);
  });
}

function renderMetricSection(target, title, metrics = {}) {
  target.innerHTML = '';
  const heading = document.createElement('div');
  heading.className = 'metric-title';
  heading.textContent = title;
  target.appendChild(heading);

  const entries = Object.entries(metrics || {});
  if (!entries.length) {
    const empty = document.createElement('div');
    empty.className = 'kv-item';
    empty.innerHTML = '<span>status</span><strong>(无指标)</strong>';
    target.appendChild(empty);
    return;
  }

  entries.forEach(([key, value]) => {
    const item = document.createElement('div');
    item.className = 'kv-item';
    const shown = typeof value === 'number' ? Number(value).toFixed(4) : value;
    item.innerHTML = `<span>${key}</span><strong>${shown}</strong>`;
    target.appendChild(item);
  });
}

function renderMetrics(metrics = null) {
  if (!metrics) {
    renderMetricSection(el.datasetMetrics, 'Dataset', {});
    renderMetricSection(el.coreMetrics, 'Core Metrics', {});
    return;
  }
  renderMetricSection(el.datasetMetrics, 'Dataset', metrics.dataset || {});
  renderMetricSection(el.coreMetrics, 'Core Metrics', metrics.core_metrics || {});
  const notes = metrics.metrics_notes || {};
  const helpText = [notes.primary_zh, notes.guard_reference_zh].filter(Boolean).join('\n\n');
  if (helpText) {
    const help = document.createElement('div');
    help.className = 'metric-help';
    help.textContent = helpText;
    el.coreMetrics.appendChild(help);
  }
}

function renderGraph(model) {
  const elements = [...model.nodes, ...model.edges];
  if (state.cy) {
    state.cy.destroy();
  }

  const nodeCount = (model.nodes || []).length;
  const layout = nodeCount > 250
    ? { name: 'breadthfirst', directed: true, spacingFactor: 1.2, padding: 10 }
    : { name: 'dagre', rankDir: 'LR', nodeSep: 36, rankSep: 80 };

  state.cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    layout,
    style: [
      {
        selector: 'node',
        style: {
          'background-color': '#f4ede2',
          'border-color': '#2f6f6d',
          'border-width': 'mapData(visit_count, 0, 30, 1, 6)',
          'label': 'data(label)',
          'color': '#1e2a2f',
          'font-size': 12,
          'text-valign': 'center',
          'text-halign': 'center',
          'width': 'mapData(visit_count, 0, 30, 42, 64)',
          'height': 'mapData(visit_count, 0, 30, 42, 64)',
        },
      },
      {
        selector: 'node[is_start = 1]',
        style: {
          'shape': 'round-rectangle',
          'background-color': '#d8efe9',
        },
      },
      {
        selector: 'node[is_end = 1]',
        style: {
          'shape': 'diamond',
          'background-color': '#f7d8ce',
        },
      },
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': '#64748b',
          'line-color': '#94a3b8',
          'width': 'mapData(probability, 0, 1, 2, 12)',
          'label': 'data(label)',
          'font-size': 11,
          'color': '#334155',
          'text-background-color': '#fffaf3',
          'text-background-opacity': 0.88,
          'text-background-padding': 3,
        },
      },
      {
        selector: 'edge[confidence = "high"]',
        style: {
          'line-color': '#2563eb',
          'target-arrow-color': '#2563eb',
        },
      },
      {
        selector: 'edge[confidence = "medium"]',
        style: {
          'line-color': '#eab308',
          'target-arrow-color': '#eab308',
        },
      },
      {
        selector: 'edge[confidence = "low"]',
        style: {
          'line-color': '#94a3b8',
          'target-arrow-color': '#94a3b8',
          'line-style': 'dashed',
        },
      },
      {
        selector: '.active-node',
        style: {
          'overlay-color': '#b85c38',
          'overlay-opacity': 0.2,
          'overlay-padding': 12,
          'border-color': '#b85c38',
        },
      },
      {
        selector: '.active-edge',
        style: {
          'overlay-color': '#b85c38',
          'overlay-opacity': 0.25,
          'overlay-padding': 6,
          'line-color': '#b85c38',
          'target-arrow-color': '#b85c38',
        },
      },
      {
        selector: '.error-edge',
        style: {
          'line-color': '#dc2626',
          'target-arrow-color': '#dc2626',
        },
      },
    ],
  });

  state.cy.on('tap', 'node, edge', (evt) => {
    const data = evt.target.data();
    if (evt.target.isEdge && evt.target.isEdge()) {
      const sections = [
        `transition: ${data.source} --[${data.symbol}]--> ${data.target}`,
        `state_probability: ${formatNum(data.probability)}`,
        `count: ${data.count}`,
        `confidence: ${data.confidence}`,
        '',
        '[guard]',
        data.guard_text || '(无 guard)',
        '',
        '[cross-message]',
        data.cross_message_text || '(无跨消息规则)',
        '',
        '[action]',
        data.action_text || '(无 action)',
        '',
        '[raw data]',
        JSON.stringify(data, null, 2),
      ];
      el.detailPanel.textContent = sections.join('\n');
      return;
    }
    el.detailPanel.textContent = JSON.stringify(data, null, 2);
  });
}

function calcCountThreshold(edges, percentile) {
  const vals = edges.map((e) => Number(e.data && e.data.count) || 0).sort((a, b) => a - b);
  if (!vals.length) return 0;
  const p = Math.max(0, Math.min(99, Number(percentile) || 0));
  const idx = Math.max(0, Math.min(vals.length - 1, Math.floor((p / 100) * (vals.length - 1))));
  return vals[idx];
}

function computeViewModel(fullModel, level, percentile) {
  if (!fullModel) return null;
  const nodes = fullModel.nodes || [];
  const edges = fullModel.edges || [];
  const startId = (fullModel.summary && fullModel.summary.start_state) || null;
  const startNodeId = startId || (nodes.find((n) => n.data && n.data.is_start) || {}).data?.id || null;
  const endIds = new Set(nodes.filter((n) => n.data && n.data.is_end).map((n) => n.data.id));

  if (level === 'full') {
    return {
      nodes,
      edges,
      summary: {
        ...(fullModel.summary || {}),
        view_level: 'full',
        view_states: nodes.length,
        view_transitions: edges.length,
      },
    };
  }

  const threshold = calcCountThreshold(edges, percentile);
  const keptEdges = edges.filter((e) => (Number(e.data && e.data.count) || 0) >= threshold);
  const nodeIds = new Set();
  keptEdges.forEach((e) => {
    nodeIds.add(e.data.source);
    nodeIds.add(e.data.target);
  });
  const keptNodes = nodes.filter((n) => nodeIds.has(n.data.id));

  if (level === 'freq') {
    return {
      nodes: keptNodes,
      edges: keptEdges,
      summary: {
        ...(fullModel.summary || {}),
        view_level: `freq_p${percentile}`,
        edge_count_threshold: threshold,
        view_states: keptNodes.length,
        view_transitions: keptEdges.length,
      },
    };
  }

  if (!startNodeId || !endIds.size) {
    return {
      nodes: keptNodes,
      edges: keptEdges,
      summary: {
        ...(fullModel.summary || {}),
        view_level: `backbone_p${percentile}`,
        edge_count_threshold: threshold,
        view_states: keptNodes.length,
        view_transitions: keptEdges.length,
      },
    };
  }

  const adj = new Map();
  const radj = new Map();
  keptEdges.forEach((e) => {
    const s = e.data.source;
    const t = e.data.target;
    if (!adj.has(s)) adj.set(s, []);
    if (!radj.has(t)) radj.set(t, []);
    adj.get(s).push(t);
    radj.get(t).push(s);
  });

  const reachFromStart = new Set();
  const q1 = [startNodeId];
  while (q1.length) {
    const cur = q1.shift();
    if (reachFromStart.has(cur)) continue;
    reachFromStart.add(cur);
    const outs = adj.get(cur) || [];
    outs.forEach((nxt) => {
      if (!reachFromStart.has(nxt)) q1.push(nxt);
    });
  }

  const canReachEnd = new Set();
  const q2 = Array.from(endIds);
  while (q2.length) {
    const cur = q2.shift();
    if (canReachEnd.has(cur)) continue;
    canReachEnd.add(cur);
    const ins = radj.get(cur) || [];
    ins.forEach((pre) => {
      if (!canReachEnd.has(pre)) q2.push(pre);
    });
  }

  const backboneNodes = nodes.filter((n) => reachFromStart.has(n.data.id) && canReachEnd.has(n.data.id));
  const backboneIds = new Set(backboneNodes.map((n) => n.data.id));
  const backboneEdges = keptEdges.filter((e) => backboneIds.has(e.data.source) && backboneIds.has(e.data.target));

  if (!backboneEdges.length && keptEdges.length) {
    return {
      nodes: keptNodes,
      edges: keptEdges,
      summary: {
        ...(fullModel.summary || {}),
        view_level: `freq_p${percentile}`,
        edge_count_threshold: threshold,
        view_states: keptNodes.length,
        view_transitions: keptEdges.length,
      },
    };
  }

  return {
    nodes: backboneNodes,
    edges: backboneEdges,
    summary: {
      ...(fullModel.summary || {}),
      view_level: `backbone_p${percentile}`,
      edge_count_threshold: threshold,
      view_states: backboneNodes.length,
      view_transitions: backboneEdges.length,
    },
  };
}

function applyGraphView() {
  if (!state.model) return;
  const level = (el.viewLevelSelect && el.viewLevelSelect.value) || state.view.level || 'full';
  const percentile = Number((el.edgePercentileInput && el.edgePercentileInput.value) || state.view.percentile || 70);
  state.view = { level, percentile };
  if (el.edgePercentileLabel) {
    el.edgePercentileLabel.textContent = `边频率阈值（百分位）：${percentile}`;
  }
  state.viewModel = computeViewModel(state.model, level, percentile) || state.model;
  renderGraph(state.viewModel);
  renderSummary(state.viewModel.summary || {});
  highlightStep(state.replay.steps[state.currentStep]);
}

function renderReplayTable() {
  el.replayTable.innerHTML = '';
  state.replay.steps.forEach((step, index) => {
    const row = document.createElement('div');
    row.className = `replay-row${index === state.currentStep ? ' active' : ''}${step.matched ? '' : ' error'}`;
    row.textContent = `#${index}  ${step.src} --[${step.symbol}]--> ${step.dst || 'unmatched'}  prob=${formatNum(step.probability)}  session=${step.session}`;
    row.addEventListener('click', () => {
      setCurrentStep(index);
      pauseReplay();
    });
    el.replayTable.appendChild(row);
  });
}

function renderReplayPanel(step) {
  if (!step) {
    el.replayPanel.textContent = '尚未开始回放';
    return;
  }
  el.replayPanel.textContent = JSON.stringify(step, null, 2);
}

function formatNum(value) {
  return value == null ? 'n/a' : Number(value).toFixed(4);
}

function setCurrentStep(index) {
  state.currentStep = Math.max(0, Math.min(index, state.replay.steps.length - 1));
  el.timelineInput.value = String(state.currentStep);
  el.stepCounter.textContent = `${state.replay.steps.length ? state.currentStep + 1 : 0} / ${state.replay.steps.length}`;

  const step = state.replay.steps[state.currentStep];
  renderReplayPanel(step);
  renderReplayTable();
  highlightStep(step);
}

function clearHighlights() {
  if (!state.cy) return;
  state.cy.elements().removeClass('active-node active-edge error-edge');
}

function highlightStep(step) {
  clearHighlights();
  if (!state.cy || !step) return;
  const srcNode = state.cy.getElementById(step.src);
  if (srcNode) srcNode.addClass('active-node');
  if (step.dst) {
    const dstNode = state.cy.getElementById(step.dst);
    if (dstNode) dstNode.addClass('active-node');
  }
  if (step.transition_id != null) {
    const edge = state.cy.getElementById(`t${step.transition_id}`);
    if (edge) edge.addClass(step.matched ? 'active-edge' : 'error-edge');
  }
}

function startReplay() {
  pauseReplay();
  const delay = Number(el.speedSelect.value || 800);
  state.timer = setInterval(() => {
    if (state.currentStep >= state.replay.steps.length - 1) {
      pauseReplay();
      return;
    }
    setCurrentStep(state.currentStep + 1);
  }, delay);
}

function pauseReplay() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

async function learnModel() {
  pauseReplay();
  setStatus('正在学习 P-EFSM ...');
  const form = new FormData();
  form.set('protocol', el.protocolSelect.value);
  form.set('data_dir', el.datasetSelect.value);
  form.set('max_pcaps', el.maxPcapsInput.value);
  form.set('max_sessions', el.maxSessionsInput.value);
  form.set('profile', el.profileSelect.value || 'balanced');
  form.set('test_ratio', el.testRatioInput.value);
  form.set('seed', el.seedInput.value);
  const dsOpt = el.datasetSelect.options[el.datasetSelect.selectedIndex];
  const mode = (dsOpt && dsOpt.dataset.mode) || 'pcap';
  form.set('dataset_mode', mode);
  form.set('synthetic_sessions', '0');
  form.set('synthetic_session_len', '20');
  form.set('prune_mode', (el.pruneModeSelect && el.pruneModeSelect.value) || 'none');
  form.set('prune_percentile', (el.prunePercentileInput && el.prunePercentileInput.value) || '70');

  const data = await fetchJSON('/api/learn', { method: 'POST', body: form });
  state.artifactId = data.artifact_id;
  state.model = data.model;
  state.metrics = data.metrics || null;
  state.replay = data.replay;
  applyGraphView();
  renderMetrics(state.metrics);
  el.timelineInput.max = String(Math.max(0, state.replay.steps.length - 1));
  setCurrentStep(0);
  setStatus(`学习完成，artifact_id=${state.artifactId}`);
}

function applyProfileDefaults() {
  const p = (el.profileSelect && el.profileSelect.value) || 'balanced';
  const d = PROFILE_DEFAULTS[p] || PROFILE_DEFAULTS.balanced;
  if (el.maxPcapsInput) el.maxPcapsInput.value = String(d.max_pcaps);
  if (el.maxSessionsInput) el.maxSessionsInput.value = String(d.max_sessions);
}

async function uploadPcap() {
  if (!state.artifactId) {
    setStatus('请先学习一个 P-EFSM 模型');
    return;
  }
  const file = el.pcapFileInput.files[0];
  if (!file) {
    setStatus('请先选择一个 pcap 文件');
    return;
  }
  pauseReplay();
  setStatus(`正在上传 ${file.name} 并生成回放 ...`);
  const form = new FormData();
  form.set('file', file);
  const data = await fetchJSON(`/api/artifacts/${state.artifactId}/upload-pcap`, { method: 'POST', body: form });
  state.replay = data;
  el.timelineInput.max = String(Math.max(0, state.replay.steps.length - 1));
  setCurrentStep(0);
  setStatus(`上传完成，回放步数=${state.replay.summary.total_steps}`);
}

function bindEvents() {
  el.datasetSelect.addEventListener('change', () => syncProtocolSelectToDataset());
  el.protocolSelect.addEventListener('change', () => syncDatasetFromProtocol());
  if (el.profileSelect) {
    el.profileSelect.addEventListener('change', applyProfileDefaults);
  }
  if (el.viewLevelSelect) {
    el.viewLevelSelect.addEventListener('change', () => applyGraphView());
  }
  if (el.edgePercentileInput) {
    el.edgePercentileInput.addEventListener('input', () => applyGraphView());
  }
  if (el.prunePercentileInput) {
    el.prunePercentileInput.addEventListener('input', () => {
      if (el.prunePercentileLabel) {
        el.prunePercentileLabel.textContent = `剪枝阈值（百分位）：${el.prunePercentileInput.value}`;
      }
    });
  }
  el.learnBtn.addEventListener('click', () => learnModel().catch((err) => setStatus(err.message)));
  el.uploadBtn.addEventListener('click', () => uploadPcap().catch((err) => setStatus(err.message)));
  el.playBtn.addEventListener('click', startReplay);
  el.pauseBtn.addEventListener('click', pauseReplay);
  el.stepBtn.addEventListener('click', () => {
    pauseReplay();
    if (state.replay.steps.length) {
      setCurrentStep(Math.min(state.currentStep + 1, state.replay.steps.length - 1));
    }
  });
  el.timelineInput.addEventListener('input', (evt) => {
    pauseReplay();
    setCurrentStep(Number(evt.target.value));
  });
}

async function bootstrap() {
  if (!el.protocolSelect || !el.datasetSelect) {
    setStatus('页面控件未找到（protocolSelect / datasetSelect），请 Ctrl+F5 强制刷新');
    return;
  }
  bindEvents();
  applyProfileDefaults();
  renderMetrics(null);
  try {
    await loadDatasets();
    setStatus('数据集已加载，可以开始学习');
  } catch (err) {
    setStatus(`加载数据集目录失败：${err.message}（协议下拉已保留页面默认选项）`);
  }
}

bootstrap().catch((err) => setStatus(err.message));
