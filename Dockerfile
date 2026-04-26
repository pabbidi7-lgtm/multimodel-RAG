<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>LangGraph Agentic Workflow — Binary Decision Tree</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #0d0d0d;
    color: #e8e8e8;
    font-family: 'Courier New', 'Consolas', monospace;
    padding: 40px 32px;
    min-height: 100vh;
  }

  .header {
    border: 1px solid #333;
    border-bottom: 2px solid #555;
    padding: 14px 20px;
    margin-bottom: 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .header-title { font-size: 13px; letter-spacing: 0.1em; color: #fff; font-weight: bold; }
  .header-sub { font-size: 11px; letter-spacing: 0.06em; color: #666; }

  .legend {
    display: flex; gap: 28px; margin-bottom: 32px; padding: 0 4px;
  }
  .legend-item { display: flex; align-items: center; gap: 8px; font-size: 11px; color: #888; letter-spacing: 0.05em; }
  .leg-line { width: 32px; height: 1px; }
  .leg-solid { background: #ccc; }
  .leg-thin { background: #555; }
  .leg-dash { background: none; border-top: 1px dashed #e8a020; }

  .tree-wrap {
    display: flex;
    gap: 0;
  }

  .left-bypass {
    width: 80px;
    flex-shrink: 0;
    position: relative;
  }

  .right-retry {
    width: 80px;
    flex-shrink: 0;
    position: relative;
  }

  .spine {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* Node */
  .node-wrap {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    position: relative;
  }

  .node {
    width: 340px;
    border: 1px solid #444;
    border-top: 2.5px solid #aaa;
    background: #111;
  }

  .node-head {
    padding: 9px 16px;
    border-bottom: 1px solid #2a2a2a;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 0.06em;
    color: #fff;
  }

  .node-body {
    padding: 8px 16px 10px;
  }

  .node-body p {
    font-size: 11px;
    color: #888;
    letter-spacing: 0.03em;
    line-height: 1.7;
  }

  /* Connectors */
  .arrow-down {
    width: 1px;
    background: #555;
    height: 32px;
    position: relative;
    margin: 0 auto;
  }
  .arrow-down::after {
    content: '';
    position: absolute;
    bottom: -5px;
    left: -4px;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #555;
  }

  .edge-label {
    font-size: 10px;
    color: #666;
    letter-spacing: 0.05em;
    margin: 2px 0;
    text-align: center;
  }

  /* Start / end nodes */
  .terminal {
    width: 200px;
    border: 1px solid #555;
    padding: 10px 0;
    text-align: center;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 0.08em;
    color: #ccc;
  }

  /* Bypass lines drawn as SVG overlays */
  .diagram-container {
    position: relative;
  }

  /* Retry traversal panel */
  .retry-panel {
    margin-top: 48px;
    border: 1px solid #2a2a2a;
    border-top: 2px solid #e8a020;
  }

  .retry-header {
    background: #161616;
    padding: 10px 18px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.1em;
    color: #e8a020;
    border-bottom: 1px solid #2a2a2a;
  }

  .retry-nodes {
    display: flex;
    align-items: center;
    padding: 18px 18px 14px;
    gap: 0;
    flex-wrap: nowrap;
    overflow-x: auto;
  }

  .r-node {
    border: 1px solid #444;
    border-top: 2px solid #aaa;
    background: #111;
    min-width: 80px;
    text-align: center;
    padding: 8px 10px 10px;
    flex-shrink: 0;
  }
  .r-node.start-node { border-top-color: #e8a020; }
  .r-node.re-entry { border-top-color: #e8a020; }

  .r-node .rn-label { font-size: 12px; font-weight: bold; color: #fff; letter-spacing: 0.04em; }
  .r-node .rn-sub   { font-size: 10px; color: #666; margin-top: 3px; letter-spacing: 0.03em; }

  .r-arrow {
    display: flex; flex-direction: column; align-items: center; padding: 0 6px; flex-shrink: 0;
  }
  .r-arrow .ra-line { width: 24px; height: 1px; background: #555; }
  .r-arrow .ra-dash { width: 24px; height: 1px; background: none; border-top: 1px dashed #e8a020; }
  .r-arrow .ra-tip  { font-size: 10px; color: #555; margin-left: 14px; line-height: 1; }
  .r-arrow .ra-tip-orange { color: #e8a020; }
  .r-arrow .ra-label { font-size: 9px; color: #555; letter-spacing: 0.04em; margin-top: 3px; text-align: center; width: 36px; }

  .retry-steps {
    border-top: 1px solid #1e1e1e;
    padding: 14px 18px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 32px;
  }
  .retry-steps p {
    font-size: 11px;
    color: #777;
    line-height: 1.6;
    letter-spacing: 0.02em;
  }
  .retry-steps p span { color: #bbb; }

  .footer-legend {
    margin-top: 20px;
    padding: 12px 18px;
    border: 1px solid #1e1e1e;
    display: flex;
    gap: 32px;
    flex-wrap: wrap;
  }
  .footer-legend p { font-size: 10px; color: #555; letter-spacing: 0.04em; line-height: 1.7; }

  /* The full diagram uses a relative container + absolute SVG for bypass lines */
  .full-diagram {
    display: flex;
    gap: 0;
    position: relative;
  }

  .bypass-col {
    width: 72px;
    flex-shrink: 0;
  }
  .retry-col {
    width: 72px;
    flex-shrink: 0;
  }
</style>
</head>
<body>

<div class="header">
  <span class="header-title">LANGGRAPH · AGENTIC RAG · BINARY DECISION TREE</span>
  <span class="header-sub">5 NODES · 3 BYPASS PATHS · 1 RETRY LOOP</span>
</div>

<div class="legend">
  <div class="legend-item">
    <div class="leg-line leg-solid"></div>
    <span>happy path (right / down)</span>
  </div>
  <div class="legend-item">
    <div class="leg-line leg-thin"></div>
    <span>bypass / short-circuit (left)</span>
  </div>
  <div class="legend-item">
    <div class="leg-line leg-dash"></div>
    <span>retry traversal up-tree (orange)</span>
  </div>
</div>

<!-- Main diagram: spine + SVG overlay for bypass and retry lines -->
<div class="full-diagram">

  <!-- Left bypass rail placeholder -->
  <div class="bypass-col" id="bypassCol"></div>

  <!-- Center spine -->
  <div class="spine" id="spine">

    <div class="terminal" id="n-start">USER QUERY</div>

    <div class="arrow-down"></div>

    <div class="node-wrap" id="nw1">
      <div class="node" id="n1">
        <div class="node-head">N1 &mdash; GUARDRAIL</div>
        <div class="node-body">
          <p>Injection check &middot; filler-word strip</p>
          <p>Intent label: text / table / chart / image</p>
        </div>
      </div>
    </div>

    <div class="edge-label">clean query</div>
    <div class="arrow-down"></div>

    <div class="node-wrap" id="nw2">
      <div class="node" id="n2">
        <div class="node-head">N2 &mdash; QUERY EXPANDER</div>
        <div class="node-body">
          <p>LLM generates 2 paraphrases of cleaned query</p>
          <p>Output: 3 variants &middot; fallback: 1 if LLM fails</p>
        </div>
      </div>
    </div>

    <div class="edge-label">3 variants</div>
    <div class="arrow-down"></div>

    <div class="node-wrap" id="nw3">
      <div class="node" id="n3">
        <div class="node-head">N3 &mdash; RETRIEVER</div>
        <div class="node-body">
          <p>3&times; independent Milvus ANN search (top-k = 40)</p>
          <p>Merge results &middot; SHA256 dedup across variants</p>
          <p>Embed via nv-embedqa-e5-v5 &middot; input_type = query</p>
        </div>
      </div>
    </div>

    <div class="edge-label">ranked candidates</div>
    <div class="arrow-down"></div>

    <div class="node-wrap" id="nw4">
      <div class="node" id="n4">
        <div class="node-head">N4 &mdash; RERANKER + QUALITY GATE</div>
        <div class="node-body">
          <p>cross-encoder ms-marco-MiniLM-L-12-v2 &middot; top-k = 15</p>
          <p>Confidence: high &ge; &minus;3.0 &middot; medium &ge; &minus;8.0</p>
          <p>Quality gate: skip LLM if top score &lt; &minus;10.0</p>
        </div>
      </div>
    </div>

    <div class="edge-label">top-k chunks</div>
    <div class="arrow-down"></div>

    <div class="node-wrap" id="nw5">
      <div class="node" id="n5">
        <div class="node-head">N5 &mdash; GENERATOR</div>
        <div class="node-body">
          <p>Primary: meta/llama-3.3-70b-instruct</p>
          <p>Fallback: nvidia/llama-3.1-nemotron-70b-instruct</p>
          <p>Prepend last 3 conversation turns &middot; hallucination check</p>
          <p>retry_count max = 1</p>
        </div>
      </div>
    </div>

    <div class="edge-label">answer</div>
    <div class="arrow-down"></div>

    <div class="terminal" id="n-end">ANSWER OUTPUT</div>

  </div>

  <!-- Right retry rail placeholder -->
  <div class="retry-col" id="retryCol"></div>

</div>

<!-- SVG overlay for bypass + retry lines -->
<svg id="overlay" style="position:absolute;top:0;left:0;pointer-events:none;overflow:visible" width="0" height="0">
  <defs>
    <marker id="ma" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M1 1L9 5L1 9" fill="none" stroke="#555" stroke-width="2" stroke-linecap="round"/>
    </marker>
    <marker id="mo" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M1 1L9 5L1 9" fill="none" stroke="#e8a020" stroke-width="2" stroke-linecap="round"/>
    </marker>
  </defs>
</svg>

<!-- Retry traversal panel -->
<div class="retry-panel">
  <div class="retry-header">RETRY TRAVERSAL &mdash; NODE-BY-NODE PATH</div>
  <div class="retry-nodes">
    <div class="r-node start-node">
      <div class="rn-label">N5</div>
      <div class="rn-sub">detect<br>hallucination</div>
    </div>
    <div class="r-arrow">
      <div class="ra-dash"></div>
      <div class="ra-tip ra-tip-orange">&#9658;</div>
      <div class="ra-label" style="color:#e8a020">climb<br>to N2</div>
    </div>
    <div class="r-node re-entry">
      <div class="rn-label">N2</div>
      <div class="rn-sub">re-enter<br>stricter query</div>
    </div>
    <div class="r-arrow">
      <div class="ra-line"></div>
      <div class="ra-tip">&#9658;</div>
      <div class="ra-label">expand</div>
    </div>
    <div class="r-node">
      <div class="rn-label">N3</div>
      <div class="rn-sub">re-fetch<br>Milvus</div>
    </div>
    <div class="r-arrow">
      <div class="ra-line"></div>
      <div class="ra-tip">&#9658;</div>
      <div class="ra-label">rank</div>
    </div>
    <div class="r-node">
      <div class="rn-label">N4</div>
      <div class="rn-sub">re-rank<br>quality gate</div>
    </div>
    <div class="r-arrow">
      <div class="ra-line"></div>
      <div class="ra-tip">&#9658;</div>
      <div class="ra-label">generate</div>
    </div>
    <div class="r-node">
      <div class="rn-label">N5</div>
      <div class="rn-sub">re-generate<br>emit result</div>
    </div>
    <div class="r-arrow">
      <div class="ra-line"></div>
      <div class="ra-tip">&#9658;</div>
    </div>
    <div class="r-node" style="border-top-color:#555;min-width:100px">
      <div class="rn-label" style="color:#aaa;font-size:11px">FINAL ANSWER</div>
      <div class="rn-sub">retry_count=1<br>stop</div>
    </div>
  </div>

  <div class="retry-steps">
    <p><span>1.</span> N5 detects hallucination phrase in LLM output (retry_count &lt; 1)</p>
    <p><span>2.</span> State cloned — answer, chunks, sources all cleared</p>
    <p><span>3.</span> Query rewritten: original + "use ONLY exact facts from document"</p>
    <p><span>4.</span> Graph skips N1 — re-enters at N2 with the stricter query</p>
    <p><span>5.</span> N2 → N3 → N4 → N5 all re-execute on the new query</p>
    <p><span>6.</span> Second pass emits result regardless — retry_count = 1, stop</p>
    <p><span>7.</span> Conversation memory updated only on clean (non-hallucinated) answer</p>
    <p><span>8.</span> wall_ms covers both full passes combined including retry</p>
  </div>
</div>

<div class="footer-legend">
  <p>DOWN arrow &rarr; happy path continues<br>LEFT bypass &rarr; injection (N1) or gate fail (N4) short-circuits to output<br>N3 no-hits &rarr; empty state passed through; generator returns "not found"</p>
  <p>DASHED orange &rarr; retry: N5 climbs back to N2 only (N1 not re-run)<br>Subtree traversed on retry: N2 &rarr; N3 &rarr; N4 &rarr; N5<br>Max retry depth = 1 (hard limit in _prepare_retry_state)</p>
</div>

<script>
(function() {
  const diagramEl = document.querySelector('.full-diagram');
  const svg = document.getElementById('overlay');
  const diagRect = diagramEl.getBoundingClientRect();

  svg.style.position = 'absolute';
  diagramEl.style.position = 'relative';
  diagramEl.appendChild(svg);

  function getCenter(el) {
    const r = el.getBoundingClientRect();
    const pr = diagramEl.getBoundingClientRect();
    return {
      x: r.left - pr.left + r.width / 2,
      y: r.top - pr.top + r.height / 2,
      top: r.top - pr.top,
      bottom: r.top - pr.top + r.height,
      left: r.left - pr.left,
      right: r.left - pr.left + r.width,
    };
  }

  const n1 = getCenter(document.getElementById('n1'));
  const n2 = getCenter(document.getElementById('n2'));
  const n3 = getCenter(document.getElementById('n3'));
  const n4 = getCenter(document.getElementById('n4'));
  const n5 = getCenter(document.getElementById('n5'));
  const nEnd = getCenter(document.getElementById('n-end'));

  const totalH = n5.bottom + 40;
  svg.setAttribute('width', diagramEl.offsetWidth);
  svg.setAttribute('height', totalH + 20);

  function line(x1,y1,x2,y2,stroke,dash,marker) {
    const el = document.createElementNS('http://www.w3.org/2000/svg','line');
    el.setAttribute('x1',x1); el.setAttribute('y1',y1);
    el.setAttribute('x2',x2); el.setAttribute('y2',y2);
    el.setAttribute('stroke',stroke);
    el.setAttribute('stroke-width','1');
    if (dash) el.setAttribute('stroke-dasharray',dash);
    if (marker) el.setAttribute('marker-end',marker);
    svg.appendChild(el);
  }
  function path(d,stroke,dash,marker) {
    const el = document.createElementNS('http://www.w3.org/2000/svg','path');
    el.setAttribute('d',d); el.setAttribute('fill','none');
    el.setAttribute('stroke',stroke); el.setAttribute('stroke-width','1');
    if (dash) el.setAttribute('stroke-dasharray',dash);
    if (marker) el.setAttribute('marker-end',marker);
    svg.appendChild(el);
  }
  function txt(x,y,label,color,anchor) {
    const el = document.createElementNS('http://www.w3.org/2000/svg','text');
    el.setAttribute('x',x); el.setAttribute('y',y);
    el.setAttribute('font-size','10');
    el.setAttribute('font-family','Courier New, monospace');
    el.setAttribute('fill', color || '#555');
    el.setAttribute('text-anchor', anchor || 'middle');
    el.textContent = label;
    svg.appendChild(el);
  }

  const bx = n1.left - 52;

  // N1 bypass → down left rail → to n-end y
  path(`M${n1.left} ${n1.y} L${bx} ${n1.y} L${bx} ${nEnd.y} L${nEnd.left} ${nEnd.y}`,
    '#444', null, 'url(#ma)');
  txt(bx - 2, (n1.y + n4.y)/2, 'inject / gate', '#444', 'end');

  // N3 no-hits small branch
  const bx2 = n3.left - 36;
  path(`M${n3.left} ${n3.y} L${bx2} ${n3.y} L${bx2} ${n4.top} L${n4.left} ${n4.top + 8}`,
    '#333', null, 'url(#ma)');
  txt(bx2 - 2, n3.y - 6, 'no hits', '#333', 'end');

  // N4 quality gate → same left rail
  const n4BypassY = n4.y + 10;
  path(`M${n4.left} ${n4BypassY} L${bx} ${n4BypassY} L${bx} ${nEnd.y}`,
    '#444', null, null);

  // Retry: N5 right → up → N2 right
  const rx = n5.right + 48;
  path(`M${n5.right} ${n5.y} L${rx} ${n5.y} L${rx} ${n2.y} L${n2.right} ${n2.y}`,
    '#e8a020', '6 3', 'url(#mo)');
  txt(rx + 4, (n2.y + n5.y)/2, 'hallucination → retry at N2', '#e8a020', 'start');

})();
</script>
</body>
</html>
