/* The exported DOM/CSS owns presentation. Only live values and actions are bound here. */
(() => {
  // Keep export example values from flashing before the first native snapshot.
  if (window.Wellbeing) document.documentElement.style.visibility = 'hidden';
  const $ = id => document.getElementById(id);
  const text = (id, value) => { if ($(id)) $(id).textContent = value; };
  const act = (action, data = {}) => window.Wellbeing?.action(action, JSON.stringify(data));
  document.querySelectorAll('[data-page]').forEach(el => el.addEventListener('click', e => {
    e.preventDefault(); location.href = el.dataset.page + '.html';
  }));
  function values() {
    return Object.fromEntries([...document.querySelectorAll('input[name]')].map(el => [el.name, el.value]));
  }
  document.querySelectorAll('[data-action]').forEach(el => el.addEventListener('click', () => {
    act(el.dataset.action, values());
    if (el.dataset.action === 'save') text('settings-notice', 'Settings saved');
  }));
  function curve(values, width, height, top = 2) {
    if (values.length < 2) return '';
    const lo = Math.min(...values), span = Math.max(...values) - lo || 1;
    return values.map((v, i) => `${i ? 'L' : 'M'}${(i * width / (values.length - 1)).toFixed(2)},${(height - top - (v - lo) / span * (height - top * 2)).toFixed(2)}`).join(' ');
  }
  function renderTrainingChart(t, history) {
    if (!$('training-line')) return;
    if (!$('training-scroll')) {
      const host = $('training-line').closest('svg').parentElement.parentElement.parentElement;
      host.style.cssText = 'height:144px;display:flex;gap:8px;padding-top:8px';
      host.innerHTML = `<div id="training-y-axis" style="width:36px;flex-shrink:0;height:112px;padding-top:4px;padding-bottom:4px;display:flex;flex-direction:column;justify-content:space-between;text-align:right;font-size:10px;color:#777587"></div>
        <div id="training-scroll" style="flex:1;min-width:0;overflow:hidden"><div id="training-plot">
        <svg id="training-svg" height="112" style="display:block;overflow:visible"><defs><linearGradient id="live-ppl-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#4f46e5" stop-opacity=".2"/><stop offset="100%" stop-color="#4f46e5" stop-opacity="0"/></linearGradient></defs>
        <g id="training-grid"></g><path id="training-area" fill="url(#live-ppl-fill)"/>
        <path id="training-line" fill="none" stroke="#4f46e5" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <g id="training-dots"></g></svg><div id="training-x-axis" style="position:relative;height:24px;font-size:11px;color:#777587;font-weight:500"></div></div></div>`;
    }
    const points = history.map(p => ({...p, index: (Math.max(1,p.round)-1) * t.total + p.step}));
    const total = t.total > 0 ? t.total * Math.max(1,t.totalRounds) : 0;
    const width = $('training-scroll').clientWidth;
    // Reserve both end labels while keeping the complete session on one axis.
    const padding = 18;
    const stepX = step => padding + (total > 1 ? (step - 1) / (total - 1) : 0) * (width - 2*padding);
    const x = p => stepX(p.index);
    const last = points.at(-1);
    const end = last ? x(last) : padding;
    const high = points.reduce((max,p)=>Math.max(max,p.ppl),0);
    $('training-svg').dataset.maximum = high;
    const y = p => 100-p.ppl/(high || 1)*90;
    $('training-plot').style.width = width+'px';
    $('training-svg').setAttribute('width',width);
    $('training-svg').setAttribute('viewBox',`0 0 ${width} 112`);
    const path = points.map((p,i)=>`${i?'L':'M'}${x(p).toFixed(2)},${y(p).toFixed(2)}`).join(' ');
    $('training-line').setAttribute('d',path);
    $('training-area').setAttribute('d',points.length>1 ? `${path} L${end},100 L${x(points[0])},100 Z` : '');
    $('training-grid').replaceChildren();
    (points.length ? [10,55,100] : [100]).forEach(yy=>{
      const line=document.createElementNS('http://www.w3.org/2000/svg','line');
      Object.entries({x1:padding,x2:width-padding,y1:yy,y2:yy,stroke:'#d3e4fe','stroke-dasharray':yy===100?'0':'4 4'}).forEach(([k,v])=>line.setAttribute(k,v));
      $('training-grid').append(line);
    });
    $('training-y-axis').replaceChildren(...(points.length?[high,high/2,0]:['–','–','–']).map(value=>{
      const el=document.createElement('span');el.textContent=typeof value==='number'?Number(value.toPrecision(4)).toString():value;return el;
    }));
    // Each completed step retains its dot; ticks describe the full planned run.
    const keep=new Set(points.map(p=>String(p.index)));
    [...$('training-dots').children].forEach(el=>{if(!keep.has(el.dataset.step))el.remove();});
    points.forEach(p=>{
      let dot=$('training-dots').querySelector(`[data-step="${p.index}"]`);
      if(!dot){dot=document.createElementNS('http://www.w3.org/2000/svg','circle');dot.dataset.step=p.index;dot.setAttribute('r','3.5');dot.setAttribute('fill','#f8f9ff');dot.setAttribute('stroke','#4f46e5');dot.setAttribute('stroke-width','2');$('training-dots').append(dot);}
      dot.setAttribute('cx',x(p));dot.setAttribute('cy',y(p));dot.setAttribute('aria-label',`Step ${p.index}, PPL ${p.ppl.toFixed(2)}`);
    });
    const divisions = Math.min(5,Math.max(0,total-1));
    const ticks = total > 1 ? Array.from({length:divisions+1},(_,i)=>1+Math.round((total-1)*i/divisions)) : total === 1 ? [1] : [];
    $('training-x-axis').replaceChildren(...ticks.map(step=>{
      const tick=document.createElement('span');tick.dataset.step=step;
      tick.style.cssText='position:absolute;top:0;transform:translateX(-50%);color:#777587;font-weight:500';
      tick.style.left=stepX(step)+'px';tick.textContent=step;return tick;
    }));
  }
  function renderAggregation(a) {
    if (!$('status-server')) return;
    if (!$('status-aggregation')) {
      const row=$('status-server').parentElement.parentElement.cloneNode(true);
      row.querySelector('#status-server').id='status-aggregation';
      row.firstElementChild.lastElementChild.textContent='Federated Server';
      row.querySelector('.material-symbols-outlined').textContent='hub';
      $('status-server').parentElement.parentElement.parentElement.append(row);
    }
    const phase=a?.phase || 'OFFLINE';
    const label={IDLE:'Idle',WAITING:`Waiting ${a?.received??0}/${a?.quorum??'–'}`,AGGREGATING:'Aggregating',COMPLETE:'Complete',FAILED:'Failed',OFFLINE:'Disconnected',NOT_CONNECTED:'Disconnected',UNCONFIGURED:'Not configured',MISMATCH:'Other session'}[phase] || 'Unknown';
    text('status-aggregation',label);
    const color=phase==='AGGREGATING'?'#885500':phase==='COMPLETE'?'#006c49':phase==='FAILED'?'#ba1a1a':'#777587';
    $('status-aggregation').style.color=color;
    $('status-aggregation').previousElementSibling.style.backgroundColor=color;
    $('status-aggregation').parentElement.parentElement.title=a?.message || `Aggregation server · round ${a?.round??'–'}`;
  }
  let initialized = false;
  window.renderWellbeing = s => {
    document.documentElement.style.visibility = '';
    text('affect-name', s.name);
    text('affect-trend', s.trend);
    if ($('affect-trend')) $('affect-trend').title = s.assessment;
    text('affect-suggestion', s.suggestion);
    text('affect-status', s.source === 'Demo inference' ? 'Demo preview' : s.valence == null ? 'Waiting for a 30s window' : 'State Detected');
    ['valence', 'arousal'].forEach(key => {
      text(key, s[key] ?? '–');
      if ($(key + '-fill')) $(key + '-fill').style.width = `${(s[key] ?? 0) * 20}%`;
    });
    const live = Object.values(s.sensors).some(v => v.length);
    text('sensor-source', live ? 'Live' : s.source === 'Demo inference' ? 'Demo' : 'Waiting');
    document.querySelectorAll('[data-sensor]').forEach(card => {
      const v = s.sensors[card.dataset.sensor] || [];
      const d = curve(v, 100, 30);
      card.querySelector('[data-line]').setAttribute('d', d);
      card.querySelector('[data-area]').setAttribute('d', d ? d + ' L100,30 L0,30 Z' : '');
      card.title = v.length ? `${card.dataset.sensor}: ${v.length} samples from the last window` : 'Waiting for sensor data';
    });
    if ($('runtime-error')) { $('runtime-error').hidden = !s.error; text('runtime-error', s.error || ''); }
    const t = s.training;
    document.documentElement.dataset.trainingPhase=t.phase;
    if ($('training-ppl') && !$('training-ppl').dataset.layoutReady) {
      const number=$('training-ppl'), pair=number.parentElement;
      pair.prepend(pair.lastElementChild);
      pair.style.cssText='display:flex;align-items:baseline;gap:8px;white-space:nowrap';
      pair.firstElementChild.style.fontSize='12px';
      number.style.fontSize='28px';number.style.lineHeight='36px';
      pair.parentElement.parentElement.style.cssText='display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:start;gap:12px;margin-bottom:12px';
      const title=$('training-live').previousElementSibling;
      title.querySelector('span')?.remove();
      title.style.cssText='min-width:0;white-space:nowrap';
      title.querySelector('h3').style.whiteSpace='nowrap';
      number.dataset.layoutReady='true';
    }
    text('training-cutting-layer', `Cutting Layer = ${t.cuttingLayer ?? '–'}`);
    const step = t.total > 0 ? `Step ${t.step.toLocaleString()} / ${t.total.toLocaleString()}` : 'Step 0 / –';
    const percent = t.total > 0 ? Math.max(0, Math.min(100, t.step / t.total * 100)) : 0;
    text('progress-text', `${Math.round(percent)}%`);
    text('training-steps', step); text('training-metric-step', step);
    if ($('training-progress')) {
      const circumference = 2 * Math.PI * 52;
      $('training-progress').style.strokeDasharray = `${circumference} ${circumference}`;
      $('training-progress').style.strokeDashoffset = circumference * (1 - percent / 100);
    }
    text('training-message', t.message || t.phase);
    if ($('training-message')) $('training-message').title = `Round ${t.round} / ${t.totalRounds}`;
    const ppl = t.loss == null ? null : Math.exp(t.loss);
    const history = t.history.filter(p => p.loss != null).map(p => ({...p, ppl: Math.exp(p.loss)})).filter(p => Number.isFinite(p.ppl));
    text('training-ppl', ppl == null ? '–' : Number.isFinite(ppl) ? ppl.toFixed(2) : '∞');
    text('training-change', history.length > 1 ? `${((history.at(-1).ppl / history[0].ppl - 1) * 100).toFixed(1)}% since start` : 'Waiting for training metrics');
    text('training-live', t.running ? 'Live' : ({COMPLETE:'Complete', FAILED:'Failed', UNAVAILABLE:'Unavailable'}[t.phase] || 'Idle'));
    renderTrainingChart(t,history);
    // The native API reports phases, not individual network link health. Never invent a connected server.
    const statuses = t.phase === 'COMPLETE' ? ['Updated','Exchanged','Session complete'] :
      t.phase === 'TRAINING' ? [`Step ${t.step} done`,`Step ${t.step} received`,'Session active'] :
      t.phase === 'WAITING' ? ['Pending','Pending','Waiting'] :
      t.phase === 'FAILED' ? ['Failed','Stopped','Check settings'] :
      t.phase === 'STARTING' ? ['Pending','Pending','Connecting'] :
      t.phase === 'CANCELLING' ? ['Stopping','Stopping','Stopping'] : ['Idle','Idle','Disconnected'];
    ['status-lora','status-gradients','status-server'].forEach((key,i)=>{
      text(key,statuses[i]);
      if ($(key)) {
        const color = t.phase==='FAILED' ? '#ba1a1a' : t.phase==='COMPLETE' ? '#006c49' : '#777587';
        $(key).style.color=color; $(key).previousElementSibling.style.backgroundColor=color;
        $(key).previousElementSibling.classList.toggle('animate-pulse', i < 2 && statuses[i] === 'Pending');
      }
    });
    renderAggregation(s.aggregation);
    ['status-lora','status-gradients','status-server','status-aggregation'].forEach(id=>{
      const label=$(id);if(!label)return;
      const badge=label.parentElement,row=badge.parentElement,identity=row.firstElementChild;
      row.style.cssText='display:grid;grid-template-columns:minmax(0,1fr) 104px;gap:8px;align-items:center;min-height:72px';
      identity.style.cssText='display:flex;align-items:center;gap:10px;min-width:0';
      identity.firstElementChild.style.cssText='width:32px;height:32px;flex:0 0 32px;border-radius:50%;display:flex;align-items:center;justify-content:center';
      identity.lastElementChild.style.fontSize='12px';identity.lastElementChild.style.lineHeight='16px';
      badge.style.cssText='display:grid;grid-template-columns:8px minmax(0,1fr);gap:6px;align-items:center;width:104px;min-height:32px';
      badge.firstElementChild.style.width='8px';badge.firstElementChild.style.height='8px';badge.firstElementChild.style.borderRadius='50%';
      label.style.lineHeight='16px';
    });
    if (!initialized) {
      document.querySelectorAll('input[name]').forEach(el=>el.value=s.config[el.name] || '');
      initialized=true;
    }
    text('inference-message', s.inferenceMessage);
    text('settings-training-message', t.message);
    text('runtime-source', s.source);
    document.querySelectorAll('[data-action="load"]').forEach(el=>el.disabled=!s.inferenceAvailable || s.processing || t.running);
    document.querySelectorAll('[data-action="start"]').forEach(el=>el.disabled=t.phase==='UNAVAILABLE' || t.running || s.processing);
    document.querySelectorAll('[data-action="cancel"]').forEach(el=>el.disabled=!t.running);
    document.querySelectorAll('[data-action="preview"], [data-action="clearPreview"]').forEach(el=>el.disabled=s.processing);
  };
})();
