'use strict';
const role=document.body.dataset.role, $=id=>document.getElementById(id);
let state,token='',paused=false,autoScroll=true,logFilter='all',clearAt=0,busy=false;
const set=(id,value)=>{if($(id))$(id).textContent=String(value);};
const error=message=>{set('app-error',message);$('app-error').hidden=!message;};
const number=n=>Number.isFinite(n)?n.toLocaleString('en-US',{maximumFractionDigits:2}):'—';
const ppl=loss=>Math.exp(Number(loss));
async function command(action){
 try{busy=true;render();const response=await fetch('/api/'+action,{method:'POST',headers:{'X-App-Token':token}});const result=await response.json();if(!response.ok)throw Error(result.error);error('');if(action==='quit'){document.body.innerHTML='<p style="padding:40px">MobiWellbeing Server stopped. You can close this window.</p>';return;}await poll(false);}
 catch(e){error(e.message);}finally{busy=false;if($('start-service'))render();}
}
$('start-service').onclick=()=>command(role+'/start');
$('stop-service').onclick=()=>{if(confirm('Stop this server? Any training using it will be interrupted.'))command(role+'/stop');};
$('quit-app').onclick=()=>{if(confirm('Exit the application and stop both local servers?'))command('quit');};
function chart(records,total){
 const clients=[...new Set(records.map(r=>r.client_id))],select=$('chart-client'),prior=select.value;
 select.hidden=clients.length<2;select.parentElement.style.display=clients.length<2?'none':'flex';
 if([...select.options].map(o=>o.value).join('|')!==clients.join('|')){select.replaceChildren(...clients.map(c=>new Option(c,c)));if(clients.includes(prior))select.value=prior;}
 const selected=records.filter(r=>r.client_id===select.value).map(r=>({...r,step:(r.global_round-1)*state.config.training.local_steps+r.local_step+1,value:ppl(r.loss)})).filter(r=>Number.isFinite(r.value));
 const last=selected.at(-1),points=selected,max=points.reduce((high,p)=>Math.max(high,p.value),0);
 const svg=$('ppl-chart'),ns='http://www.w3.org/2000/svg';
 const width=svg.getBoundingClientRect().width,height=270,left=60,right=48,top=12,bottom=250;
 if(width<=left+right)return;
 const x=step=>left+(step-1)/Math.max(1,total-1)*(width-left-right);
 const y=value=>bottom-value/(max||1)*(bottom-top);
 svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.dataset.maximum=max;svg.replaceChildren();
 const el=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e;};
 const defs=el('defs',{}),gradient=el('linearGradient',{id:'ppl-fill',x1:0,y1:0,x2:0,y2:1});
 gradient.append(el('stop',{offset:'0%','stop-color':'#4f46e5','stop-opacity':'.2'}),el('stop',{offset:'100%','stop-color':'#4f46e5','stop-opacity':0}));defs.append(gradient);svg.append(defs);
 [0,.5,1].forEach(f=>{const yy=top+(bottom-top)*f;svg.append(el('line',{x1:left,x2:width-right,y1:yy,y2:yy,stroke:'#d3e4fe','stroke-dasharray':f===1?'0':'4 4'}),el('text',{x:left-12,y:yy+4,'text-anchor':'end',fill:'#777587','font-size':11},points.length?number(max*(1-f)):'—'));});
 const path=points.map((p,i)=>(i?'L':'M')+x(p.step)+','+y(p.value)).join(' ');
 if(points.length>1)svg.append(el('path',{id:'ppl-area',d:`${path} L${x(last.step)},${bottom} L${x(points[0].step)},${bottom} Z`,fill:'url(#ppl-fill)'}));
 svg.append(el('path',{d:path,fill:'none',stroke:'#4f46e5','stroke-width':2.5,'stroke-linecap':'round','stroke-linejoin':'round'}));
 points.forEach(p=>{const dot=el('circle',{cx:x(p.step),cy:y(p.value),r:4,'fill':'#ffffff',stroke:'#4f46e5','stroke-width':2,'data-step':p.step});dot.append(el('title',{},`Step ${p.step}: PPL ${number(p.value)}`));svg.append(dot);});
 const ticks=[...new Set(Array.from({length:Math.min(6,total)},(_,i)=>1+Math.round((total-1)*i/Math.max(1,Math.min(5,total-1)))))];
 $('x-axis').replaceChildren(...ticks.map(n=>{const e=document.createElement('span');e.textContent='Step '+n;e.style.left=x(n)+'px';return e;}));
 set('step',last?.step||0);set('total','/ '+total);set('ppl',last?number(last.value):'—');
 const pct=(last?.step||0)/total*100;set('progress',number(pct)+'% progress');$('step-bar').style.width=Math.min(100,pct)+'%';
 set('ppl-delta',last?'Loss: '+Number(last.loss).toFixed(6):'Waiting for first step');
 set('chart-state',last?.step===total?'Steps complete':last?'Training':'Waiting');
}
function cleanServiceLogs(lines){
 let result=[],loading='';
 for(const raw of lines){const line=raw.replace(/\x1b\[[0-9;]*[A-Za-z]/g,'').trim();if(!line)continue;
  if(line.includes('Loading weights:')){loading=/100%/.test(line)?'Model weights loaded.':'Loading model weights…';continue;}
  result.push(line);
 }
 return loading?[loading,...result]:result;
}
function renderMainLogs(lines){
 const area=$('logTerminal'),content=lines.join('\n')||'Waiting for service events';
 if(area.dataset.content!==content){const previous=area.scrollTop;area.dataset.content=content;
  area.replaceChildren(...(lines.length?lines:['Waiting for service events']).map(line=>{const row=document.createElement('div');row.className='log-row';
   const tag=document.createElement('span');tag.className='log-tag';
   const match=line.match(/^\[([^\]]+)\]\s*(.*)$/);tag.textContent=match?match[1]:/error|exception|traceback/i.test(line)?'ERROR':'SYSTEM';
   if(tag.textContent==='ERROR')tag.classList.add('log-error');
   const body=document.createElement('span');body.className='log-body';body.textContent=match?match[2]:line;row.append(tag,body);return row;}));
  if(!autoScroll)area.scrollTop=previous;
 }
 if(autoScroll)area.scrollTop=area.scrollHeight;
}
function render(){
 if(!state)return;
 const service=state.services[role],cfg=state.config,t=cfg.training,fed=state.aggregation;
 renderDeployment();
 set('node-state',(role==='main'?'Main':'Federated')+' Server · '+service.state);
 set('run-name','Run: '+state.runId+' · Cutting Layer = '+t.cut_layer);
 set('footer-state','Main: '+state.services.main.state+' · Federated: '+state.services.federated.state);
 $('start-service').disabled=busy||!token||['Ready','Starting'].includes(service.state);$('stop-service').disabled=busy||!['Ready','Starting'].includes(service.state);
 let logs=service.logs.slice();
 if(role==='main'){
  const records=state.metrics,ids=new Set(records.map(r=>r.client_id));
  set('clients',ids.size);set('client-summary',records.length?records.length+' completed training RPCs':'Waiting for mobile training');
  set('model','Llama 3.2 1B · Cutting Layer = '+t.cut_layer);set('batch',t.batch_size);set('lr',cfg.model.learning_rate);set('tokens',t.sequence_length+' tok');set('device','Device: '+cfg.model.device);set('rpc-bind','gRPC: '+service.bind);
  chart(records,t.local_steps*t.total_rounds);
  logs=cleanServiceLogs(logs);
  const steps=records.map(r=>`[Round ${r.global_round} · Step ${r.local_step+1}] ${r.client_id} | Loss ${r.loss.toFixed(6)} | PPL ${number(ppl(r.loss))} | ${r.duration_seconds.toFixed(2)}s`);
  logs=logFilter==='all'?logs.concat(steps):logFilter==='steps'?steps:records.map(r=>`Step ${r.local_step+1} | Loss ${r.loss.toFixed(6)}`);
 }else{
  const clients=fed.clients||[],received=fed.received||0,round=fed.round||1,done=fed.phase==='COMPLETE'?round:round-1;
  set('clients',clients.length);set('uploaded',received+' uploaded');set('standby',Math.max(0,t.submission_quorum-received)+' needed for quorum');set('round',round);set('round-total','/ '+t.total_rounds);set('round-progress',number(done/t.total_rounds*100)+'% complete');set('phase',fed.phase);set('batch-round','Round: '+round);$('round-ring').setAttribute('stroke-dasharray',(done/t.total_rounds*100)+', 100');
  const active=['IDLE','WAITING'].includes(fed.phase)?0:fed.phase==='AGGREGATING'?1:fed.phase==='COMPLETE'?2:-1;
  [0,1,2].forEach(i=>$('stage-'+i).classList.toggle('active',i===active));set('stage-detail-0',received+' / '+t.submission_quorum+' uploads');set('stage-detail-1',fed.phase==='AGGREGATING'?'Computing weighted average':fed.phase==='COMPLETE'?'Aggregation complete':'Waiting');set('stage-detail-2',fed.phase==='COMPLETE'?'Checkpoint saved; weights available to clients':'Waiting for aggregation');
  const body=document.querySelector('tbody');body.replaceChildren(...clients.map(c=>{const tr=document.createElement('tr');[c.id,new Date(c.lastSeen*1000).toLocaleTimeString(),c.sequences??'—',c.bytes?number(c.bytes/1048576)+' MiB':'—',c.round,c.state].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});return tr;}));
  if(!clients.length){const row=document.createElement('tr'),cell=document.createElement('td');cell.colSpan=6;cell.textContent='Waiting for mobile clients';row.append(cell);body.append(row);}
  set('table-count',clients.length+' clients registered in this run');
  logs=logs.concat((fed.events||[]).map(e=>`[${new Date(e.updatedAt*1000).toLocaleTimeString()}] Round ${e.round} · ${e.phase} · ${e.received}/${e.quorum} uploads`));
 }
 if(role==='main'){renderMainLogs(logs);}else if(!paused){const area=$('logStreamContainer');area.textContent=logs.slice(clearAt).join('\n')||'Waiting for service events';area.scrollTop=area.scrollHeight;}
 if(service.state==='Failed')error('Server exited. See the service log below.');
}
async function poll(repeat=true){try{const response=await fetch('/api/state');if(!response.ok)throw Error('Application status unavailable');state=await response.json();token=state.token;render();}catch(e){error(e.message);}finally{if(repeat&&$('start-service'))setTimeout(poll,1000);}}
if($('chart-client'))$('chart-client').onchange=render;
document.querySelectorAll('[data-log-filter]').forEach(b=>b.onclick=()=>{logFilter=b.dataset.logFilter;clearAt=0;render();});
if($('autoScrollBtn'))$('autoScrollBtn').onclick=()=>{autoScroll=!autoScroll;$('autoScrollBtn').textContent=autoScroll?'Auto-Scroll: ON':'Auto-Scroll: OFF';render();};
if($('pauseLogBtn'))$('pauseLogBtn').onclick=()=>{paused=!paused;$('pauseLogBtn').firstElementChild.textContent=paused?'play_arrow':'pause';render();};
if($('clearLogBtn'))$('clearLogBtn').onclick=()=>{clearAt=state.services[role].logs.length+(state.aggregation.events||[]).length;render();};
if($('refreshClientsBtn'))$('refreshClientsBtn').onclick=()=>poll(false);
$('downloadLogBtn').onclick=()=>{const content=role==='main'?$('logTerminal').dataset.content:$('logStreamContainer').textContent,url=URL.createObjectURL(new Blob([content],{type:'text/plain'})),a=document.createElement('a');a.href=url;a.download=state.runId+'-'+role+'.log';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
if($('ppl-chart'))new ResizeObserver(()=>{if(state)chart(state.metrics,state.config.training.local_steps*state.config.training.total_rounds);}).observe($('ppl-chart'));
poll();

async function refreshDevices(){if(!$('deploy-device'))return;try{const response=await fetch('/api/devices'),devices=await response.json();if(!response.ok)throw Error(devices.error);const old=$('deploy-device').value;$('deploy-device').replaceChildren(...devices.map(d=>new Option(d,d)));if(devices.includes(old))$('deploy-device').value=old;}catch(e){error(e.message);}}
function renderDeployment(){if(!$('deploy-phone'))return;const d=state.deployment||{};if($('deploy-cut').dataset.run!==state.runId){$('deploy-cut').value=String(state.config.training.cut_layer);$('deploy-steps').value=String(state.config.training.local_steps);$('deploy-cut').dataset.run=state.runId;}set('deployment-message',d.available?d.message||($('deploy-device').options.length?'Choose an authorized ADB phone, then build and deploy.':'No authorized phone. Connect USB debugging or reconnect wireless ADB, then Refresh Phones.'):'Android deployment toolchain is not configured.');set('deployment-logs',(d.logs||[]).join('\n'));$('deploy-phone').disabled=!d.available||!$('deploy-device').value||d.phase==='Working';$('train-phone').disabled=!d.prepared||d.phase==='Working';for(const id of ['deploy-cut','deploy-steps','deploy-device','refresh-devices'])$(id).disabled=d.phase==='Working';}
async function deploymentAction(start){try{const body={cut:Number($('deploy-cut').value),steps:Number($('deploy-steps').value),serial:$('deploy-device').value};const response=await fetch(start?'/api/train':'/api/deploy',{method:'POST',headers:{'X-App-Token':token,'Content-Type':'application/json'},body:JSON.stringify(body)});const result=await response.json();if(!response.ok)throw Error(result.error);error('');await poll(false);}catch(e){error(e.message);}}
if($('deploy-phone')){$('refresh-devices').onclick=refreshDevices;$('deploy-phone').onclick=()=>{if(confirm('Build and install the debug APK, replace the selected phone model assets, and prepare a new run? Existing local servers will stop when preparation finishes.'))deploymentAction(false);};$('train-phone').onclick=()=>deploymentAction(true);refreshDevices();}
