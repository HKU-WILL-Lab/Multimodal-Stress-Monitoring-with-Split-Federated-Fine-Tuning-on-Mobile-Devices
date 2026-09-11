const {chromium}=require('playwright');const assert=require('assert/strict');
const path=require('path');const fs=require('fs');const {pathToFileURL}=require('url');
async function main(){
 fs.mkdirSync(path.resolve(__dirname,'../build'),{recursive:true});
 const browser=await chromium.launch({headless:true,...(process.env.PLAYWRIGHT_CHANNEL?{channel:process.env.PLAYWRIGHT_CHANNEL}:{})});
 const page=await browser.newPage({viewport:{width:360,height:800}});
 await page.goto(pathToFileURL(path.resolve(__dirname,'../phone/src/main/assets/stitch/training.html')).href);
 const state={sensors:{},config:{},aggregation:{phase:'IDLE'},training:{phase:'TRAINING',running:true,round:1,totalRounds:1,total:5,cuttingLayer:1,step:0,history:[]}};
 const render=()=>page.evaluate(s=>window.renderWellbeing(s),state);
 await render();
 assert.equal(await page.locator('#training-cutting-layer').textContent(),'Cutting Layer = 1');
 assert.equal(await page.locator('#status-aggregation').evaluate(el=>el.parentElement.parentElement.firstElementChild.lastElementChild.textContent),'Federated Server');
 assert.deepEqual(await page.locator('#training-x-axis span').allTextContents(),['1','2','3','4','5']);
 state.training.total=0;await render();
 assert.equal(await page.locator('#training-x-axis span').count(),0);
 assert.equal(await page.locator('#training-grid line').count(),1,'Unknown total retains baseline');
 state.training.total=1;state.training.history=[{round:1,step:1,loss:Math.log(10)}];await render();
 assert.deepEqual(await page.locator('#training-x-axis span').allTextContents(),['1']);
 assert.equal(await page.locator('#training-dots circle').getAttribute('cx'),'18');
 assert(!(await page.locator('#training-line').getAttribute('d')).includes('NaN'));
 state.training.history=[];state.training.total=2000;await render();
 assert.deepEqual(await page.locator('#training-x-axis span').allTextContents(),['1','401','801','1200','1600','2000']);
 state.training.history=[{round:1,step:1,loss:Math.log(10)}];await render();
 const first=await page.locator('#training-dots circle').getAttribute('cx');
 state.training.history.push({round:1,step:2,loss:Math.log(25)});await render();
 assert.equal(await page.locator('#training-dots circle').first().getAttribute('cx'),first,'X stays fixed when Y maximum grows');
 assert(Math.abs(Number(await page.locator('#training-svg').getAttribute('data-maximum'))-25)<1e-8);
 assert(Math.abs(Number(await page.locator('#training-dots circle').last().getAttribute('cy'))-10)<1e-8,'New maximum reaches top');
 assert(await page.locator('#training-scroll').evaluate(el=>el.scrollWidth===el.clientWidth),'2000 steps fit full viewport');
 state.training.total=5;state.training.history=[];await render();
 for(let step=1;step<=5;step++){
  state.training.step=step;state.training.loss=Math.log(18-step);state.training.history.push({round:1,step,loss:state.training.loss});
  await render();
  assert.equal(await page.locator('#training-dots circle').count(),step,'Every sample gets a point');
  assert.deepEqual(await page.locator('#training-x-axis span').allTextContents(),['1','2','3','4','5']);
  if(step===1)await page.evaluate(()=>{window.firstDot=document.querySelector('#training-dots circle');window.firstX=firstDot.getAttribute('cx');window.firstY=firstDot.getAttribute('cy');});
  assert(await page.evaluate(()=>firstDot===document.querySelector('#training-dots circle')&&firstDot.getAttribute('cx')===firstX&&firstDot.getAttribute('cy')===firstY),'Previous point stays in place');
  assert(await page.evaluate(()=>[...document.querySelectorAll('#training-dots circle')].every(dot=>{
   const label=document.querySelector(`#training-x-axis [data-step="${dot.dataset.step}"]`);return Math.abs(Number(dot.getAttribute('cx'))-parseFloat(label.style.left))<.01;
  })),'Point directly above its own tick');
 }
 assert(await page.evaluate(()=>new Set([...document.querySelectorAll('#training-x-axis span')].map(el=>getComputedStyle(el).color)).size===1),'All tick labels have the same color');
 for(const phase of ['IDLE','WAITING','AGGREGATING','COMPLETE','OFFLINE','NOT_CONNECTED']){
  state.aggregation={phase,received:0,quorum:1};await render();
  assert((await page.locator('#status-aggregation').textContent()).toLowerCase().startsWith(phase==='WAITING'?'waiting':['OFFLINE','NOT_CONNECTED'].includes(phase)?'disconnected':phase.toLowerCase()));
 }
 for(const phase of ['IDLE','STARTING','WAITING','TRAINING','COMPLETE','FAILED']){
  state.training.phase=phase;await render();
  const pending=['STARTING','WAITING'].includes(phase);
  for(const id of ['status-lora','status-gradients']){
   assert.equal(await page.locator('#'+id).evaluate(el=>el.previousElementSibling.classList.contains('animate-pulse')),pending);
   if(phase==='IDLE'||pending)assert.equal(await page.locator('#'+id).textContent(),pending?'Pending':'Idle');
  }
 }
 assert(!(await page.locator('main').textContent()).includes('(Perplexity)'));
 assert(await page.locator('#training-ppl').evaluate(el=>el.previousElementSibling.textContent==='Current PPL'));
 assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert(await page.evaluate(()=>{
  const labels=['status-lora','status-gradients','status-server','status-aggregation'].map(id=>document.getElementById(id));
  const rects=labels.map(el=>el.parentElement.getBoundingClientRect());
  return rects.every(r=>r.x===rects[0].x&&r.width===104)&&labels.every(el=>{const r=el.parentElement.parentElement.firstElementChild.firstElementChild.getBoundingClientRect();return r.width===32&&r.height===32});
 }),'Status columns align and icons remain circles');
 await page.screenshot({path:path.resolve(__dirname,'../build/training-chart-fixed.png'),fullPage:true});
 await browser.close();console.log('PASS: full planned axis, unknown baseline, incremental dots, adaptive maximum, circular icons and aligned status columns');
}
main().catch(e=>{console.error(e);process.exit(1)});
