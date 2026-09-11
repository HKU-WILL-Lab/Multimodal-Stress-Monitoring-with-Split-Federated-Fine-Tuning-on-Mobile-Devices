// Debug-build automation, reachable only through an explicitly authorized ADB device.
const fs=require('fs');
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function main(){
 const [port,mode,file]=process.argv.slice(2);
 let target;
 for(let i=0;i<30;i++){
  const targets=await(await fetch(`http://127.0.0.1:${Number(port)}/json/list`)).json();
  target=targets.find(t=>t.url.includes('appassets.androidplatform.net'));
  if(target)break;await delay(300);
 }
 if(!target)throw Error('Phone debug WebView is unavailable; unlock and open the app.');
 const ws=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true});});
 let id=0;const pending=new Map();
 ws.addEventListener('message',e=>{const m=JSON.parse(e.data);if(pending.has(m.id)){pending.get(m.id)(m);pending.delete(m.id);}});
 const evaluate=expression=>new Promise((resolve,reject)=>{const n=++id,timer=setTimeout(()=>{pending.delete(n);reject(Error('Phone response timed out'));},15000);pending.set(n,m=>{clearTimeout(timer);if(m.error||m.result.exceptionDetails)reject(Error(JSON.stringify(m)));else resolve(m.result.result.value);});ws.send(JSON.stringify({id:n,method:'Runtime.evaluate',params:{expression,returnByValue:true}}));});
 try{
  await evaluate("location.href='training.html'");await delay(800);
  const phase=await evaluate('document.documentElement.dataset.trainingPhase');
  if(!['UNAVAILABLE','IDLE','COMPLETE','FAILED','STARTING','TRAINING','WAITING','CANCELLING'].includes(phase))throw Error('Cannot confirm that the phone is idle. Open its Training page and try again.');
  if(['STARTING','TRAINING','WAITING','CANCELLING'].includes(phase))throw Error('Phone is already training. Finish or cancel that session first.');
  if(mode==='probe'){console.log('Phone is idle');return;}
  const config=JSON.parse(fs.readFileSync(file,'utf8'));
  await evaluate(`window.Wellbeing.action('save',${JSON.stringify(JSON.stringify(config))})`);
  await delay(300);
  await evaluate(`location.href=${JSON.stringify(mode==='start'?'training.html':'settings.html')}`);await delay(700);
  if(mode==='start'){
   await evaluate(`window.Wellbeing.action('start',${JSON.stringify(JSON.stringify(config))})`);
   await delay(800);
   const result=await evaluate('({phase:document.documentElement.dataset.trainingPhase,error:document.getElementById("runtime-error")?.textContent})');
   if(result.error||!['STARTING','TRAINING','WAITING'].includes(result.phase))throw Error('Phone rejected training: '+JSON.stringify(result));
   console.log('Phone acknowledged training start');
  }else console.log('Phone configuration applied');
 }finally{ws.close();}
}
main().catch(e=>{console.error(e.message);process.exit(1)});
