import { chromium } from 'playwright';
import assert from 'node:assert/strict';
const baseUrl = process.env.WEB_URL ?? 'http://127.0.0.1:4173/';
const devUrl = process.env.WEB_DEV_URL;
const browser = await chromium.launch({executablePath:process.env.CHROME_PATH || undefined,headless:true});
try {
const page = await browser.newPage();
const errors=[]; const api=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(r.url().includes('/api/'))api.push(r.url());});
if (devUrl) {
await page.goto(new URL('?white=human&black=human', devUrl).href);
const parity = await page.evaluate(async()=>{
 const {loadBrowserModel,evaluateBrowserModel}=await import('/src/neural/runtime.ts');
 const manifest=await (await fetch('/bots.json')).json();
 const results=[];
 for(const bot of manifest.bots){
  const fixture=await(await fetch(`/models/${bot.id}-fixtures.json`)).json();
  const rows=fixture.rows;
  for(const preferGpu of [false,true]){
   const model=await loadBrowserModel(bot,location.origin+'/',()=>{},preferGpu);
   let maxLogit=0,maxValue=0; const start=performance.now();
   for(const row of rows){
    const result=await evaluateBrowserModel(model.session,row.position,row.repetitions,row.remaining);
    for(let i=0;i<209;i++)maxLogit=Math.max(maxLogit,Math.abs(result.logits[i]-row.logits[i]));
    maxValue=Math.max(maxValue,Math.abs(result.value-row.value));
   }
   results.push({bot:bot.id,provider:model.provider,rows:rows.length,maxLogit,maxValue,ms:performance.now()-start});
  }
 }
 return results;
});
console.log('PARITY',JSON.stringify(parity));
for(const r of parity){assert.ok(r.maxLogit<0.002);assert.ok(r.maxValue<0.0002);}
}
await page.goto(new URL('?white=browser:resnet-small&black=browser:resnet-rl&sims=8', baseUrl).href);
await page.waitForFunction(()=>document.querySelectorAll('.history li span').length>=6,{},{timeout:60000});
await page.getByRole('button',{name:'Pause',exact:true}).click();
const moves=await page.locator('.history').innerText();
await page.waitForTimeout(1500);
assert.equal(await page.locator('.history').innerText(),moves);
await page.getByRole('button',{name:'New game',exact:true}).click();
assert.equal(await page.locator('.history li').count(),0);
await page.waitForTimeout(1000);assert.equal(await page.locator('.history li').count(),0);
await page.getByRole('button',{name:'Resume',exact:true}).click();
await page.waitForFunction(()=>document.querySelectorAll('.history li').length>=1,{},{timeout:20000});
assert.equal(api.length,0);assert.deepEqual(errors,[]);
if(process.env.SCREENSHOT_PATH) await page.screenshot({path:process.env.SCREENSHOT_PATH});
console.log('ARENA PASS', {moves, api,errors});
const context = await browser.newContext();
const failure = await context.newPage();
await failure.route('**/*.onnx', route => route.fulfill({status:503,body:'Unavailable'}));
await failure.goto(new URL('?white=browser:resnet-small&black=human&sims=8',baseUrl).href);
await failure.getByRole('alert').waitFor();
assert.match(await failure.getByRole('alert').innerText(),/download failed/i);
await failure.getByText('Advanced bot settings',{exact:true}).click();
await failure.getByLabel('White player').selectOption('engine');
await failure.waitForFunction(()=>document.querySelectorAll('.history li').length>=1,{},{timeout:20000});
console.log('DOWNLOAD FAILURE RECOVERY PASS');
await context.close();
} finally { await browser.close(); }
