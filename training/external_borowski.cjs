// Run Borowski's unchanged decision/path code with a minimal read-only DOM facade.
// Only UI output functions are replaced with move capture; no search fallback.
const fs = require('fs'), path = require('path'), vm = require('vm');
const readline = require('readline');
const repo = process.argv[2];
let state, values, picked, rawHorizontal, rawVertical;
function edge(r,c,vertical) { const a=r*9+c; return `${a}-${a+(vertical?1:9)}`; }
function fits(r,c,v) {
  const own=v?state.v:state.h, other=v?state.h:state.v, i=r*8+c;
  return !own.includes(i) && !other.includes(i)
    && !(v ? (r>0&&own.includes(i-8)) || (r<7&&own.includes(i+8))
           : (c>0&&own.includes(i-1)) || (c<7&&own.includes(i+1)));
}
function setup(s) {
  state=s; picked=undefined;
  const available=[];
  // Match the page's row-major, vertical-then-horizontal textarea ordering.
  for(let r=0;r<8;r++) {
    for(let c=0;c<8;c++) available.push(edge(r,c,true));
    for(let c=0;c<8;c++) available.push(edge(r,c,false));
  }
  values={'#walls':'','#nowalls':available.join(','),'#possibleWalls':''};
  const inventory=state.w1;
  for(const a of s.wallHistory) {
    const v=a>=145,i=a-(v?145:81);
    (v?rawVertical:rawHorizontal)(edge(Math.floor(i/8),i%8,v));
  }
  state.w1=inventory;
}
function dollar(selector) {
  const match=selector.match(/data-pos=['"]?([^\]'" ]+)/);
  let loc=match?match[1]:null, steps=0;
  const obj={
    parent(){return this;},children(){return this;},next(){steps++;return this;},
    addClass(){return this;},removeClass(){return this;},
    val(value){if(!(selector in values))throw Error('Unknown value '+selector);if(value!==undefined){values[selector]=String(value);return this;}return values[selector];},
    text(value){const field=selector==='#showPlayerPieces'?'w0':selector==='#showCompPieces'?'w1':null;if(!field)throw Error('Unknown text '+selector);if(value!==undefined){state[field]=Number(value);return this;}return String(state[field]);},
    attr(name){
      if(selector==='.piecePlayer')return String(state.p0);
      if(selector==='.pieceComputer')return String(state.p1);
      // jQuery attr() on an empty selection returns undefined (the original
      // candidate arrays can contain empty tokens after UI wall updates).
      if(loc===null && selector.includes('td[data-pos='))return undefined;
      if(loc===null)throw Error('Unknown attr '+selector);
      if(name==='class') { const [a,b]=loc.split('-').map(Number);return b-a===9?'wallPlacementHoriz':'wallPlacementVert'; }
      if(name==='data-pos') { if(!steps)return loc;const [a,b]=loc.split('-').map(Number);return `${a+Math.floor(steps/2)}-${b+Math.floor(steps/2)}`; }
      throw Error('Unknown attr name '+name);
    }
  };return obj;
}
const context={console:{log(){}},$:dollar,Math:Object.create(Math)};
vm.createContext(context);
for(const file of ['js/path/graph.js','js/path/astar.js','js/board.js','js/ai.js','js/placewall.js'])
  vm.runInContext(fs.readFileSync(path.join(repo,file),'utf8'),context,{filename:file});
rawHorizontal=context.placeHorizWall;rawVertical=context.placeVertWall;
function capture(a){if(picked!==undefined)throw Error('Multiple selected moves');picked=a;}
context.movePiece=(from,to)=>{if(!Number.isInteger(Number(to)))throw Error('Invalid pawn destination '+to);capture(Number(to));};
function wall(str,v){const a=Number(str.split('-')[0]);capture((v?145:81)+Math.floor(a/9)*8+a%9);}
context.placeHorizWall=s=>wall(s,false);
context.placeVertWall=s=>wall(s,true);
readline.createInterface({input:process.stdin}).on('line',line=>{
  try {
    const q=JSON.parse(line);let result;
    if(q.op==='reset') {
      let seed=q.seed>>>0;
      context.Math.random=()=>{seed+=0x6D2B79F5;let t=seed;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};
      result={ok:true};
    } else {
      setup(q.state);
      if(q.op==='choose') {
        vm.runInContext('computerMove()',context,{timeout:120000});
        if(picked===undefined)throw Error('Bot selected no move');
        result={move:picked};
      } else if(q.op==='legal') {
        const out=Array.from(context.possibleMoves(state.p1,true,state.p0,null)[1]);
        if(state.w1>0)for(const token of values['#nowalls'].split(',').filter(Boolean)) {
          const [a,b]=token.split('-').map(Number), v=b-a===1;
          const second=`${a+(v?9:1)}-${b+(v?9:1)}`;
          if(context.pathToEndExists(token+','+second,null,null))out.push((v?145:81)+Math.floor(a/9)*8+a%9);
        }
        result={legal:out};
      } else throw Error('Unknown operation');
    }
    process.stdout.write(JSON.stringify(result)+'\n');
  }catch(e){process.stdout.write(JSON.stringify({error:String(e.stack)})+'\n');}
});
