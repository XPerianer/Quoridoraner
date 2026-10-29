// JSON-lines adapter; opponent source is loaded unchanged from its pinned checkout.
const fs = require('fs');
const vm = require('vm');
const path = require('path');
const readline = require('readline');
const repo = process.argv[2];
console.log = () => {};
vm.runInThisContext(['game.js', 'ai.js'].map(f => fs.readFileSync(path.join(repo, 'src/js', f), 'utf8')).join('\n'));
let game;
function encode(m) {
  if (m[0]) return m[0][0] * 9 + m[0][1];
  const i = m[1] ? 1 : 2, p = m[i];
  return (i === 1 ? 81 : 145) + p[0] * 8 + p[1];
}
function decode(a) {
  if (a < 81) return [[Math.floor(a / 9), a % 9], null, null];
  const base = a < 145 ? 81 : 145, i = a - base;
  return base === 81 ? [null, [Math.floor(i / 8), i % 8], null] : [null, null, [Math.floor(i / 8), i % 8]];
}
function legal() {
  const out = game.getArrOfValidNextPositionTuples().map(p => p[0] * 9 + p[1]);
  if (game.pawnOfTurn.numberOfLeftWalls > 0) {
    for (const p of game.getArrOfValidNoBlockNextHorizontalWallPositions()) out.push(81 + p[0] * 8 + p[1]);
    for (const p of game.getArrOfValidNoBlockNextVerticalWallPositions()) out.push(145 + p[0] * 8 + p[1]);
  }
  return out.sort((a, b) => a - b);
}
readline.createInterface({input: process.stdin}).on('line', line => {
  try {
    const q = JSON.parse(line); let result;
    if (q.op === 'reset') {
      let seed = q.seed >>> 0;
      Math.random = () => { seed += 0x6D2B79F5; let t = seed; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
      game = new Game(true); result = {ok: true};
    } else if (q.op === 'apply') {
      if (!legal().includes(q.move)) throw new Error('Illegal incoming move: ' + q.move);
      game.doMove(decode(q.move)); result = {ok: true};
    } else if (q.op === 'legal') result = {legal: legal()};
    else if (q.op === 'state') result = {
      pawns: game.board.pawns.map(p => p.position.row * 9 + p.position.col),
      walls: game.board.pawns.map(p => p.numberOfLeftWalls),
      turn: game.turn % 2,
      winner: game.winner === null ? -1 : game.winner.index,
    };
    else if (q.op === 'choose') result = {move: encode(new AI(q.simulations, 0.2).chooseNextMove(game))};
    else throw new Error('Unknown operation');
    process.stdout.write(JSON.stringify(result) + '\n');
  } catch (e) { process.stdout.write(JSON.stringify({error: String(e.stack)}) + '\n'); }
});
