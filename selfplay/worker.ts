import { parentPort } from 'node:worker_threads'
import { playGame, type GameTask } from './game'

parentPort!.on('message', (task: GameTask) => parentPort!.postMessage(playGame(task)))
