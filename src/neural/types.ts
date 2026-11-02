import type { Position } from '../engine/types'

export interface WebBot {
  id: string
  name: string
  url: string
  sha256: string
  bytes: number
  planes: 19
  simulations: number
}

export interface NeuralRequest {
  id: number
  bot: WebBot
  positions: Position[]
  simulations: number
  baseUrl: string
}
export type NeuralResponse =
  | { id: number; type: 'status'; status: string }
  | { id: number; type: 'done'; move: number; simulations: number }
  | { id: number; type: 'error'; error: string }
