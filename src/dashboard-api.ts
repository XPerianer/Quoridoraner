export interface TrainingStatus {
  updated: number; lifecycle: string; mode: string; phase: string; detail: string
  iteration: number; iterations: number; completed: number; target: number; unit: string
  device: string; session_seconds: number; moves: number; moves_per_second: number
  active_games: number; mean_ply: number | null; tasks: Record<string, number>; suites: Record<string, number>
  replay: Record<string, number>; losses: Record<string, { policy: number; value: number } | null>
  curriculum?: { stage: number; streak: number } | null
  opponent?: { nodes: number; streak: number } | null
}
export interface Summary {
  mode: string; phase: string; iteration: number; suite?: string; games: number
  wins: number; losses: number; cutoffs: number; score: number; mean_plies?: number; goal_rate?: number; repetition_rate?: number
}
export interface Run {
  status?: TrainingStatus | null; id: string; updated: number; bots: string[]; config: Record<string, unknown>
  summary: Summary[]; performance: Record<string, number>
  losses?: { mode: string; iteration: number; update: number; policy: number; value: number }[]
}
export async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) { const error = await response.json(); throw new Error(error.error || 'Request failed') }
  return response.json()
}
