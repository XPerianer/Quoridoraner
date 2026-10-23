import { interpolateLab } from 'd3-interpolate'
import { useSyncExternalStore } from 'react'

const query = window.matchMedia('(prefers-color-scheme: dark)')
const subscribe = (cb: () => void) => {
  query.addEventListener('change', cb)
  return () => query.removeEventListener('change', cb)
}

export function useDark(): boolean {
  return useSyncExternalStore(subscribe, () => query.matches)
}

const PALETTE = {
  light: {
    positive: '#2a78d6',
    negative: '#e34948',
    neutral: '#f0efec',
    seqLow: '#cde2fb',
    seqHigh: '#0d366b',
  },
  dark: {
    positive: '#3987e5',
    negative: '#e66767',
    neutral: '#383835',
    seqLow: '#184f95',
    seqHigh: '#cde2fb',
  },
}

export interface Scales {
  /** t in [-1, 1]: negative = bad for the side to move, positive = good. */
  diverging: (t: number) => string
  /** t in [0, 1] */
  sequential: (t: number) => string
}

export function scales(dark: boolean): Scales {
  const p = dark ? PALETTE.dark : PALETTE.light
  const pos = interpolateLab(p.neutral, p.positive)
  const neg = interpolateLab(p.neutral, p.negative)
  const seq = interpolateLab(p.seqLow, p.seqHigh)
  return {
    diverging: (t) => (t >= 0 ? pos(Math.min(1, t)) : neg(Math.min(1, -t))),
    sequential: (t) => seq(Math.max(0, Math.min(1, t))),
  }
}
