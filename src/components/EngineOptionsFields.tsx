import { DEFAULT_OPTIONS } from '../engine/search'
import type { MoveGen, SearchOptions } from '../engine/types'
import { DEPTHS, NODE_LIMITS, THINK_TIMES } from '../presets'

export function EngineOptionsFields({ options, onChange }: { options: SearchOptions; onChange: (options: SearchOptions) => void }) {
  const set = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => onChange({ ...options, [key]: value })
  return (
    <>
      <div className="field-row">
        <label>
          Think time
          <select value={options.timeMs} onChange={(e) => set('timeMs', Number(e.target.value))}>
            {THINK_TIMES.map((t) => <option key={t} value={t}>{t / 1000}s</option>)}
          </select>
        </label>
        <label>
          Max depth
          <select value={options.maxDepth} onChange={(e) => set('maxDepth', Number(e.target.value))}>
            {DEPTHS.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </label>
        <label>
          Walls considered
          <select value={options.moveGen} onChange={(e) => set('moveGen', e.target.value as MoveGen)}>
            <option value="all">All legal</option>
            <option value="likely">Near pawns/walls</option>
          </select>
        </label>
        <label>
          Node limit
          <select value={options.maxNodes} onChange={(e) => set('maxNodes', Number(e.target.value))}>
            {NODE_LIMITS.map((n) => <option key={n} value={n}>{n === DEFAULT_OPTIONS.maxNodes ? 'None' : n.toLocaleString()}</option>)}
          </select>
        </label>
      </div>
      <div className="field-row">
        <label title="Eval points per wall-in-hand advantage; one path step = 10">
          Wall value
          <input type="number" step={1} value={options.wallWeight} onChange={(e) => set('wallWeight', e.target.valueAsNumber || 0)} />
        </label>
        <label title="Eval bonus for the side to move; one path step = 10">
          Tempo
          <input type="number" step={1} value={options.tempo} onChange={(e) => set('tempo', e.target.valueAsNumber || 0)} />
        </label>
        <label className="check">
          <input type="checkbox" checked={options.exactRoot} onChange={(e) => set('exactRoot', e.target.checked)} />
          Exact score for every candidate (slower)
        </label>
        <button className="link" onClick={() => onChange(DEFAULT_OPTIONS)}>Reset</button>
      </div>
    </>
  )
}
