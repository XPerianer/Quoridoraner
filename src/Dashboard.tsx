import { useEffect, useState } from 'react'
import { api, type Run } from './dashboard-api'
import App from './App'
import './dashboard.css'

const COLORS = ['#267e87', '#cb7542', '#7970bd', '#a83f64', '#649649', '#be912b']
const pct = (n: number | undefined) => n === undefined ? '—' : `${(n * 100).toFixed(1)}%`
function Curve({ title, series, percent = false, axis }: { title: string; series: { name: string; points: [number, number][] }[]; percent?: boolean; axis: string }) {
  const all = series.flatMap(s => s.points).filter(p => Number.isFinite(p[1]))
  const xmax = Math.max(1, ...all.map(p => p[0]))
  const ymax = percent ? 1 : Math.max(0.01, ...all.map(p => p[1])) * 1.05
  return <section className="panel curve"><h3>{title}</h3>{!all.length ? <p className="muted">No saved observations yet.</p> : <>
    <svg viewBox="0 0 600 230" role="img" aria-label={title}>
      {[0, 0.25, 0.5, 0.75, 1].map(t => <g key={t}><line x1="48" x2="580" y1={190-t*165} y2={190-t*165} stroke="var(--grid)"/><text x="40" y={194-t*165} textAnchor="end">{percent ? pct(t) : (t*ymax).toFixed(2)}</text></g>)}
      {series.map((s, i) => <g key={s.name}><polyline fill="none" stroke={COLORS[i % COLORS.length]} strokeWidth="2.5" points={s.points.filter(p => Number.isFinite(p[1])).map(([x,y]) => `${48+x/xmax*532},${190-y/ymax*165}`).join(' ')}/>{s.points.filter(p => Number.isFinite(p[1])).map(([x,y], j) => <circle key={j} cx={48+x/xmax*532} cy={190-y/ymax*165} r="2.5" fill={COLORS[i % COLORS.length]}><title>{s.name}: {x} → {percent ? pct(y) : y.toFixed(4)}</title></circle>)}</g>)}
      <text x="48" y="212">0</text><text x="580" y="212" textAnchor="end">{xmax}</text><text x="310" y="228" textAnchor="middle">{axis}</text>
    </svg>
    <div className="legend">{series.map((s,i) => <span key={s.name}><i style={{ background: COLORS[i % COLORS.length] }}/>{s.name}</span>)}</div>
  </>}</section>
}
const number = (n: number | undefined | null, digits = 0) => n == null ? '—' : n.toLocaleString(undefined, { maximumFractionDigits: digits })
const duration = (seconds: number) => `${Math.floor(seconds / 3600)}h ${Math.floor(seconds % 3600 / 60)}m ${Math.floor(seconds % 60)}s`
function RunProgress({ run, mode, now }: { run: Run; mode: string; now: number }) {
  const s = run.status
  if (!s) return <section className="panel progress-card"><h3>{run.id}</h3><p className="muted">Live status unavailable. Start or resume this run with the updated trainer to publish task progress.</p></section>
  const stale = s.lifecycle === 'running' && now / 1000 - s.updated > 60
  const label = stale ? 'No recent update' : s.lifecycle
  const loss = s.losses[mode]
  const fraction = s.lifecycle === 'complete' ? 1 : Math.min(1, s.completed / Math.max(1, s.target))
  return <section className="panel progress-card">
    <div className="progress-heading"><h3>{run.id}</h3><span className={`status-badge ${stale ? 'stale' : s.lifecycle}`}>{label}</span></div>
    <h2>{s.phase === 'train' ? 'Generating training games' : s.phase === 'update' ? 'Optimizing the network' : s.phase === 'eval' ? 'Evaluating against negamax' : 'Experiment complete'}</h2>
    <p className="muted">{s.mode === 'selfplay' ? 'Self-play' : 'Negamax arm'} · iteration {s.iteration} / {s.iterations} · {s.device}</p>
    {s.detail && <p className="task-detail">{s.detail}</p>}
    <progress aria-label={`${run.id} current phase progress`} value={fraction} max={1}/>
    <div className="progress-caption"><span>{s.lifecycle === 'complete' ? 'All phases complete' : `${number(s.completed)} / ${number(s.target)} ${s.unit} in this phase`}</span><span>Updated {new Date(s.updated * 1000).toLocaleTimeString()}</span></div>
    {stale && <p className="muted small">The trainer may be busy or stopped. These are the last reported values.</p>}
    <div className="metric-grid">
      {[['Active games', number(s.active_games)], ['Mean active ply', number(s.mean_ply, 1)],
        ['Session moves / sec', number(s.moves_per_second, 1)], ['Session elapsed', duration(s.session_seconds)],
        [`${mode} replay samples`, `${number(s.replay[mode])} / ${number(Number(run.config.replay_size))}`],
        ['Latest policy / value loss', loss ? `${loss.policy.toFixed(3)} / ${loss.value.toFixed(3)}` : 'Awaiting updates']
      ].map(([name, value]) => <div key={name}><span>{name}</span><strong>{value}</strong></div>)}
    </div>
    <p className="small"><strong>Active tasks: </strong>{Object.entries(s.tasks).map(([task, count]) => `${task} (${count})`).join(' · ') || 'No games active'}</p>
    {s.phase === 'eval' && <p className="small"><strong>Evaluation suites: </strong>{Object.entries(s.suites).map(([name, count]) => `${name} (${count})`).join(' · ') || 'Preparing games'}</p>}
    {s.curriculum && <p className="small">Curriculum race-{s.curriculum.stage} · passing streak {s.curriculum.streak} / {String(run.config.curriculum_streak ?? '—')}</p>}
    {s.opponent && <p className="small">Adaptive opponent: {number(s.opponent.nodes)} nodes · promotion streak {s.opponent.streak} / {String(run.config.promotion_streak ?? '—')}</p>}
  </section>
}
export default function Dashboard() {
  const [now, setNow] = useState(() => Date.now())
  const [tab, setTab] = useState(() => new URLSearchParams(window.location.search).get('tab') ?? 'runs')
  const [runs, setRuns] = useState<Run[]>([])
  const [selected, setSelected] = useState<string[]>([])
  const [details, setDetails] = useState<Run[]>([])
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')
  const [mode, setMode] = useState('selfplay')
  const [suite, setSuite] = useState('monitor')
  const [refresh, setRefresh] = useState(0)
  useEffect(() => {
    let active = true
    const update = () => api<Run[]>('/api/runs').then(data => { if (active) { setRuns(data); setNow(Date.now()); setError(''); setSelected(ids => ids.length ? ids : data.slice(0,1).map(r => r.id)) } }).catch(e => { if (active) setError(`${e.message}. Start with npm run dashboard to connect training runs.`) })
    void update(); const timer = setInterval(update, 5000)
    return () => { active = false; clearInterval(timer) }
  }, [refresh])
  useEffect(() => {
    let active = true
    void Promise.all(selected.map(id => api<Run>(`/api/run?id=${encodeURIComponent(id)}`))).then(data => { if (active) setDetails(data) }).catch(e => { if (active) setError(e.message) })
    return () => { active = false }
  }, [selected, runs])
  const rows = (r: Run) => r.summary.filter(s => s.mode === mode && s.phase === 'eval' && (s.suite || 'monitor') === suite).sort((a,b) => a.iteration-b.iteration)
  return <div className="dashboard">
    <header className="dash-head"><div><div className="eyebrow">QUORIDORANER / TRAINING LAB</div><h1>From training to the board.</h1><p className="muted">Compare experiments. Inspect learning. Put your bots to the test.</p></div><div className="live-label">Local workspace · refreshes every 5s</div></header>
    <nav className="dash-tabs"><button className={tab === 'runs' ? 'selected' : ''} onClick={() => setTab('runs')}>Training runs <span>{runs.length}</span></button><button className={tab === 'arena' ? 'selected' : ''} onClick={() => setTab('arena')}>Play & watch</button></nav>
    {error && <div className="dash-error" role="alert">{error} <button onClick={() => setRefresh(n => n+1)}>Retry</button></div>}
    <div hidden={tab !== 'runs'}>
      <div className="dash-controls"><input aria-label="Filter runs" placeholder="Find a training run…" value={filter} onChange={e => setFilter(e.target.value)}/><label>Model<select value={mode} onChange={e => setMode(e.target.value)}><option value="selfplay">Self-play</option><option value="negamax">Trained vs negamax</option></select></label><label>Evaluation suite<select value={suite} onChange={e => setSuite(e.target.value)}><option value="monitor">Monitoring</option><option value="heldout">Held-out</option></select></label><button onClick={() => setRefresh(n => n+1)}>Refresh</button></div>
      <div className="run-table panel"><table><thead><tr><th>Compare / run</th><th>Current task</th><th>Iteration</th><th>Eval score</th><th>W / C / L</th><th>Training goal / repetition</th><th>Setup</th><th>Saved</th></tr></thead><tbody>{runs.filter(r => r.id.toLowerCase().includes(filter.toLowerCase())).map(r => {
        const last = rows(r).at(-1); const train = r.summary.filter(s => s.mode === mode && s.phase === 'train').sort((a,b) => a.iteration-b.iteration).at(-1)
        return <tr key={r.id} className={selected.includes(r.id) ? 'chosen' : ''}><td><label><input type="checkbox" checked={selected.includes(r.id)} onChange={e => setSelected(ids => e.target.checked ? [...ids,r.id] : ids.filter(id => id !== r.id))}/><strong>{r.id}</strong></label><small>{r.bots.length ? r.bots.join(' · ') : 'No bot exports yet'}</small></td><td>{r.status ? `${r.status.mode} · ${r.status.phase}` : 'Status unavailable'}<small>{r.status ? `${r.status.lifecycle === 'running' && now / 1000 - r.status.updated > 60 ? 'No recent update' : r.status.lifecycle} · ${r.status.completed}/${r.status.target} ${r.status.unit}` : 'Saved results only'}</small></td><td>{Math.max(r.status?.iteration ?? 0,...r.summary.map(s => s.iteration))} / {String(r.config.iterations ?? '—')}</td><td><strong>{pct(last?.score)}</strong><small>{last ? `${last.games} games · iteration ${last.iteration}` : 'Awaiting evaluation'}</small></td><td>{last ? `${last.wins} / ${last.cutoffs} / ${last.losses}` : '—'}</td><td>{pct(train?.goal_rate)} / {pct(train?.repetition_rate)}</td><td>{String(r.config.device ?? 'auto')} · {String(r.config.engine ?? 'typescript')}<small>{String(r.config.simulations ?? '—')} sims · {String(r.config.walls ?? 10)} walls</small></td><td>{new Date(r.updated*1000).toLocaleString()}</td></tr>
      })}</tbody></table>{!runs.length && <p className="empty">Training runs will appear here when config.json is saved under experiments/training.</p>}</div>
      <p className="muted small">Score = (wins + ½ artificial cutoffs) / games against fixed negamax. C = repetition or move-limit cutoff. Compare runs with matching wall and search budgets; held-out results are shown separately.</p>
      <div className="dash-charts">{details.map(r => <RunProgress key={r.id} run={r} mode={mode} now={now}/>)}</div>
      {!selected.length ? <p className="empty">Select runs above to compare their learning curves.</p> : <div className="dash-charts">
        <Curve title="Strength against fixed negamax" axis="Training iteration" percent series={details.map(r => ({name:r.id, points:rows(r).map(s => [s.iteration,s.score])}))}/>
        <Curve title="Training games reaching a goal" axis="Training iteration" percent series={details.map(r => ({name:r.id,points:r.summary.filter(s => s.mode === mode && s.phase === 'train' && s.goal_rate !== undefined).map(s => [s.iteration,s.goal_rate!]).sort((a,b)=>a[0]-b[0]) as [number,number][]}))}/>
        <Curve title="Average training game length" axis="Training iteration (plies per game)" series={details.map(r => ({name:r.id, points:r.summary.filter(s => s.mode === mode && s.phase === 'train' && s.mean_plies !== undefined).sort((a,b)=>a.iteration-b.iteration).map(s => [s.iteration,s.mean_plies!])}))}/>
        <Curve title="Training repetition cutoffs" axis="Training iteration" percent series={details.map(r => ({name:r.id, points:r.summary.filter(s => s.mode === mode && s.phase === 'train' && s.repetition_rate !== undefined).sort((a,b)=>a.iteration-b.iteration).map(s => [s.iteration,s.repetition_rate!])}))}/>
        {(['policy','value'] as const).map(metric => <Curve key={metric} title={`${metric === 'policy' ? 'Policy' : 'Value'} loss`} axis="Cumulative optimizer update" series={details.map(r => ({name:r.id, points:(r.losses || []).filter(s => s.mode === mode).map((s,i) => [i+1,s[metric]])}))}/>)}
      </div>}
      <div className="dash-charts">{details.map(r => <section className="panel run-detail" key={r.id}><h3>{r.id}</h3><p className="muted">Saved configuration & performance</p><dl>{Object.entries({...r.config,...Object.fromEntries(Object.entries(r.performance).map(([k,v]) => [`measured ${k}`,v]))}).map(([k,v]) => <div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{typeof v === 'number' ? Number(v.toFixed(3)).toLocaleString() : String(v)}</dd></div>)}</dl></section>)}</div>
    </div>
    <div hidden={tab !== 'arena'}><App runs={runs}/></div>
  </div>
}
