import { useEffect, useRef, useState } from 'react'
import { Icon } from './Icons'

function ShareLink({ title, description, url }: { title: string; description: string; url: string }) {
  const input = useRef<HTMLInputElement>(null)
  const [state, setState] = useState<'idle' | 'copied' | 'manual'>('idle')
  useEffect(() => {
    if (state !== 'copied') return
    const timer = window.setTimeout(() => setState('idle'), 2000)
    return () => clearTimeout(timer)
  }, [state])
  const copy = () => {
    // navigator.clipboard is missing outside secure contexts, so failures can be synchronous too.
    Promise.resolve().then(() => navigator.clipboard.writeText(url)).then(() => setState('copied'), () => {
      input.current?.select()
      setState('manual')
    })
  }
  return (
    <div className="share-link">
      <div><strong>{title}</strong><p className="muted small">{description}</p></div>
      <div className="share-row">
        <input ref={input} readOnly value={url} aria-label={title} onFocus={e => e.currentTarget.select()} />
        <button type="button" onClick={copy}><Icon name={state === 'copied' ? 'check' : 'copy'} /> {state === 'copied' ? 'Copied' : 'Copy'}</button>
        {'share' in navigator && (
          <button type="button" onClick={() => { navigator.share({ title: 'Quoridoraner', url }).catch(() => { /* Dismissed by the user. */ }) }}>
            <Icon name="share" /> Share
          </button>
        )}
      </div>
      <span className="muted small" role="status">{state === 'manual' ? 'Copying is blocked here; the link is selected for you to copy.' : ''}</span>
    </div>
  )
}

export function SharePanel({ gameUrl, settingsUrl, moves }: { gameUrl: string; settingsUrl: string; moves: number }) {
  return (
    <section className="panel share-panel" aria-label="Share">
      {moves > 0 && (
        <ShareLink title="Link to this game" url={gameUrl}
          description={`Replays the ${moves} ${moves === 1 ? 'move' : 'moves'} so far, then play continues with the same players.`} />
      )}
      <ShareLink title="Link to these settings" url={settingsUrl} description="Starts a new game with the same players and strengths." />
    </section>
  )
}
