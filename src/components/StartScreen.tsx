import { useEffect, useRef, useState } from 'react'
import type { WebBot } from '../neural/types'
import { humanSide, presetSide, type SideConfig, type Strength } from '../presets'
import { Icon } from './Icons'
import { PlaySetup, WatchSetup, type Color } from './PresetControls'

type Choice = 'play' | 'watch'

export function StartScreen({ bots, onStart, onResume }: { bots: WebBot[]; onStart: (sides: [SideConfig, SideConfig]) => void; onResume?: () => void }) {
  const [choice, setChoice] = useState<Choice | null>(null)
  const [strength, setStrength] = useState<Strength>('medium')
  const [color, setColor] = useState<Color>('white')
  const [watch, setWatch] = useState<[Strength, Strength]>(['hard', 'medium'])
  const setup = useRef<HTMLDivElement>(null)
  // On narrow screens the setup appears below both cards, out of view.
  useEffect(() => { if (choice) setup.current?.scrollIntoView({ block: 'nearest', behavior: 'smooth' }) }, [choice])

  const start = () => {
    if (choice === 'watch') return onStart([presetSide(watch[0], bots), presetSide(watch[1], bots)])
    const white = color === 'random' ? Math.random() < 0.5 : color === 'white'
    onStart(white ? [humanSide(), presetSide(strength, bots)] : [presetSide(strength, bots), humanSide()])
  }

  const card = (value: Choice, icon: 'pawn' | 'eye', title: string, text: string) => (
    <button type="button" className="choice-card" aria-pressed={choice === value} onClick={() => setChoice(value)}>
      <Icon name={icon} size={28} />
      <strong>{title}</strong>
      <span>{text}</span>
    </button>
  )

  return (
    <section className="start-screen">
      <div className="start-intro">
        <h2>A race across the board.</h2>
        <p>Reach the opposite edge first. Move your pawn or place a wall, leaving a path for both players.</p>
      </div>
      <div className="choice-cards">
        {card('play', 'pawn', 'Play against a bot', 'Pick a strength and a side.')}
        {card('watch', 'eye', 'Watch bots', 'Two bots play each other.')}
      </div>
      {choice && (
        <div ref={setup} className="panel start-setup">
          {choice === 'play'
            ? <PlaySetup strength={strength} color={color} bots={bots} allowRandom onStrength={setStrength} onColor={setColor} />
            : <WatchSetup strengths={watch} bots={bots} onChange={(side, value) => setWatch(w => side ? [w[0], value] : [value, w[1]])} />}
          <button type="button" className="primary start-button" onClick={start}>
            <Icon name="play" /> Start game
          </button>
        </div>
      )}
      {onResume && <button type="button" className="link" onClick={onResume}>Back to the current game</button>}
      <p className="muted small">Every move runs on your device. Neural bots download when first used; their speed depends on your device.</p>
    </section>
  )
}
