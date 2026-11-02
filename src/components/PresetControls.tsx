import type { WebBot } from '../neural/types'
import { describeSide, presetSide, STRENGTH_NAMES, STRENGTHS, type Strength } from '../presets'

export type Color = 'white' | 'black' | 'random'

export function Segmented<T extends string>({ label, value, options, onChange }: {
  label: string
  value: T | null
  options: readonly { value: T; label: string }[]
  onChange: (value: T) => void
}) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map(option => (
        <button key={option.value} type="button" aria-pressed={value === option.value} onClick={() => onChange(option.value)}>
          {option.label}
        </button>
      ))}
    </div>
  )
}

const STRENGTH_OPTIONS = STRENGTHS.map(value => ({ value, label: STRENGTH_NAMES[value] }))

function StrengthField({ label, value, bots, onChange }: { label: string; value: Strength | null; bots: WebBot[]; onChange: (value: Strength) => void }) {
  return (
    <div className="preset-field">
      <span className="preset-label">{label}</span>
      <Segmented label={label} value={value} options={STRENGTH_OPTIONS} onChange={onChange} />
      <span className="muted small">{value ? describeSide(presetSide(value, bots), bots) : 'Custom settings: see Advanced bot settings'}</span>
    </div>
  )
}

export function PlaySetup({ strength, color, bots, allowRandom = false, onStrength, onColor }: {
  strength: Strength | null
  color: Color
  bots: WebBot[]
  allowRandom?: boolean
  onStrength: (value: Strength) => void
  onColor: (value: Color) => void
}) {
  const colors: { value: Color; label: string }[] = [{ value: 'white', label: 'White' }, { value: 'black', label: 'Black' }]
  if (allowRandom) colors.push({ value: 'random', label: 'Random' })
  return (
    <div className="preset-fields">
      <StrengthField label="Bot strength" value={strength} bots={bots} onChange={onStrength} />
      <div className="preset-field">
        <span className="preset-label">You play</span>
        <Segmented label="You play" value={color} options={colors} onChange={onColor} />
        <span className="muted small">White moves first.</span>
      </div>
    </div>
  )
}

export function WatchSetup({ strengths, bots, onChange }: { strengths: [Strength | null, Strength | null]; bots: WebBot[]; onChange: (side: 0 | 1, value: Strength) => void }) {
  return (
    <div className="preset-fields">
      <StrengthField label="White bot" value={strengths[0]} bots={bots} onChange={value => onChange(0, value)} />
      <StrengthField label="Black bot" value={strengths[1]} bots={bots} onChange={value => onChange(1, value)} />
    </div>
  )
}
