import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import App from './App'
import type { WebBot } from './neural/types'

vi.mock('./components/BoardView', () => ({ BoardView: () => null }))
vi.mock('./components/SearchPanel', () => ({ SearchPanel: () => null }))

const bot: WebBot = {
  id: 'resnet', name: 'ResNet', url: 'models/resnet.onnx',
  sha256: 'a'.repeat(64), bytes: 1000, planes: 19, simulations: 32,
}
afterEach(() => vi.unstubAllGlobals())

describe('public arena configuration', () => {
  it('opens on the start screen without a shared setup', () => {
    vi.stubGlobal('window', { location: { search: '' } })
    const html = renderToStaticMarkup(<App publicArena bots={[bot]}/>)
    expect(html).toContain('Play against a bot')
    expect(html).toContain('Watch bots')
    expect(html).not.toContain('Advanced bot settings')
  })

  it('offers browser bots and omits Python server controllers', () => {
    vi.stubGlobal('window', { location: { search: '?black=browser:resnet' } })
    const html = renderToStaticMarkup(<App publicArena bots={[bot]}/>)
    expect(html).toContain('Advanced bot settings')
    expect(html).toContain('value="browser:resnet" selected=""')
    expect(html).toContain('value="32" selected=""')
    expect(html).not.toContain('Reference negamax')
    expect(html).toContain('aria-label="Share"')
  })

  it('applies a shared preset and describes the matchup', () => {
    vi.stubGlobal('window', { location: { search: '?white=human&black=easy' } })
    const html = renderToStaticMarkup(<App publicArena bots={[bot]}/>)
    expect(html).toContain('You (White) vs Easy bot')
    expect(html).toContain('aria-pressed="true">Easy</button>')
  })

  it('rejects malformed controllers and illegal replay URLs without crashing', () => {
    vi.stubGlobal('window', { location: { search: '?white=%7Bbad&black=browser:missing&moves=999&sims=-1' } })
    const html = renderToStaticMarkup(<App publicArena bots={[bot]}/>)
    expect(html).toContain('value="human" selected=""')
    expect(html).toContain('value="engine" selected=""')
    expect(html).not.toContain('Play replay')
  })

  it('preserves local server controller options in the training arena', () => {
    vi.stubGlobal('window', { location: { search: '' } })
    const html = renderToStaticMarkup(<App/>)
    expect(html).toContain('Reference negamax')
    expect(html).not.toContain('Play against a bot')
  })
})
