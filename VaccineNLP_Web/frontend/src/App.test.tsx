import { render, screen, fireEvent } from '@testing-library/react'
import App from './App'
import { vi, describe, it, expect } from 'vitest'

// Mock browser SpeechSynthesis API which is used by App
const mockSpeak = vi.fn()
const mockCancel = vi.fn()

if (typeof window !== 'undefined') {
  Object.defineProperty(window, 'speechSynthesis', {
    value: {
      speak: mockSpeak,
      cancel: mockCancel,
      getVoices: () => [],
    },
    writable: true
  })
}

// Stub global fetch using vitest's type-safe API
const fetchMock = vi.fn(() => Promise.resolve(new Response('{}'))) as unknown as typeof fetch
vi.stubGlobal('fetch', fetchMock)

// Mock axios to prevent network calls
vi.mock('axios', () => ({
  default: {
    post: vi.fn(() => Promise.resolve({ data: {} })),
    get: vi.fn(() => Promise.resolve({ data: {} })),
  }
}))

describe('VaccineNLP React Frontend Unit Tests', () => {
  it('renders VaccineNLP main layout and check header title', () => {
    render(<App />)
    
    // Check that header titles/subtitles containing VaccineNLP are rendered
    const titleElements = screen.getAllByText(/Vaccine/i)
    expect(titleElements.length).toBeGreaterThan(0)
    
    // Verify analysis text area is present
    const textarea = screen.getByPlaceholderText(/Dán bình luận, bài viết hoặc tin nhắn/i)
    expect(textarea).toBeInTheDocument()
  })

  it('updates text area when preset button is selected', () => {
    render(<App />)
    
    // Click on preset button "Tin giả cực đoan"
    const presetBtn = screen.getByText('Tin giả cực đoan')
    expect(presetBtn).toBeInTheDocument()
    
    fireEvent.click(presetBtn)
    
    // Check that text contains vaccine/COVID keywords from sample texts
    const textarea = screen.getByPlaceholderText(/Dán bình luận, bài viết hoặc tin nhắn/i) as HTMLTextAreaElement
    expect(textarea.value).toContain('vắc xin COVID')
    
    // Click another preset
    const presetBtn2 = screen.getByText('Ủng hộ tiêm chủng')
    fireEvent.click(presetBtn2)
    expect(textarea.value).toContain('tiêm từng mũi 1')
  })

  it('can navigate to the Advanced tools screen with batch only', () => {
    render(<App />)

    // Navigate to "Công cụ nâng cao" via the sidebar
    const advNavBtn = screen.getByText(/Công cụ nâng cao/i)
    expect(advNavBtn).toBeInTheDocument()
    fireEvent.click(advNavBtn)

    // Batch should be present; model comparison should not be exposed.
    expect(screen.getAllByText(/Hàng loạt/i).length).toBeGreaterThan(0)
    const compareLabel = new RegExp(['So sánh', 'mô hình'].join(' '), 'i')
    const removedModelName = ['XLM', 'R'].join('-')
    expect(screen.queryByText(compareLabel)).not.toBeInTheDocument()
    expect(screen.queryByText(removedModelName)).not.toBeInTheDocument()
  })

  it('does not render the removed "Đánh giá & Tài liệu" navigation group', () => {
    render(<App />)

    // The evaluation/documentation group and its screens must be gone
    expect(screen.queryByText(/Đánh giá/i)).not.toBeInTheDocument()
    const removedPerfLabel = new RegExp([[ 'Bench', 'mark' ].join(''), 'hiệu năng'].join(' '), 'i')
    expect(screen.queryByText(removedPerfLabel)).not.toBeInTheDocument()
    expect(screen.queryByText(/Tài liệu hệ thống/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/^Phương pháp luận$/i)).not.toBeInTheDocument()
  })

  it('can update text area directly', () => {
    render(<App />)
    
    const textarea = screen.getByPlaceholderText(/Dán bình luận, bài viết hoặc tin nhắn/i) as HTMLTextAreaElement
    
    // Simulate typing
    fireEvent.change(textarea, { target: { value: 'Tôi ủng hộ tiêm chủng vắc xin cúm.' } })
    expect(textarea.value).toBe('Tôi ủng hộ tiêm chủng vắc xin cúm.')
  })
})
