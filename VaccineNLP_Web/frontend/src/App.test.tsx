import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import App from './App'
import axios from 'axios'
import { vi, describe, it, expect, beforeEach } from 'vitest'

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
const fetchMock = vi.fn(() => Promise.resolve(new Response('{}')))
vi.stubGlobal('fetch', fetchMock as unknown as typeof fetch)

// Mock axios to prevent network calls
vi.mock('axios', () => ({
  default: {
    post: vi.fn(() => Promise.resolve({ data: {} })),
    get: vi.fn(() => Promise.resolve({ data: {} })),
  }
}))

const axiosPost = axios.post as unknown as ReturnType<typeof vi.fn>
const axiosGet = axios.get as unknown as ReturnType<typeof vi.fn>

const analysisFixture = {
  id: 1,
  misinfo_label: 'Fake',
  misinfo_score: 0.9928,
  stance_label: 'Against',
  stance_score: 0.5862,
  sentiment_label: 'Negative',
  sentiment_score: 0.5447,
  phobert_probs: {
    misinfo: { Fake: 0.9928, Real: 0.0072 },
    stance: { Favor: 0.1121, Against: 0.5862, Neutral: 0.3018 },
    sentiment: { Negative: 0.5447, Neutral: 0.2001, Positive: 0.2552 },
  },
  consistency_flag: 'high_risk',
  xai_status: 'idle',
  xai_explanation: null,
}

const historyFixture = {
  ...analysisFixture,
  source_text: 'Vắc xin gây vô sinh là tin đồn sai lệch.',
  source_url: null,
  created_at: '2026-06-08T09:00:00',
}

const crawlFixture = {
  source_type: 'news',
  source_info: 'Báo điện tử',
  count: 2,
  texts: [
    'Crawled vaccine article content for analysis.',
    'Second crawled vaccine comment for batch mode.',
  ],
  batch_text: 'Crawled vaccine article content for analysis.\n\n---\n\nSecond crawled vaccine comment for batch mode.',
}

beforeEach(() => {
  axiosPost.mockReset()
  axiosGet.mockReset()
  axiosPost.mockResolvedValue({ data: {} })
  axiosGet.mockResolvedValue({ data: [] })
  fetchMock.mockClear()
})

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

  it('loads a CSV file into the batch textarea', async () => {
    render(<App />)

    fireEvent.click(screen.getByText(/Công cụ nâng cao/i))
    const fileInput = screen.getByLabelText(/Nhập file dữ liệu/i) as HTMLInputElement
    const file = new File(['text\nVắc xin tốt\nTin đồn gây vô sinh'], 'sample.csv', { type: 'text/csv' })

    fireEvent.change(fileInput, { target: { files: [file] } })

    await waitFor(() => expect(screen.getByText(/Đã nạp 2 dòng/i)).toBeInTheDocument())
    const textarea = screen.getByDisplayValue(/Vắc xin tốt/i) as HTMLTextAreaElement
    expect(textarea.value).toContain('Tin đồn gây vô sinh')
  })

  it('crawls a URL and fills the batch textarea', async () => {
    axiosPost.mockResolvedValueOnce({ data: crawlFixture })
    render(<App />)

    fireEvent.click(screen.getByText(/Công cụ nâng cao/i))
    fireEvent.change(screen.getByPlaceholderText(/youtube\.com/i), {
      target: { value: 'https://vnexpress.net/vaccine-demo' },
    })
    fireEvent.click(screen.getByRole('button', { name: /Thu thập/i }))

    await waitFor(() => expect(axiosPost).toHaveBeenCalledWith(
      expect.stringContaining('/api/crawl-url'),
      expect.objectContaining({ url: 'https://vnexpress.net/vaccine-demo', max_items: 30 })
    ))
    await waitFor(() => expect(screen.getByText(/2 mục/i)).toBeInTheDocument())
    const textarea = screen.getByDisplayValue(/Crawled vaccine article/i) as HTMLTextAreaElement
    expect(textarea.value).toContain('Second crawled vaccine comment')
  })

  it('runs batch analysis through the database-backed endpoint', async () => {
    axiosPost.mockResolvedValueOnce({ data: { rows: [historyFixture] } })
    render(<App />)

    fireEvent.click(screen.getByText(/Công cụ nâng cao/i))
    fireEvent.click(screen.getByRole('button', { name: /Phân tích hàng loạt/i }))

    await waitFor(() => expect(axiosPost).toHaveBeenCalledWith(
      expect.stringContaining('/api/batch-analyze'),
      expect.objectContaining({ texts: expect.any(Array) })
    ))
    await waitFor(() => expect(screen.getByText(/Vắc xin gây vô sinh/i)).toBeInTheDocument())
    expect(screen.getAllByRole('button', { name: /CSV/i }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('button', { name: /Báo cáo/i }).length).toBeGreaterThan(0)
  })

  it('loads saved analysis history from the API', async () => {
    axiosGet.mockResolvedValueOnce({ data: [historyFixture] })
    render(<App />)

    fireEvent.click(screen.getByText(/Lịch sử phân tích/i))

    await waitFor(() => expect(axiosGet).toHaveBeenCalledWith(expect.stringContaining('/api/history?limit=80')))
    await waitFor(() => expect(screen.getByText(/Vắc xin gây vô sinh/i)).toBeInTheDocument())
    expect(screen.getByRole('button', { name: /Mở/i })).toBeInTheDocument()
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

  it('renders the risk radar after an analysis response with phobert_probs', async () => {
    axiosPost.mockResolvedValueOnce({ data: analysisFixture })
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Tiến hành phân tích/i }))

    await waitFor(() => expect(screen.getByTestId('risk-radar')).toBeInTheDocument())
    expect(screen.getByText(/Hồ sơ rủi ro/i)).toBeInTheDocument()
    expect(screen.getAllByText(/99\.3%/).length).toBeGreaterThan(0)
  })

  it('shows Captum IG controls and renders embedding-layer metadata', async () => {
    axiosPost
      .mockResolvedValueOnce({ data: analysisFixture })
      .mockResolvedValueOnce({
        data: {
          pred_class: 0,
          pred_label: 'Fake',
          embedding_layer: 'encoder.embeddings',
          tokens: [
            { token: 'vắc_xin', score: 0.42 },
            { token: 'vô_sinh', score: 0.73 },
          ],
        },
      })
    render(<App />)

    fireEvent.click(screen.getByRole('button', { name: /Tiến hành phân tích/i }))
    await waitFor(() => expect(screen.getByTestId('risk-radar')).toBeInTheDocument())
    fireEvent.click(screen.getByRole('button', { name: /Captum IG/i }))
    fireEvent.click(screen.getByRole('button', { name: /Tính token attribution/i }))

    await waitFor(() => expect(screen.getByText('encoder.embeddings')).toBeInTheDocument())
    expect(screen.getByText('vắc_xin')).toBeInTheDocument()
    expect(screen.getByText('vô_sinh')).toBeInTheDocument()
  })
})
