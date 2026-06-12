import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = import.meta.env.VITE_API_URL || ''

/* ============================================================
   ICON SPRITE (line-icons) + helper <Icon/>
   ============================================================ */
function IconSprite() {
  return (
    <svg width="0" height="0" style={{ position: 'absolute' }} aria-hidden="true">
      <symbol id="i-analyze" viewBox="0 0 24 24"><circle cx="11" cy="11" r="7" /><path d="m21 21-4.3-4.3" /><path d="M11 8v6M8 11h6" /></symbol>
      <symbol id="i-advanced" viewBox="0 0 24 24"><path d="M12 3 3 8l9 5 9-5-9-5Z" /><path d="m3 13 9 5 9-5M3 18l9 5 9-5" /></symbol>
      <symbol id="i-bench" viewBox="0 0 24 24"><path d="M3 21h18" /><rect x="5" y="11" width="4" height="7" /><rect x="14" y="6" width="4" height="12" /></symbol>
      <symbol id="i-docs" viewBox="0 0 24 24"><path d="M4 5a2 2 0 0 1 2-2h9l5 5v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2Z" /><path d="M14 3v5h5M8 13h8M8 17h6" /></symbol>
      <symbol id="i-method" viewBox="0 0 24 24"><path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3" /><path d="M7.5 15h9" /></symbol>
      <symbol id="i-shield" viewBox="0 0 24 24"><path d="M12 3 5 6v5c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6Z" /><path d="M12 8v4M12 15.5h.01" /></symbol>
      <symbol id="i-check" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" /><path d="m8.5 12 2.5 2.5L16 9.5" /></symbol>
      <symbol id="i-download" viewBox="0 0 24 24"><path d="M12 3v12m0 0 4-4m-4 4-4-4" /><path d="M5 19h14" /></symbol>
      <symbol id="i-link" viewBox="0 0 24 24"><path d="M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-1 1" /><path d="M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l1-1" /></symbol>
      <symbol id="i-spark" viewBox="0 0 24 24"><path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18" /></symbol>
      <symbol id="i-arrow" viewBox="0 0 24 24"><path d="M5 12h14m0 0-6-6m6 6-6 6" /></symbol>
      <symbol id="i-send" viewBox="0 0 24 24"><path d="M22 2 11 13M22 2l-7 20-4-9-9-4Z" /></symbol>
      <symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4 12H2M22 12h-2M5 5l1.5 1.5M17.5 17.5 19 19M19 5l-1.5 1.5M6.5 17.5 5 19" /></symbol>
      <symbol id="i-upload" viewBox="0 0 24 24"><path d="M12 15V3m0 0L8 7m4-4 4 4" /><path d="M5 17v2a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-2" /></symbol>
      <symbol id="i-refresh" viewBox="0 0 24 24"><path d="M20 12a8 8 0 1 1-2.34-5.66" /><path d="M20 4v5h-5" /></symbol>
      <symbol id="i-scale" viewBox="0 0 24 24"><path d="M12 3v18M7 7h10M5 7l-2 5a3 3 0 0 0 6 0L7 7M17 7l-2 5a3 3 0 0 0 6 0l-2-5" /></symbol>
      <symbol id="i-data" viewBox="0 0 24 24"><ellipse cx="12" cy="6" rx="7" ry="3" /><path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" /></symbol>
      <symbol id="i-chevron" viewBox="0 0 24 24"><path d="m9 6 6 6-6 6" /></symbol>
    </svg>
  )
}
const Icon = ({ id, className = 'icon', style }: { id: string; className?: string; style?: React.CSSProperties }) => (
  <svg className={className} style={style}><use href={`#${id}`} /></svg>
)

/* ============================================================
   ÁNH XẠ NHÃN + LỚP MÀU
   ============================================================ */
type Probs = Record<string, number>
interface Analysis {
  id: number
  misinfo_label: string; misinfo_score: number
  stance_label: string; stance_score: number
  sentiment_label: string; sentiment_score: number
  phobert_probs?: { misinfo: Probs; stance: Probs; sentiment: Probs }
  consistency_flag: string
  xai_status: string
  xai_explanation?: {
    parse_ok: boolean; reasoning?: string; raw_output?: string
    disagreement?: Record<string, boolean>; gemma_labels?: Record<string, string>
  } | null
  obfuscation?: {
    score: number
    level: string
    flags: {
      zero_width: boolean
      confusable: boolean
      leet: boolean
      separator: boolean
      spacing: boolean
      non_vn_latin_ratio: number
    }
  } | null
  evidence?: Array<{
    id: string
    topic: string
    myth: string
    fact: string
    sources: Array<{ org: string; title: string; url?: string }>
    score: number
  }> | null
  coded_language?: Array<{
    variant: string
    canonical: string
    category: string
  }> | null
  anomaly?: {
    is_anomalous: boolean
    max_similarity: number
    anomaly_tau: number
  } | null
}
interface AttrData { pred_class: number; pred_label: string; embedding_layer: string; tokens: { token: string; score: number }[] }
interface HistoryItem extends Analysis { source_text: string; source_url?: string | null; created_at?: string | null }
interface BatchRow {
  i: number
  id?: number
  text: string
  misinfo: string
  stance: string
  sentiment: string
  consistency_flag?: string
}
interface CrawlData {
  source_type: string
  source_info: string
  count: number
  texts: string[]
  batch_text: string
}

const VI: Record<string, Record<string, string>> = {
  misinfo: { Fake: 'Có dấu hiệu tin giả', Real: 'Không phát hiện dấu hiệu sai lệch' },
  stance: { Favor: 'Ủng hộ', Against: 'Phản đối', Neutral: 'Trung lập' },
  sentiment: { Positive: 'Tích cực', Negative: 'Tiêu cực', Neutral: 'Trung tính' },
}
const BAD = new Set(['Fake', 'Against', 'Negative'])
const GOOD = new Set(['Real', 'Favor', 'Positive'])
const valClass = (label: string) => BAD.has(label) ? 'bad' : GOOD.has(label) ? 'ok' : 'neu'
const meterClass = (label: string) => BAD.has(label) ? 'bad' : GOOD.has(label) ? 'teal' : ''
const barColor = (label: string) => BAD.has(label) ? 'var(--danger)' : GOOD.has(label) ? 'var(--teal)' : 'var(--ink-3)'

const CONSISTENCY: Record<string, { pill: string; text: string; icon: string }> = {
  plausible: { pill: 'ok', text: 'Tổ hợp nhãn hợp lệ', icon: 'i-check' },
  unusual: { pill: 'warn', text: 'Bất thường — nghi mô hình sai', icon: 'i-shield' },
  high_risk: { pill: 'danger', text: 'Nguy cơ cao — nên rà soát', icon: 'i-shield' },
  evasion_suspected: { pill: 'danger', text: 'Nghi ngờ lách bộ lọc', icon: 'i-shield' },
}

const PRESETS = [
  { label: 'Tin giả cực đoan', text: 'Cảnh báo: vắc xin COVID có thể gây vô sinh ở phụ nữ và biến đổi gen ở trẻ em. Mọi người nên tìm hiểu kỹ trước khi làm chuột bạch cho các tập đoàn dược phẩm.' },
  { label: 'Ủng hộ tiêm chủng', text: 'Em cũng đang tiêm từng mũi 1 cho con, chậm mà đủ và an toàn cho con là được. Trộm vía bé chưa sốt, chưa hành mũi nào.' },
  { label: 'Thông tin chuẩn', text: 'Bộ Y tế khuyến cáo trẻ em từ 6 tháng tuổi cần tiêm đủ các mũi vaccine cơ bản theo Chương trình Tiêm chủng Mở rộng để phòng các bệnh truyền nhiễm nguy hiểm.' },
]

const SCREENS = [
  { id: 'analyze', icon: 'i-analyze', label: 'Phân tích văn bản', crumb: 'Phân tích · Đa nhiệm' },
  { id: 'advanced', icon: 'i-advanced', label: 'Công cụ nâng cao', crumb: 'Phân tích · Nâng cao' },
  { id: 'history', icon: 'i-data', label: 'Lịch sử phân tích', crumb: 'Dữ liệu · History' },
] as const
type ScreenId = typeof SCREENS[number]['id']

const TEXT_FIELD_KEYS = ['text', 'content', 'comment', 'message', 'body', 'noi dung', 'nội dung', 'nội_dung']

function splitBatchInput(raw: string): string[] {
  const chunks = raw.includes('---') ? raw.split('---') : raw.split('\n')
  return chunks.map(s => s.trim()).filter(Boolean).slice(0, 100)
}

function parseCsvLine(line: string): string[] {
  const cells: string[] = []
  let cur = ''
  let quoted = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    const next = line[i + 1]
    if (ch === '"' && quoted && next === '"') {
      cur += '"'
      i++
    } else if (ch === '"') {
      quoted = !quoted
    } else if (ch === ',' && !quoted) {
      cells.push(cur.trim())
      cur = ''
    } else {
      cur += ch
    }
  }
  cells.push(cur.trim())
  return cells
}

function pickTextFromRecord(record: Record<string, unknown>): string {
  const normalized = Object.entries(record).map(([key, value]) => ({
    key,
    norm: key.trim().toLowerCase(),
    value: String(value ?? '').trim(),
  }))
  const found = normalized.find(item => TEXT_FIELD_KEYS.some(k => item.norm.includes(k)))
  return (found?.value || normalized.find(item => item.value)?.value || '').trim()
}

function parseDelimitedText(raw: string, delimiter: ',' | '\t'): string[] {
  const lines = raw.replace(/^\uFEFF/, '').split(/\r?\n/).map(l => l.trim()).filter(Boolean)
  if (!lines.length) return []
  const parse = delimiter === ',' ? parseCsvLine : (line: string) => line.split('\t').map(c => c.trim())
  const header = parse(lines[0])
  const headerLooksNamed = header.some(h => TEXT_FIELD_KEYS.some(k => h.toLowerCase().includes(k)))
  if (!headerLooksNamed) return lines.map(line => parse(line)[0]).filter(Boolean)
  return lines.slice(1).map(line => {
    const cells = parse(line)
    const row: Record<string, unknown> = {}
    header.forEach((h, i) => { row[h] = cells[i] || '' })
    return pickTextFromRecord(row)
  }).filter(Boolean)
}

function parseJsonText(raw: string): string[] {
  const trimmed = raw.trim()
  if (!trimmed) return []
  try {
    const parsed = JSON.parse(trimmed)
    const rows = Array.isArray(parsed) ? parsed : [parsed]
    return rows.map(item => typeof item === 'string' ? item : pickTextFromRecord(item as Record<string, unknown>)).filter(Boolean)
  } catch {
    return trimmed.split(/\r?\n/).map(line => {
      try {
        const item = JSON.parse(line)
        return typeof item === 'string' ? item : pickTextFromRecord(item as Record<string, unknown>)
      } catch {
        return ''
      }
    }).filter(Boolean)
  }
}

function parseUploadedDataset(fileName: string, raw: string): string[] {
  const lower = fileName.toLowerCase()
  if (lower.endsWith('.json') || lower.endsWith('.jsonl')) return parseJsonText(raw)
  if (lower.endsWith('.tsv')) return parseDelimitedText(raw, '\t')
  if (lower.endsWith('.csv')) return parseDelimitedText(raw, ',')
  return splitBatchInput(raw)
}

function csvEscape(value: unknown): string {
  const s = String(value ?? '')
  return /[",\n\r]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s
}

function downloadTextFile(fileName: string, content: string, type = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = fileName
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function batchRowsToCsv(rows: BatchRow[]): string {
  const header = ['id', 'text', 'misinfo', 'stance', 'sentiment', 'consistency_flag']
  return [
    header.join(','),
    ...rows.map(r => [r.id, r.text, r.misinfo, r.stance, r.sentiment, r.consistency_flag || ''].map(csvEscape).join(',')),
  ].join('\n')
}

function crawledTextsToCsv(texts: string[]): string {
  return [
    ['stt', 'text'].join(','),
    ...texts.map((text, i) => [i + 1, text].map(csvEscape).join(',')),
  ].join('\n')
}

function buildAnalysisReport(text: string, analysis: Analysis): string {
  const explanation = analysis.xai_explanation?.reasoning || 'Chưa sinh giải thích XAI.'
  return `# Báo cáo phân tích VaccineNLP

**Mã phân tích:** #${analysis.id}

## Văn bản

> ${text}

## Kết quả PhoBERT-v2

| Trục | Nhãn | Độ tin cậy |
|---|---|---:|
| Dấu hiệu sai lệch | ${VI.misinfo[analysis.misinfo_label] || analysis.misinfo_label} | ${(analysis.misinfo_score * 100).toFixed(1)}% |
| Thái độ với vaccine | ${VI.stance[analysis.stance_label] || analysis.stance_label} | ${(analysis.stance_score * 100).toFixed(1)}% |
| Cảm xúc tổng thể | ${VI.sentiment[analysis.sentiment_label] || analysis.sentiment_label} | ${(analysis.sentiment_score * 100).toFixed(1)}% |

**Cờ nhất quán:** \`${analysis.consistency_flag}\`

## Giải thích XAI

${explanation}
`
}

function batchRowsToMarkdown(rows: BatchRow[]): string {
  return `# Báo cáo phân tích hàng loạt VaccineNLP

Tổng số dòng: ${rows.length}

| # | ID | Văn bản | Dấu hiệu sai lệch | Lập trường | Cảm xúc | Cờ |
|---:|---:|---|---|---|---|---|
${rows.map(r => `| ${r.i} | ${r.id || ''} | ${String(r.text).replaceAll('|', '\\|')} | ${VI.misinfo[r.misinfo] || r.misinfo} | ${VI.stance[r.stance] || r.stance} | ${VI.sentiment[r.sentiment] || r.sentiment} | ${r.consistency_flag || ''} |`).join('\n')}
`
}

/* ============================================================
   APP
   ============================================================ */
export default function App() {
  const [screen, setScreen] = useState<ScreenId>('analyze')
  const [theme, setTheme] = useState<'light' | 'dark'>(() => (localStorage.getItem('vnlp-theme') as 'light' | 'dark') || 'light')

  // analyze state
  const [text, setText] = useState(PRESETS[0].text)
  const [sourceUrl, setSourceUrl] = useState('')
  const [urlOpen, setUrlOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [explaining, setExplaining] = useState(false)
  const [tab, setTab] = useState<'cot' | 'cap'>('cot')

  // token attribution state
  const [attrLoading, setAttrLoading] = useState(false)
  const [attrData, setAttrData] = useState<AttrData | null>(null)
  const [attrError, setAttrError] = useState('')

  // advanced state
  const [batchText, setBatchText] = useState(PRESETS.map(p => p.text).join('\n---\n'))
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchRows, setBatchRows] = useState<BatchRow[]>([])
  const [batchProgress, setBatchProgress] = useState('')
  const [fileStatus, setFileStatus] = useState('')
  const [crawlUrl, setCrawlUrl] = useState('')
  const [crawlMax, setCrawlMax] = useState(30)
  const [crawlLoading, setCrawlLoading] = useState(false)
  const [crawlData, setCrawlData] = useState<CrawlData | null>(null)
  const [crawlError, setCrawlError] = useState('')

  // history state
  const [historyRows, setHistoryRows] = useState<HistoryItem[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState('')
  const [historyReloadKey, setHistoryReloadKey] = useState(0)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('vnlp-theme', theme)
  }, [theme])

  useEffect(() => {
    if (screen !== 'history') return
    let alive = true
    const loadHistory = async () => {
      setHistoryLoading(true)
      setHistoryError('')
      try {
        const { data } = await axios.get<HistoryItem[]>(`${API_URL}/api/history?limit=80`)
        if (alive) setHistoryRows(data)
      } catch {
        if (alive) setHistoryError('Không tải được lịch sử phân tích.')
      } finally {
        if (alive) setHistoryLoading(false)
      }
    }
    void loadHistory()
    return () => { alive = false }
  }, [screen, historyReloadKey])

  const cur = SCREENS.find(s => s.id === screen)!

  /* ---- API: phân tích nhanh (PhoBERT) ---- */
  const handleAnalyze = async () => {
    if (!text.trim()) return
    setLoading(true); setAnalysis(null); setTab('cot'); setAttrData(null); setAttrError('')
    try {
      const { data } = await axios.post<Analysis>(`${API_URL}/api/analyze`, { text, source_url: sourceUrl || null })
      setAnalysis(data)
    } catch { alert('Lỗi khi phân tích văn bản. Kiểm tra api_service (cổng 8000).') }
    finally { setLoading(false) }
  }

  /* ---- API: sinh giải thích on-demand (Gemma, stream) ---- */
  const handleExplain = async () => {
    if (!analysis || !text.trim()) return
    setExplaining(true)
    setAnalysis(p => p ? { ...p, xai_status: 'pending', evidence: null, anomaly: null, xai_explanation: { parse_ok: true, reasoning: '' } } : p)
    try {
      const resp = await fetch(`${API_URL}/api/explain-stream`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, source_url: sourceUrl || null }),
      })
      if (!resp.body) throw new Error('no body')
      const reader = resp.body.getReader(); const dec = new TextDecoder('utf-8'); let buf = ''
      while (true) {
        const { value, done } = await reader.read(); if (done) break
        buf += dec.decode(value, { stream: true }); const lines = buf.split('\n'); buf = lines.pop() || ''
        for (const line of lines) {
          if (!line.startsWith('data:')) continue
          const s = line.slice(5).trim(); if (s === '[DONE]') continue
          try {
            const ev = JSON.parse(s)
            if (ev.type === 'token') {
              setAnalysis(p => p ? { ...p, xai_status: 'pending', xai_explanation: { ...(p.xai_explanation || { parse_ok: true }), parse_ok: true, reasoning: ((p.xai_explanation?.reasoning) || '') + ev.content } } : p)
            } else if (ev.type === 'final') {
              setAnalysis(p => p ? { ...p, xai_status: 'done', evidence: ev.evidence || null, anomaly: ev.anomaly || null, xai_explanation: { parse_ok: ev.parse_ok, reasoning: ev.reasoning, raw_output: ev.raw_output, disagreement: ev.disagreement, gemma_labels: ev.gemma_labels } } : p)
            } else if (ev.type === 'error') {
              setAnalysis(p => p ? { ...p, xai_status: 'failed' } : p)
            }
          } catch { /* ignore parse */ }
        }
      }
    } catch { setAnalysis(p => p ? { ...p, xai_status: 'failed' } : p) }
    finally { setExplaining(false) }
  }

  /* ---- API: token attribution (Captum IG, on-demand) ---- */
  const handleAttribute = async () => {
    if (!text.trim()) return
    setAttrLoading(true); setAttrError(''); setAttrData(null)
    try {
      const { data } = await axios.post<AttrData>(`${API_URL}/api/attribute`, { text })
      setAttrData(data)
    } catch (e: unknown) {
      setAttrError(axios.isAxiosError(e) && e.response?.status === 503
        ? 'Cần PhoBERT thật (api_service đang chạy MOCK).'
        : 'Không tính được Integrated Gradients.')
    } finally { setAttrLoading(false) }
  }

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    if (/\.(xlsx|xls)$/i.test(file.name)) {
      setFileStatus('Excel .xlsx tạm thời không được nạp trực tiếp trên Web vì parser frontend phổ biến có lỗ hổng audit cao. Hãy xuất CSV/TXT/JSONL để phân tích hàng loạt.')
      return
    }
    try {
      const raw = await file.text()
      const texts = parseUploadedDataset(file.name, raw)
      if (!texts.length) {
        setFileStatus(`Không tìm thấy cột/nội dung phân tích trong ${file.name}.`)
        return
      }
      setBatchText(texts.join('\n---\n'))
      setFileStatus(`Đã nạp ${texts.length} dòng từ ${file.name}.`)
    } catch {
      setFileStatus(`Không đọc được file ${file.name}.`)
    }
  }

  const handleDownloadCurrentReport = () => {
    if (!analysis) return
    downloadTextFile(`VaccineNLP_Report_${analysis.id}.md`, buildAnalysisReport(text, analysis), 'text/markdown;charset=utf-8')
  }

  const handleDownloadBatchCsv = () => {
    if (!batchRows.length) return
    downloadTextFile('VaccineNLP_Batch_Results.csv', batchRowsToCsv(batchRows), 'text/csv;charset=utf-8')
  }

  const handleDownloadBatchReport = () => {
    if (!batchRows.length) return
    downloadTextFile('VaccineNLP_Batch_Report.md', batchRowsToMarkdown(batchRows), 'text/markdown;charset=utf-8')
  }

  const handleCrawlUrl = async () => {
    if (!crawlUrl.trim()) return
    setCrawlLoading(true)
    setCrawlError('')
    setCrawlData(null)
    try {
      const { data } = await axios.post<CrawlData>(`${API_URL}/api/crawl-url`, {
        url: crawlUrl.trim(),
        max_items: crawlMax,
      })
      setCrawlData(data)
      setBatchText(data.batch_text)
      setFileStatus(`Đã đưa ${data.count} dòng thu thập từ URL vào batch.`)
    } catch (e: unknown) {
      const detail = axios.isAxiosError(e) ? e.response?.data?.detail : ''
      setCrawlError(detail || 'Không thu thập được dữ liệu từ URL này.')
    } finally {
      setCrawlLoading(false)
    }
  }

  const handleUseCrawledText = (value: string) => {
    setText(value)
    setSourceUrl(crawlUrl)
    setAnalysis(null)
    setAttrData(null)
    setAttrError('')
    setTab('cot')
    setScreen('analyze')
  }

  const handleSendCrawlToBatch = () => {
    if (!crawlData) return
    setBatchText(crawlData.batch_text)
    setFileStatus(`Đã đưa ${crawlData.count} dòng thu thập từ URL vào batch.`)
  }

  const handleDownloadCrawlCsv = () => {
    if (!crawlData?.texts.length) return
    downloadTextFile('VaccineNLP_Crawled_Data.csv', crawledTextsToCsv(crawlData.texts), 'text/csv;charset=utf-8')
  }

  const handleOpenHistoryItem = (row: HistoryItem) => {
    setText(row.source_text)
    setSourceUrl(row.source_url || '')
    setAnalysis(row)
    setAttrData(null)
    setAttrError('')
    setTab('cot')
    setScreen('analyze')
  }

  const handleDownloadServerReport = (id: number) => {
    const a = document.createElement('a')
    a.href = `${API_URL}/api/report/${id}.md`
    a.download = `VaccineNLP_Report_${id}.md`
    document.body.appendChild(a)
    a.click()
    a.remove()
  }

  /* ---- Batch: chạy THẬT qua /api/batch-analyze và lưu database ---- */
  const handleBatch = async () => {
    const lines = splitBatchInput(batchText)
    if (!lines.length) return
    setBatchLoading(true); setBatchRows([])
    setBatchProgress(`0/${lines.length}`)
    try {
      const { data } = await axios.post<{ rows: HistoryItem[] }>(`${API_URL}/api/batch-analyze`, { texts: lines })
      const rows = data.rows.map((r, i) => ({
        i: i + 1,
        id: r.id,
        text: r.source_text,
        misinfo: r.misinfo_label,
        stance: r.stance_label,
        sentiment: r.sentiment_label,
        consistency_flag: r.consistency_flag,
      }))
      setBatchRows(rows)
      setBatchProgress(`${rows.length}/${lines.length}`)
    } catch {
      setFileStatus('Không phân tích được batch. Kiểm tra api_service.')
    } finally {
      setBatchLoading(false)
    }
  }

  return (
    <>
      <IconSprite />
      <div className="app">
        {/* ============ SIDEBAR ============ */}
        <aside className="sidebar">
          <div className="brand">
            <div className="logo"><Icon id="i-shield" className="icon lg" /></div>
            <div>
              <div className="name">Vaccine<span>NLP</span></div>
              <div className="tag">Hệ thống giám sát thông tin sai lệch về vaccine</div>
            </div>
          </div>
          <div className="nav-group">Phân tích</div>
          {SCREENS.map(s => (
            <button key={s.id} className={`nav-item ${screen === s.id ? 'active' : ''}`} onClick={() => setScreen(s.id)}>
              <Icon id={s.icon} /> {s.label}
            </button>
          ))}
          <div className="side-foot">
            <div className="status-chip"><span className="pulse" /> Mô hình: <b>PhoBERT-v2</b> · trực tuyến</div>
            <button className="theme-toggle" onClick={() => setTheme(t => t === 'light' ? 'dark' : 'light')}>
              <span>Giao diện sáng / tối</span>
              <span className="sw"><i><Icon id="i-sun" className="icon sm" /></i></span>
            </button>
          </div>
        </aside>

        {/* ============ MAIN ============ */}
        <div className="main">
          <header className="topbar">
            <div>
              <div className="crumb">{cur.crumb}</div>
              <h1>{cur.label}</h1>
            </div>
            <div className="spacer" />
          </header>

          <div className="content">
            {/* ====================== ANALYZE ====================== */}
            {screen === 'analyze' && (
              <div className="screen">
                <div className="grid-2">
                  {/* -------- INPUT -------- */}
                  <div className="card card-pad" style={{ position: 'sticky', top: 96 }}>
                    <div className="field-label">Nội dung cần đối soát <span className="hint">tiếng Việt</span></div>
                    <textarea className="input" value={text} onChange={e => setText(e.target.value)}
                      placeholder="Dán bình luận, bài viết hoặc tin nhắn về vaccine…" />

                    <div className="field-label" style={{ margin: '18px 0 6px' }}>Bộ ví dụ mẫu</div>
                    <div className="seg">
                      {PRESETS.map(p => (
                        <button key={p.label} className={text === p.text ? 'on' : ''} onClick={() => setText(p.text)}>{p.label}</button>
                      ))}
                    </div>

                    <div style={{ marginTop: 16 }}>
                      <div className="accordion-head" onClick={() => setUrlOpen(o => !o)}>
                        <span><Icon id="i-link" className="icon sm" /> Hoặc thu thập từ URL</span>
                        <Icon id="i-chevron" className="icon sm" />
                      </div>
                      {urlOpen && (
                        <input className="input" style={{ marginTop: 8 }} value={sourceUrl} onChange={e => setSourceUrl(e.target.value)}
                          placeholder="https:// … (Báo chí, YouTube, Facebook)" />
                      )}
                    </div>

                    <button className="btn primary block" style={{ marginTop: 20 }} onClick={handleAnalyze} disabled={loading || !text.trim()}>
                      <Icon id="i-send" className="icon sm" /> {loading ? 'Đang phân tích…' : 'Tiến hành phân tích đa nhiệm'}
                    </button>
                  </div>

                  {/* -------- RESULT -------- */}
                  <div className="stack">
                    {!analysis ? (
                      <div className="card"><div className="empty-state">
                        <div className="ring"><Icon id="i-analyze" className="icon lg" /></div>
                        <h3>Chưa có phân tích</h3>
                        <div className="muted" style={{ fontSize: 13, maxWidth: 360 }}>Nhập văn bản và bấm “Tiến hành phân tích đa nhiệm” để chạy PhoBERT-v2 trên 3 trục nhãn.</div>
                      </div></div>
                    ) : (
                      <>
                        {/* VERDICT HERO */}
                        {(() => {
                          const fake = analysis.misinfo_label === 'Fake'
                          const c = CONSISTENCY[analysis.consistency_flag] || CONSISTENCY.plausible
                          return (
                            <div className={`verdict ${fake ? '' : 'ok'}`}>
                              <div className="glyph"><Icon id={fake ? 'i-shield' : 'i-check'} className="icon lg" /></div>
                              <div>
                                <div className="vtitle">Kết luận đối soát</div>
                                <div className="vmain">{fake ? 'Có dấu hiệu tin giả' : 'Không phát hiện dấu hiệu sai lệch'}</div>
                                <div className="vmeta">Độ tin cậy mô hình <b className="mono">{(analysis.misinfo_score * 100).toFixed(1)}%</b></div>
                              </div>
                              <div className="vside">
                                {(analysis.consistency_flag === 'evasion_suspected' || analysis.obfuscation?.level === 'high') && (
                                  <span
                                    className="pill danger"
                                    style={{ cursor: 'help' }}
                                    title="Văn bản có dấu hiệu chèn ký tự lạ/ẩn để né phát hiện; kết quả lấy từ bản đã chuẩn hoá."
                                  >
                                    ⚠️ Nghi ngờ lách bộ lọc (nhiễu ký tự)
                                  </span>
                                )}
                                {analysis.coded_language && analysis.coded_language.length > 0 && (
                                  <span
                                    className="pill warn"
                                    style={{ cursor: 'help' }}
                                    title={`Phát hiện ngôn từ ngụy trang né bộ lọc: ${analysis.coded_language.map(x => x.variant).join(', ')}`}
                                  >
                                    ⚠️ Phát hiện uyển ngữ/ngôn ngữ ngụy trang
                                  </span>
                                )}
                                {analysis.anomaly?.is_anomalous && (
                                  <span
                                    className="pill warn"
                                    style={{ cursor: 'help' }}
                                    title={`Tín hiệu tham khảo. Max similarity: ${(analysis.anomaly.max_similarity * 100).toFixed(1)}% (ngưỡng: ${analysis.anomaly.anomaly_tau * 100}%).`}
                                  >
                                    ℹ️ Văn bản lệch xa diễn ngôn vaccine đã biết
                                  </span>
                                )}
                                <span className={`pill ${c.pill}`}><Icon id={c.icon} className="icon sm" /> {c.text}</span>
                                <span className="muted mono" style={{ fontSize: 12 }}>Phiên #{analysis.id} · PhoBERT-v2</span>
                              </div>
                            </div>
                          )
                        })()}

                        {/* CLASSIFICATION CARD */}
                        <div className="card card-pad">
                          <div className="section-label"><Icon id="i-analyze" className="icon sm ic" /> Kết quả phân loại nhãn</div>
                          <div className="axes">
                            <AxisCard task="Dấu hiệu sai lệch" label={analysis.misinfo_label} score={analysis.misinfo_score} scaleHint="Ngưỡng 50%" />
                            <AxisCard task="Lập trường" label={analysis.stance_label} score={analysis.stance_score} scaleHint="Favor · Against · Neutral" />
                            <AxisCard task="Cảm xúc" label={analysis.sentiment_label} score={analysis.sentiment_score} scaleHint="Pos · Neg · Neutral" />
                          </div>

                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 16 }}>
                            <div style={{ border: '1px solid var(--line)', borderRadius: 'var(--r)', background: 'var(--surface-2)', padding: 16 }}>
                              <div className="muted" style={{ fontSize: 11, fontWeight: 600, letterSpacing: '.4px', textTransform: 'uppercase', marginBottom: 6 }}>Hồ sơ rủi ro · P(Tin giả / Phản đối / Tiêu cực)</div>
                              <RiskRadar probs={analysis.phobert_probs} />
                            </div>
                            <div style={{ border: '1px solid var(--line)', borderRadius: 'var(--r)', background: 'var(--surface-2)', padding: 16 }}>
                              <div className="muted" style={{ fontSize: 11, fontWeight: 600, letterSpacing: '.4px', textTransform: 'uppercase', marginBottom: 10 }}>Phân phối softmax · phobert_probs</div>
                              {analysis.phobert_probs ? (
                                <div className="distrib">
                                  {(['misinfo', 'stance', 'sentiment'] as const).map(t => (
                                    <div className="grp" key={t}>
                                      <div className="glabel">{t}</div>
                                      {Object.entries(analysis.phobert_probs![t]).map(([lab, p]) => (
                                        <div className="bar" key={lab}>
                                          <span className="nm">{VI[t][lab] || lab}</span>
                                          <span className="track"><i style={{ width: `${(p * 100).toFixed(1)}%`, background: barColor(lab) }} /></span>
                                          <span className="pct mono">{(p * 100).toFixed(1)}%</span>
                                        </div>
                                      ))}
                                    </div>
                                  ))}
                                </div>
                              ) : <div className="muted" style={{ fontSize: 12 }}>Không có dữ liệu phân phối.</div>}
                            </div>
                          </div>

                          <div style={{ marginTop: 16, borderTop: '1px solid var(--line-2)', paddingTop: 14 }}>
                            <div className="muted" style={{ fontSize: 12, marginBottom: 9 }}>Cờ nhất quán tam giác nhãn <span className="mono">(consistency_flag)</span> — mức ưu tiên rà soát:</div>
                            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              {(['plausible', 'unusual', 'high_risk', 'evasion_suspected'] as const).map(k => {
                                const on = analysis.consistency_flag === k
                                return <span key={k} className={`pill ${CONSISTENCY[k].pill}`} style={on ? { outline: '2px solid color-mix(in srgb,var(--ink-3) 40%,transparent)', outlineOffset: 1 } : { opacity: .55 }}>{k} · {CONSISTENCY[k].text}{on ? ' ◂ hiện tại' : ''}</span>
                              })}
                            </div>
                          </div>
                        </div>

                        {/* XAI CARD */}
                        <div className="card card-pad">
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', marginBottom: 4 }}>
                            <div className="section-label"><Icon id="i-spark" className="icon sm ic" /> Giải thích của mô hình (XAI)</div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <XaiPill status={analysis.xai_status} />
                              <button className="btn sm ghost" onClick={handleDownloadCurrentReport}>
                                <Icon id="i-download" className="icon sm" /> Tải báo cáo
                              </button>
                              <button className="btn sm" onClick={handleExplain} disabled={explaining || analysis.xai_status === 'pending'}>
                                <Icon id="i-spark" className="icon sm" /> {explaining ? 'Đang sinh…' : analysis.xai_status === 'done' ? 'Sinh lại' : 'Sinh giải thích (Gemma, chậm)'}
                              </button>
                            </div>
                          </div>
                          <p className="muted" style={{ fontSize: 12, margin: '2px 0 8px' }}>Nhịp 1 · PhoBERT trả nhãn tức thời → Nhịp 2 · Gemma-4B stream lý giải CoT theo token <span className="mono">(/api/explain-stream)</span></p>

                          <div className="tabs">
                            <button className={`tab ${tab === 'cot' ? 'on' : ''}`} onClick={() => setTab('cot')}><Icon id="i-spark" className="icon sm" /> Chain-of-Thought</button>
                            <button className={`tab ${tab === 'cap' ? 'on' : ''}`} onClick={() => setTab('cap')}><Icon id="i-analyze" className="icon sm" /> Captum IG</button>
                          </div>

                          {tab === 'cot' && (
                            <div className="tabpane">
                              {analysis.xai_status === 'idle' && (
                                <div className="muted" style={{ fontSize: 13.5, padding: '8px 0' }}>Bấm “Sinh giải thích” để Gemma-4B lý giải chuỗi suy luận (CoT) cho kết luận trên.</div>
                              )}
                              {(analysis.xai_status === 'pending') && (
                                <div className="streamwrap">{analysis.xai_explanation?.reasoning}<span className="cursor" /></div>
                              )}
                              {analysis.xai_status === 'failed' && (
                                <div className="streamwrap" style={{ color: 'var(--danger)' }}>Không sinh được giải thích. Kiểm tra xai_service / GGUF.</div>
                              )}
                              {analysis.xai_status === 'done' && analysis.xai_explanation && (
                                <>
                                  {analysis.xai_explanation.parse_ok && analysis.xai_explanation.reasoning ? (
                                    <div className="streamwrap">{analysis.xai_explanation.reasoning}</div>
                                  ) : analysis.xai_explanation.raw_output ? (
                                    <div className="streamwrap"><span className="pill warn" style={{ marginBottom: 8 }}><Icon id="i-shield" className="icon sm" /> Định dạng chưa tối ưu — hiển thị thô</span>{'\n'}{analysis.xai_explanation.raw_output}</div>
                                  ) : <div className="muted">Không có nội dung lý giải.</div>}

                                  {analysis.xai_explanation.parse_ok && analysis.xai_explanation.gemma_labels && (
                                    <div style={{ marginTop: 20 }}>
                                      <div className="section-label" style={{ marginBottom: 10 }}><Icon id="i-scale" className="icon sm ic" /> Bất đồng thuận nhãn — PhoBERT vs Gemma</div>
                                      <table className="dtable">
                                        <thead><tr><th>Trục</th><th>PhoBERT-v2</th><th>Gemma-4-4B</th><th style={{ textAlign: 'center' }}>Khớp</th></tr></thead>
                                        <tbody>
                                          {([['misinfo', 'Dấu hiệu sai lệch', analysis.misinfo_label], ['stance', 'Lập trường', analysis.stance_label], ['sentiment', 'Cảm xúc', analysis.sentiment_label]] as const).map(([key, vi, pLabel]) => {
                                            const g = analysis.xai_explanation!.gemma_labels![key]
                                            const diff = analysis.xai_explanation!.disagreement?.[key]
                                            return (
                                              <tr key={key} className={diff ? 'flag' : ''}>
                                                <td>{vi}</td>
                                                <td>{VI[key][pLabel] || pLabel}</td>
                                                <td>{g ? (VI[key][g] || g) : '—'}</td>
                                                <td style={{ textAlign: 'center' }} className={diff ? 'no' : 'yes'}>{diff ? '≠' : '✓'}</td>
                                              </tr>
                                            )
                                          })}
                                        </tbody>
                                      </table>
                                    </div>
                                  )}
                                </>
                              )}
                            </div>
                          )}

                          {tab === 'cap' && (
                            <div className="tabpane">
                              <p className="muted" style={{ fontSize: 13.5, margin: '0 0 14px' }}>
                                Captum Integrated Gradients đo mức đóng góp của từng token vào nhãn <b style={{ color: analysis.misinfo_label === 'Fake' ? 'var(--danger)' : 'var(--teal-strong)' }}>“{VI.misinfo[analysis.misinfo_label]}”</b>.
                              </p>
                              {!attrData && !attrLoading && (
                                <button className="btn sm" onClick={handleAttribute}><Icon id="i-analyze" className="icon sm" /> Tính token attribution (IG · chậm)</button>
                              )}
                              {attrLoading && <div className="streamwrap">Đang tính Integrated Gradients…<span className="cursor" /></div>}
                              {attrError && <div className="muted" style={{ color: 'var(--danger)', fontSize: 13 }}>{attrError}</div>}
                              {attrData && (
                                <>
                                  <div className="ig-meta">
                                    <span className="pill ok"><Icon id="i-check" className="icon sm" /> Captum IG</span>
                                    <span>Target misinfo: <b>{attrData.pred_label}</b></span>
                                    <span>Embedding: <b className="mono">{attrData.embedding_layer}</b></span>
                                  </div>
                                  <div className="heat">
                                    {attrData.tokens.map((t, i) => {
                                      const a = Math.abs(t.score); const al = Math.min(a, 0.85); const fake = attrData.pred_class === 0
                                      const hot = a >= 0.15
                                      return <span className="tok" key={i} style={hot ? { background: fake ? `rgba(210,69,58,${al})` : `rgba(14,147,132,${al})`, color: al > 0.5 ? '#fff' : 'var(--ink-2)', borderColor: 'transparent' } : undefined}>{t.token}</span>
                                    })}
                                  </div>
                                  <div className="legend-row">
                                    <span>Thấp</span>
                                    <span className="scale"><i style={{ background: 'var(--surface-2)' }} /><i style={{ background: 'var(--danger-50)' }} /><i style={{ background: 'color-mix(in srgb,var(--danger) 45%,transparent)' }} /><i style={{ background: 'var(--danger)' }} /></span>
                                    <span>Cao</span>
                                    <button className="btn sm ghost" style={{ marginLeft: 'auto' }} onClick={handleAttribute}>Tính lại</button>
                                  </div>
                                </>
                              )}
                            </div>
                          )}
                        </div>

                        {/* EVIDENCE PANEL */}
                        {((analysis.evidence && analysis.evidence.length > 0) || (analysis.anomaly && analysis.anomaly.is_anomalous)) && (
                          <div className="card card-pad" style={{ animation: 'rise .42s cubic-bezier(.2,.7,.3,1)' }}>
                            <div className="section-label" style={{ marginBottom: 4 }}>
                              <span style={{ marginRight: 6 }}>🔍</span> Đối chiếu bằng chứng
                            </div>
                            <p className="muted" style={{ fontSize: 12, margin: '2px 0 16px' }}>
                              Đối soát tự động với cơ sở tri thức y khoa (Fact-KB) đã xác thực thông qua so khớp vector.
                            </p>

                            {analysis.anomaly && analysis.anomaly.is_anomalous && (
                              <div style={{ background: 'var(--warn-50)', border: '1px solid color-mix(in srgb, var(--warn) 30%, transparent)', borderRadius: 'var(--r-sm)', padding: 12, marginBottom: 14, color: 'var(--warn)', fontSize: 13, display: 'flex', alignItems: 'center', gap: 8 }}>
                                <span style={{ fontSize: 16 }}>ℹ️</span>
                                <div>
                                  <b>Cảnh báo lệch diễn ngôn:</b> Văn bản lệch xa diễn ngôn vaccine đã biết (độ tương đồng tối đa: {(analysis.anomaly.max_similarity * 100).toFixed(1)}% &lt; {analysis.anomaly.anomaly_tau * 100}%). Ghi chú: Đây chỉ là tín hiệu tham khảo.
                                </div>
                              </div>
                            )}

                            {analysis.evidence && analysis.evidence.length > 0 && (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                                {analysis.evidence.map((item, idx) => (
                                  <div key={item.id || idx} style={{ border: '1px solid var(--line)', borderRadius: 'var(--r)', background: 'var(--surface-2)', padding: 16 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8, borderBottom: '1px solid var(--line-2)', paddingBottom: 8, marginBottom: 10 }}>
                                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                        <span className="pill ok" style={{ fontSize: 11, padding: '2px 8px' }}>
                                          Chủ đề: {item.topic || 'Chưa phân loại'}
                                        </span>
                                        <span className="pill" style={{ fontSize: 11, padding: '2px 8px', fontFamily: 'var(--mono)' }}>
                                          Độ tương đồng: {(item.score * 100).toFixed(1)}%
                                        </span>
                                      </div>
                                      <span className="muted mono" style={{ fontSize: 11 }}>#{item.id}</span>
                                    </div>

                                    <div style={{ fontSize: 13.5, lineHeight: 1.6 }}>
                                      <div style={{ marginBottom: 8 }}>
                                        <b style={{ color: 'var(--danger)', marginRight: 4 }}>Ý kiến / Tin đồn:</b>
                                        <span style={{ color: 'var(--ink)' }}>{item.myth}</span>
                                      </div>
                                      <div style={{ marginBottom: 8 }}>
                                        <b style={{ color: 'var(--teal-strong)', marginRight: 4 }}>Thông tin xác thực:</b>
                                        <span style={{ color: 'var(--ink)' }}>{item.fact}</span>
                                      </div>
                                    </div>

                                    {item.sources && item.sources.length > 0 && (
                                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginTop: 12, paddingTop: 8, borderTop: '1px dashed var(--line)' }}>
                                        <span className="muted" style={{ fontSize: 11.5 }}>Nguồn đối chiếu:</span>
                                        {item.sources.map((src, sIdx) => {
                                          const srcText = `${src.org}${src.title ? ` - ${src.title}` : ''}`
                                          if (src.url) {
                                            return (
                                              <a
                                                key={sIdx}
                                                href={src.url}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                className="pill ok"
                                                style={{ textDecoration: 'none', fontSize: 11, padding: '2px 8px', display: 'inline-flex', alignItems: 'center', gap: 4 }}
                                              >
                                                {srcText}
                                                <Icon id="i-link" className="icon sm" style={{ width: 10, height: 10 }} />
                                              </a>
                                            )
                                          }
                                          return (
                                            <span key={sIdx} className="pill" style={{ fontSize: 11, padding: '2px 8px' }}>
                                              {srcText}
                                            </span>
                                          )
                                        })}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}

                            <div className="muted" style={{ fontSize: 11.5, fontStyle: 'italic', marginTop: 16, borderTop: '1px solid var(--line)', paddingTop: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
                              <span>⚠️</span> <b>Tuyên bố miễn trừ:</b> Thông tin dùng để đối chiếu khách quan, không đại diện cho kết luận cuối cùng về tính đúng/sai.
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* ====================== ADVANCED ====================== */}
            {screen === 'advanced' && (
              <div className="screen">
                <div className="tabpane">
                  <div className="card card-pad crawler-card">
                    <div className="section-head compact">
                      <div>
                        <div className="section-label"><Icon id="i-link" className="icon sm ic" /> Thu thập dữ liệu từ URL</div>
                        <p className="muted" style={{ fontSize: 12, margin: '6px 0 0' }}>Hỗ trợ báo điện tử, YouTube và nguồn mạng xã hội qua Apify khi có token trong .env.</p>
                      </div>
                      <div className="row-actions">
                        <button className="btn sm ghost" onClick={handleSendCrawlToBatch} disabled={!crawlData}>
                          <Icon id="i-send" className="icon sm" /> Đưa vào batch
                        </button>
                        <button className="btn sm ghost" onClick={handleDownloadCrawlCsv} disabled={!crawlData}>
                          <Icon id="i-download" className="icon sm" /> CSV
                        </button>
                      </div>
                    </div>
                    <div className="crawl-form">
                      <input
                        className="input"
                        type="url"
                        placeholder="https://vnexpress.net/... hoặc https://www.youtube.com/watch?v=..."
                        value={crawlUrl}
                        onChange={e => setCrawlUrl(e.target.value)}
                      />
                      <input
                        className="input crawl-max"
                        type="number"
                        min={1}
                        max={50}
                        value={crawlMax}
                        onChange={e => setCrawlMax(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
                        aria-label="Số mục tối đa"
                      />
                      <button className="btn primary" onClick={handleCrawlUrl} disabled={crawlLoading || !crawlUrl.trim()}>
                        <Icon id="i-download" className="icon sm" /> {crawlLoading ? 'Đang thu thập…' : 'Thu thập'}
                      </button>
                    </div>
                    {crawlError && <div className="muted" style={{ color: 'var(--danger)', fontSize: 13, marginTop: 10 }}>{crawlError}</div>}
                    {crawlData && (
                      <div className="crawl-result">
                        <div className="ig-meta">
                          <span className="pill ok"><Icon id="i-check" className="icon sm" /> {crawlData.count} mục</span>
                          <span>Nguồn: <b>{crawlData.source_info}</b></span>
                          <span>Loại: <b className="mono">{crawlData.source_type}</b></span>
                        </div>
                        <div className="crawl-list">
                          {crawlData.texts.map((item, i) => (
                            <div className="crawl-item" key={`${i}-${item.slice(0, 20)}`}>
                              <div>
                                <span className="mono">#{i + 1}</span>
                                <p>{item.length > 240 ? item.slice(0, 240) + '…' : item}</p>
                              </div>
                              <button className="btn sm ghost" onClick={() => handleUseCrawledText(item)}>
                                <Icon id="i-analyze" className="icon sm" /> Phân tích
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="grid-2">
                    <div className="card card-pad">
                      <div className="field-label">Dữ liệu hàng loạt <span className="hint">phân tách bằng <span className="mono">---</span> hoặc xuống dòng</span></div>
                      <label className="file-drop">
                        <input
                          type="file"
                          accept=".csv,.tsv,.txt,.json,.jsonl,text/csv,text/tab-separated-values,text/plain,application/json"
                          onChange={handleFileUpload}
                        />
                        <span><Icon id="i-upload" className="icon sm" /> Nhập file dữ liệu</span>
                        <b>CSV, TSV, TXT, JSON, JSONL</b>
                      </label>
                      {fileStatus && <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>{fileStatus}</div>}
                      <textarea className="input" style={{ minHeight: 160 }} value={batchText} onChange={e => setBatchText(e.target.value)} />
                      <button className="btn primary block" style={{ marginTop: 18 }} onClick={handleBatch} disabled={batchLoading}>
                        <Icon id="i-send" className="icon sm" /> {batchLoading ? `Đang phân tích ${batchProgress}…` : 'Phân tích hàng loạt'}
                      </button>
                      <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>Chạy qua <span className="mono">/api/batch-analyze</span>, lưu DB. Tối đa 100 dòng/lần.</p>
                    </div>
                    <div className="card card-pad">
                      <div className="section-head compact">
                        <div className="section-label"><Icon id="i-data" className="icon sm ic" /> Kết quả {batchRows.length ? `· ${batchRows.length} dòng` : ''}</div>
                        <div className="row-actions">
                          <button className="btn sm ghost" onClick={handleDownloadBatchCsv} disabled={!batchRows.length}>
                            <Icon id="i-download" className="icon sm" /> CSV
                          </button>
                          <button className="btn sm ghost" onClick={handleDownloadBatchReport} disabled={!batchRows.length}>
                            <Icon id="i-download" className="icon sm" /> Báo cáo
                          </button>
                        </div>
                      </div>
                      {!batchRows.length ? (
                        <div className="empty-state" style={{ padding: '30px 16px' }}><div className="muted" style={{ fontSize: 13 }}>Chưa có kết quả. Bấm “Phân tích hàng loạt”.</div></div>
                      ) : (
                        <table className="dtable">
                          <thead><tr><th>#</th><th>ID</th><th>Trích đoạn</th><th>Dấu hiệu sai lệch</th><th>Lập trường</th><th>Cảm xúc</th><th>Báo cáo</th></tr></thead>
                          <tbody>
                            {batchRows.map(r => (
                              <tr key={r.i}>
                                <td className="mono">{r.i}</td>
                                <td className="mono">{r.id ?? '—'}</td>
                                <td style={{ maxWidth: 200 }}>{r.text.length > 60 ? r.text.slice(0, 60) + '…' : r.text}</td>
                                <td style={{ color: r.misinfo === 'Fake' ? 'var(--danger)' : r.misinfo === 'Real' ? 'var(--teal-strong)' : undefined }}>{VI.misinfo[r.misinfo] || r.misinfo}</td>
                                <td>{VI.stance[r.stance] || r.stance}</td>
                                <td>{VI.sentiment[r.sentiment] || r.sentiment}</td>
                                <td>
                                  <button className="icon-btn" title="Tải báo cáo Markdown" onClick={() => r.id && handleDownloadServerReport(r.id)} disabled={!r.id}>
                                    <Icon id="i-download" className="icon sm" />
                                  </button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ====================== HISTORY ====================== */}
            {screen === 'history' && (
              <div className="screen">
                <div className="tabpane">
                  <div className="card card-pad">
                    <div className="section-head compact">
                      <div>
                        <div className="section-label"><Icon id="i-data" className="icon sm ic" /> Lịch sử phân tích</div>
                        <p className="muted" style={{ fontSize: 12, margin: '6px 0 0' }}>Các bản ghi đã lưu trong database của api_service.</p>
                      </div>
                      <button className="btn sm ghost" onClick={() => setHistoryReloadKey(k => k + 1)} disabled={historyLoading}>
                        <Icon id="i-refresh" className="icon sm" /> Làm mới
                      </button>
                    </div>
                    {historyError && <div className="muted" style={{ color: 'var(--danger)', fontSize: 13, marginTop: 12 }}>{historyError}</div>}
                    {historyLoading ? (
                      <div className="streamwrap" style={{ marginTop: 16 }}>Đang tải lịch sử…<span className="cursor" /></div>
                    ) : !historyRows.length ? (
                      <div className="empty-state" style={{ padding: '36px 16px', marginTop: 16 }}>
                        <div className="muted" style={{ fontSize: 13 }}>Chưa có bản ghi. Hãy phân tích một văn bản hoặc chạy batch trước.</div>
                      </div>
                    ) : (
                      <table className="dtable history-table">
                        <thead><tr><th>ID</th><th>Thời gian</th><th>Nội dung</th><th>Dấu hiệu sai lệch</th><th>Lập trường</th><th>Cảm xúc</th><th>Thao tác</th></tr></thead>
                        <tbody>
                          {historyRows.map(row => (
                            <tr key={row.id}>
                              <td className="mono">#{row.id}</td>
                              <td>{row.created_at ? new Date(row.created_at).toLocaleString('vi-VN') : '—'}</td>
                              <td style={{ maxWidth: 280 }}>{row.source_text.length > 92 ? row.source_text.slice(0, 92) + '…' : row.source_text}</td>
                              <td style={{ color: row.misinfo_label === 'Fake' ? 'var(--danger)' : row.misinfo_label === 'Real' ? 'var(--teal-strong)' : undefined }}>{VI.misinfo[row.misinfo_label] || row.misinfo_label}</td>
                              <td>{VI.stance[row.stance_label] || row.stance_label}</td>
                              <td>{VI.sentiment[row.sentiment_label] || row.sentiment_label}</td>
                              <td>
                                <div className="row-actions">
                                  <button className="btn sm ghost" onClick={() => handleOpenHistoryItem(row)}>
                                    <Icon id="i-analyze" className="icon sm" /> Mở
                                  </button>
                                  <button className="icon-btn" title="Tải báo cáo Markdown" onClick={() => handleDownloadServerReport(row.id)}>
                                    <Icon id="i-download" className="icon sm" />
                                  </button>
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                </div>
              </div>
            )}

          </div>
        </div>
      </div>
    </>
  )
}

/* ============================================================
   RiskRadar — radar hồ sơ rủi ro P(Tin giả / Phản đối / Tiêu cực)
   ============================================================ */
interface AxisCardProps {
  task: string
  label: string
  score: number
  scaleHint: string
}

function AxisCard({ task, label, score, scaleHint }: AxisCardProps) {
  return (
    <div className="axis">
      <div className="cap">{task}</div>
      <div className={`val ${valClass(label)}`}>{VI[taskKey(task)][label] || label}</div>
      <div className={`meter ${meterClass(label)}`}><i style={{ width: `${(score * 100).toFixed(0)}%` }} /></div>
      <div className="scoreline"><span>{scaleHint}</span><span className="mono">{(score * 100).toFixed(1)}%</span></div>
    </div>
  )
}

function RiskRadar({ probs }: { probs?: { misinfo: Probs; stance: Probs; sentiment: Probs } }) {
  if (!probs) return <div className="muted" style={{ fontSize: 12 }}>Không có dữ liệu phân phối.</div>
  const data = [
    { axis: 'Tin giả', v: probs.misinfo?.Fake ?? 0, labelX: 100, labelY: 12, anchor: 'middle' as const },
    { axis: 'Phản đối', v: probs.stance?.Against ?? 0, labelX: 178, labelY: 154, anchor: 'middle' as const },
    { axis: 'Tiêu cực', v: probs.sentiment?.Negative ?? 0, labelX: 22, labelY: 154, anchor: 'middle' as const },
  ]
  const cx = 100
  const cy = 104
  const radius = 68
  const angles = [-90, 30, 150]
  const point = (angle: number, scale = 1) => {
    const rad = (Math.PI / 180) * angle
    return { x: cx + Math.cos(rad) * radius * scale, y: cy + Math.sin(rad) * radius * scale }
  }
  const ring = (scale: number) => angles.map(a => point(a, scale)).map(p => `${p.x},${p.y}`).join(' ')
  const area = data.map((d, i) => point(angles[i], Math.max(0, Math.min(1, d.v)))).map(p => `${p.x},${p.y}`).join(' ')
  return (
    <div className="risk-radar" data-testid="risk-radar" role="img" aria-label="Radar hồ sơ rủi ro PhoBERT">
      <svg viewBox="0 0 200 188" width="100%" height="188" aria-hidden="true">
        {[0.25, 0.5, 0.75, 1].map(scale => (
          <polygon key={scale} points={ring(scale)} fill="none" stroke="var(--line)" strokeWidth="1" />
        ))}
        {angles.map((a, i) => {
          const p = point(a)
          return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="var(--line-2)" strokeWidth="1" />
        })}
        <polygon points={area} fill="rgba(210,69,58,.24)" stroke="var(--danger)" strokeWidth="2" />
        {data.map((d, i) => {
          const p = point(angles[i], Math.max(0, Math.min(1, d.v)))
          return <circle key={d.axis} cx={p.x} cy={p.y} r="3.5" fill="var(--danger)" />
        })}
        {data.map(d => (
          <g key={d.axis}>
            <text x={d.labelX} y={d.labelY} textAnchor={d.anchor} className="radar-label">{d.axis}</text>
            <text x={d.labelX} y={d.labelY + 15} textAnchor={d.anchor} className="radar-value">{(d.v * 100).toFixed(1)}%</text>
          </g>
        ))}
      </svg>
    </div>
  )
}

/* helper: map nhãn axis card về key taxonomy */
function taskKey(task: string): string {
  if (task.includes('sai lệch') || task.includes('xác thực')) return 'misinfo'
  if (task.includes('Lập trường')) return 'stance'
  return 'sentiment'
}

function XaiPill({ status }: { status: string }) {
  if (status === 'done') return <span className="pill ok"><Icon id="i-check" className="icon sm" /> xai_status: done</span>
  if (status === 'pending') return <span className="pill warn"><Icon id="i-spark" className="icon sm" /> đang sinh…</span>
  if (status === 'failed') return <span className="pill danger"><Icon id="i-shield" className="icon sm" /> lỗi</span>
  return <span className="pill"><Icon id="i-spark" className="icon sm" /> chưa sinh</span>
}
