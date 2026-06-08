import { useState, useEffect } from 'react'
import axios from 'axios'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

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
}
interface AttrData { pred_class: number; pred_label: string; embedding_layer: string; tokens: { token: string; score: number }[] }
interface BatchRow { i: number; text: string; misinfo: string; stance: string; sentiment: string }

const VI: Record<string, Record<string, string>> = {
  misinfo: { Fake: 'Tin giả', Real: 'Tin thật' },
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
}

const PRESETS = [
  { label: 'Tin giả cực đoan', text: 'Cảnh báo: vắc xin COVID có thể gây vô sinh ở phụ nữ và biến đổi gen ở trẻ em. Mọi người nên tìm hiểu kỹ trước khi làm chuột bạch cho các tập đoàn dược phẩm.' },
  { label: 'Ủng hộ tiêm chủng', text: 'Em cũng đang tiêm từng mũi 1 cho con, chậm mà đủ và an toàn cho con là được. Trộm vía bé chưa sốt, chưa hành mũi nào.' },
  { label: 'Thông tin chuẩn', text: 'Bộ Y tế khuyến cáo trẻ em từ 6 tháng tuổi cần tiêm đủ các mũi vaccine cơ bản theo Chương trình Tiêm chủng Mở rộng để phòng các bệnh truyền nhiễm nguy hiểm.' },
]

const SCREENS = [
  { id: 'analyze', icon: 'i-analyze', label: 'Phân tích văn bản', crumb: 'Phân tích · Đa nhiệm' },
  { id: 'advanced', icon: 'i-advanced', label: 'Công cụ nâng cao', crumb: 'Phân tích · Nâng cao' },
] as const
type ScreenId = typeof SCREENS[number]['id']

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

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('vnlp-theme', theme)
  }, [theme])

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
    setAnalysis(p => p ? { ...p, xai_status: 'pending', xai_explanation: { parse_ok: true, reasoning: '' } } : p)
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
              setAnalysis(p => p ? { ...p, xai_status: 'done', xai_explanation: { parse_ok: ev.parse_ok, reasoning: ev.reasoning, raw_output: ev.raw_output, disagreement: ev.disagreement, gemma_labels: ev.gemma_labels } } : p)
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

  /* ---- Batch: chạy THẬT bằng cách lặp /api/analyze ---- */
  const handleBatch = async () => {
    const parts = batchText.includes('---') ? batchText.split('---') : batchText.split('\n')
    const lines = parts.map(s => s.trim()).filter(Boolean).slice(0, 30)
    if (!lines.length) return
    setBatchLoading(true); setBatchRows([])
    const rows: BatchRow[] = []
    for (let i = 0; i < lines.length; i++) {
      setBatchProgress(`${i + 1}/${lines.length}`)
      try {
        const { data } = await axios.post<Analysis>(`${API_URL}/api/analyze`, { text: lines[i] })
        rows.push({ i: i + 1, text: lines[i], misinfo: data.misinfo_label, stance: data.stance_label, sentiment: data.sentiment_label })
      } catch { rows.push({ i: i + 1, text: lines[i], misinfo: '—', stance: '—', sentiment: '—' }) }
      setBatchRows([...rows])
    }
    setBatchProgress(''); setBatchLoading(false)
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
                                <div className="vmain">{fake ? 'Tin giả' : 'Tin thật'}</div>
                                <div className="vmeta">Độ tin cậy mô hình <b className="mono">{(analysis.misinfo_score * 100).toFixed(1)}%</b></div>
                              </div>
                              <div className="vside">
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
                            <AxisCard task="Tính xác thực" label={analysis.misinfo_label} score={analysis.misinfo_score} scaleHint="Ngưỡng 50%" />
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
                              {(['plausible', 'unusual', 'high_risk'] as const).map(k => {
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
                              <button className="btn sm" onClick={handleExplain} disabled={explaining || analysis.xai_status === 'pending'}>
                                <Icon id="i-spark" className="icon sm" /> {explaining ? 'Đang sinh…' : analysis.xai_status === 'done' ? 'Sinh lại' : 'Sinh giải thích (Gemma, chậm)'}
                              </button>
                            </div>
                          </div>
                          <p className="muted" style={{ fontSize: 12, margin: '2px 0 8px' }}>Nhịp 1 · PhoBERT trả nhãn tức thời → Nhịp 2 · Gemma-4B stream lý giải CoT theo token <span className="mono">(/api/explain-stream)</span></p>

                          <div className="tabs">
                            <button className={`tab ${tab === 'cot' ? 'on' : ''}`} onClick={() => setTab('cot')}><Icon id="i-spark" className="icon sm" /> Chain-of-Thought</button>
                            <button className={`tab ${tab === 'cap' ? 'on' : ''}`} onClick={() => setTab('cap')}><Icon id="i-analyze" className="icon sm" /> Token Attribution</button>
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
                                          {([['misinfo', 'Tính xác thực', analysis.misinfo_label], ['stance', 'Lập trường', analysis.stance_label], ['sentiment', 'Cảm xúc', analysis.sentiment_label]] as const).map(([key, vi, pLabel]) => {
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
                                Mức đóng góp của từng token vào nhãn <b style={{ color: analysis.misinfo_label === 'Fake' ? 'var(--danger)' : 'var(--teal-strong)' }}>“{VI.misinfo[analysis.misinfo_label]}”</b> (Integrated Gradients).
                              </p>
                              {!attrData && !attrLoading && (
                                <button className="btn sm" onClick={handleAttribute}><Icon id="i-analyze" className="icon sm" /> Tính token attribution (IG · chậm)</button>
                              )}
                              {attrLoading && <div className="streamwrap">Đang tính Integrated Gradients…<span className="cursor" /></div>}
                              {attrError && <div className="muted" style={{ color: 'var(--danger)', fontSize: 13 }}>{attrError}</div>}
                              {attrData && (
                                <>
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
                  <div className="grid-2">
                    <div className="card card-pad">
                      <div className="field-label">Dán nhiều dòng (phân tách bằng <span className="mono">---</span> hoặc xuống dòng)</div>
                      <textarea className="input" style={{ minHeight: 160 }} value={batchText} onChange={e => setBatchText(e.target.value)} />
                      <button className="btn primary block" style={{ marginTop: 18 }} onClick={handleBatch} disabled={batchLoading}>
                        <Icon id="i-send" className="icon sm" /> {batchLoading ? `Đang phân tích ${batchProgress}…` : 'Phân tích hàng loạt'}
                      </button>
                      <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>Chạy thật qua <span className="mono">/api/analyze</span> từng dòng (tối đa 30).</p>
                    </div>
                    <div className="card card-pad">
                      <div className="section-label" style={{ marginBottom: 14 }}><Icon id="i-data" className="icon sm ic" /> Kết quả {batchRows.length ? `· ${batchRows.length} dòng` : ''}</div>
                      {!batchRows.length ? (
                        <div className="empty-state" style={{ padding: '30px 16px' }}><div className="muted" style={{ fontSize: 13 }}>Chưa có kết quả. Bấm “Phân tích hàng loạt”.</div></div>
                      ) : (
                        <table className="dtable">
                          <thead><tr><th>#</th><th>Trích đoạn</th><th>Xác thực</th><th>Lập trường</th><th>Cảm xúc</th></tr></thead>
                          <tbody>
                            {batchRows.map(r => (
                              <tr key={r.i}>
                                <td className="mono">{r.i}</td>
                                <td style={{ maxWidth: 200 }}>{r.text.length > 60 ? r.text.slice(0, 60) + '…' : r.text}</td>
                                <td style={{ color: r.misinfo === 'Fake' ? 'var(--danger)' : r.misinfo === 'Real' ? 'var(--teal-strong)' : undefined }}>{VI.misinfo[r.misinfo] || r.misinfo}</td>
                                <td>{VI.stance[r.stance] || r.stance}</td>
                                <td>{VI.sentiment[r.sentiment] || r.sentiment}</td>
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
    { axis: 'Tin giả', v: probs.misinfo?.Fake ?? 0 },
    { axis: 'Phản đối', v: probs.stance?.Against ?? 0 },
    { axis: 'Tiêu cực', v: probs.sentiment?.Negative ?? 0 },
  ]
  return (
    <div style={{ height: 188 }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={data} outerRadius="70%">
          <PolarGrid stroke="var(--line)" />
          <PolarAngleAxis dataKey="axis" tick={{ fill: 'var(--ink-2)', fontSize: 12 }} />
          <PolarRadiusAxis domain={[0, 1]} tick={false} axisLine={false} />
          <Radar dataKey="v" stroke="#d2453a" fill="#d2453a" fillOpacity={0.25} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

/* helper: map nhãn axis card về key taxonomy */
function taskKey(task: string): string {
  if (task.includes('xác thực')) return 'misinfo'
  if (task.includes('Lập trường')) return 'stance'
  return 'sentiment'
}

function XaiPill({ status }: { status: string }) {
  if (status === 'done') return <span className="pill ok"><Icon id="i-check" className="icon sm" /> xai_status: done</span>
  if (status === 'pending') return <span className="pill warn"><Icon id="i-spark" className="icon sm" /> đang sinh…</span>
  if (status === 'failed') return <span className="pill danger"><Icon id="i-shield" className="icon sm" /> lỗi</span>
  return <span className="pill"><Icon id="i-spark" className="icon sm" /> chưa sinh</span>
}
