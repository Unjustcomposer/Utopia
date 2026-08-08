import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Layers, Play, History } from 'lucide-react'

export default function App() {
  // Bypassing Auth0 for local dev
  const isAuthenticated = true
  const isLoading = false
  const getAccessTokenSilently = async () => 'mock-token'
  const loginWithRedirect = () => {}
  const logout = () => { console.log('Mock logout') }
  
  const [scenarios, setScenarios] = useState<string[]>([])
  const [profiles, setProfiles] = useState<string[]>([])
  
  const [scenario, setScenario] = useState('baseline')
  const [calibrationProfile, setCalibrationProfile] = useState('')
  const [agents, setAgents] = useState(200)
  const [ticks, setTicks] = useState(120)
  
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)

  // Academic / Research Mode (Phase 4.3)
  const [isAcademicMode, setIsAcademicMode] = useState(false)
  const [firmLearningRate, setFirmLearningRate] = useState(0.01)
  const [dmpMatchEfficiency, setDmpMatchEfficiency] = useState(0.5)
  const [baseSavingsRate, setBaseSavingsRate] = useState(0.1)

  // Consultant Workspace (Phase 4.1)
  const [view, setView] = useState<'scenario' | 'consultant'>('scenario')
  
  const dummyPortfolios = [
    { id: 1, name: "Acme Corp Supply Chain", clients: 12, lastRun: "2 hrs ago" },
    { id: 2, name: "Global Logistics Inc", clients: 8, lastRun: "1 day ago" },
    { id: 3, name: "FinTech Banking Sector", clients: 24, lastRun: "3 days ago" }
  ]

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = await getAccessTokenSilently()
        const headers = { Authorization: `Bearer ${token}` }
        fetch('/api/scenarios', { headers }).then(r => r.json()).then(setScenarios).catch(console.error)
        fetch('/api/calibration_profiles', { headers }).then(r => r.json()).then(setProfiles).catch(console.error)
      } catch (e) {
        console.error("Failed to fetch initial data", e)
      }
    }
    if (isAuthenticated) {
      fetchData()
    }
  }, [isAuthenticated, getAccessTokenSilently])

  const runSimulation = async () => {
    setLoading(true)
    try {
      const token = await getAccessTokenSilently()
      const res = await fetch('/api/run/compare', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          scenario, 
          agents, 
          ticks,
          ...(isAcademicMode && {
            matching_efficiency: dmpMatchEfficiency,
            savings_rate_min: baseSavingsRate,
          }),
          ...(calibrationProfile && { calibration_profile: calibrationProfile })
        })
      })
      const data = await res.json()
      setResults(data)
    } catch (e) {
      console.error(e)
      alert("Error running simulation.")
    } finally {
      setLoading(false)
    }
  }

  // Process data for charts
  let chartData: any[] = []
  if (results && results.baseline && results.scenario) {
    const len = results.baseline.metrics_history.length
    for (let i = 0; i < len; i++) {
      chartData.push({
        tick: i,
        baselineGini: results.baseline.metrics_history[i].gini_coefficient,
        scenarioGini: results.scenario.metrics_history[i].gini_coefficient,
        baselineUnemployment: results.baseline.metrics_history[i].unemployment_rate,
        scenarioUnemployment: results.scenario.metrics_history[i].unemployment_rate,
        baselinePrice: results.baseline.metrics_history[i].price_index,
        scenarioPrice: results.scenario.metrics_history[i].price_index,
      })
    }
  }

  const exportToCSV = () => {
    if (!results) return
    const csvRows = ['Tick,Baseline Gini,Scenario Gini,Baseline Unemployment,Scenario Unemployment,Baseline Price,Scenario Price']
    chartData.forEach(row => {
      csvRows.push(`${row.tick},${row.baselineGini},${row.scenarioGini},${row.baselineUnemployment},${row.scenarioUnemployment},${row.baselinePrice},${row.scenarioPrice}`)
    })
    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'simulation_results.csv'
    a.click()
  }

  const exportToJSON = () => {
    if (!results) return
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'simulation_results.json'
    a.click()
  }

  if (isLoading) {
    return (
      <div className="app-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <p style={{ color: 'var(--text-secondary)' }}>Loading Utopia...</p>
      </div>
    )
  }

  if (!isAuthenticated) {
    return (
      <div className="app-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <div className="panel" style={{ maxWidth: '400px', textAlign: 'center', padding: '3rem' }}>
          <Activity size={48} style={{ color: 'var(--primary)', marginBottom: '1rem' }} />
          <h1 style={{ marginBottom: '1rem', fontSize: '1.5rem', fontWeight: 600 }}>Utopia</h1>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem', lineHeight: 1.5 }}>
            Enterprise Supply Chain Simulation
          </p>
          <button onClick={() => loginWithRedirect()} style={{ width: '100%', padding: '0.75rem', fontWeight: 500 }}>
            Log In
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="app-container">
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <h1><Activity style={{display: 'inline', verticalAlign: 'middle', marginRight: 8}}/> Utopia</h1>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button 
              onClick={() => setView('scenario')} 
              style={{ background: view === 'scenario' ? 'var(--primary-color)' : 'transparent', border: '1px solid var(--border)' }}
            >
              Scenario Builder
            </button>
            <button 
              onClick={() => setView('consultant')} 
              style={{ background: view === 'consultant' ? 'var(--primary-color)' : 'transparent', border: '1px solid var(--border)' }}
            >
              Consultant Workspace
            </button>
          </div>
        </div>
        <div className="header-controls">
          <label className="toggle-switch">
            <input 
              type="checkbox" 
              checked={isAcademicMode} 
              onChange={e => setIsAcademicMode(e.target.checked)} 
            />
            Academic Mode
          </label>
          <button onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })} style={{ background: 'transparent', border: '1px solid var(--border)', padding: '0.5rem 1rem', cursor: 'pointer' }}>
            Log Out
          </button>
        </div>
      </header>

      {view === 'consultant' ? (
        <div className="consultant-workspace">
          <div className="panel">
            <h2>Client Portfolios</h2>
            <p style={{ color: 'var(--text-secondary)' }}>Select a portfolio to manage client-specific simulations.</p>
            <div className="portfolio-list">
              {dummyPortfolios.map(p => (
                <div key={p.id} className="portfolio-card">
                  <h3>{p.name}</h3>
                  <p>{p.clients} Active Clients</p>
                  <p style={{ marginTop: '0.5rem', fontSize: '0.75rem', opacity: 0.7 }}>Last Run: {p.lastRun}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <>
      <div className="grid-2">
        <div className="panel">
          <h2><Layers style={{display: 'inline', verticalAlign: 'middle', marginRight: 8}}/> Parameters</h2>
          <div className="form-group">
            <label>Shock Scenario</label>
            <select value={scenario} onChange={e => setScenario(e.target.value)}>
              {scenarios.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Calibration Profile (Base Economy)</label>
            <select value={calibrationProfile} onChange={e => setCalibrationProfile(e.target.value)}>
              <option value="">Default JAX Initialization</option>
              {profiles.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label>Number of Agents</label>
              <input type="number" value={agents} onChange={e => setAgents(Number(e.target.value))} />
            </div>
            <div className="form-group">
              <label>Simulation Ticks</label>
              <input type="number" value={ticks} onChange={e => setTicks(Number(e.target.value))} />
            </div>
          </div>
          
          {isAcademicMode && (
            <div className="grid-2" style={{ marginTop: '1rem', padding: '1rem', background: '#111318', borderRadius: '8px', border: '1px solid var(--border)' }}>
              <div className="form-group">
                <label>Firm Learning Rate</label>
                <input type="number" step="0.001" value={firmLearningRate} onChange={e => setFirmLearningRate(Number(e.target.value))} />
              </div>
              <div className="form-group">
                <label>DMP Match Efficiency</label>
                <input type="number" step="0.01" value={dmpMatchEfficiency} onChange={e => setDmpMatchEfficiency(Number(e.target.value))} />
              </div>
              <div className="form-group">
                <label>Base Savings Rate</label>
                <input type="number" step="0.01" value={baseSavingsRate} onChange={e => setBaseSavingsRate(Number(e.target.value))} />
              </div>
            </div>
          )}
          
          <button onClick={runSimulation} disabled={loading} style={{width: '100%', marginTop: '1rem'}}>
            {loading ? 'Running JAX Engine...' : <><Play size={16} style={{display:'inline', verticalAlign:'middle'}}/> Run Comparison</>}
          </button>
        </div>

        <div className="panel">
          <h2><History style={{display: 'inline', verticalAlign: 'middle', marginRight: 8}}/> History & Diff</h2>
          {results ? (
            <div>
              <p>Comparison complete.</p>
              <p>Baseline Final Gini: <strong>{results.baseline.summary.mean_gini.toFixed(3)}</strong></p>
              <p>Scenario Final Gini: <strong>{results.scenario.summary.mean_gini.toFixed(3)}</strong></p>
            </div>
          ) : (
            <p style={{color: 'var(--text-secondary)'}}>Run a simulation to see results.</p>
          )}
        </div>
      </div>

      {results && (
        <div className="panel">
          <h2>Results Dashboard</h2>
          
          <div className="metrics-grid">
            <div className="metric-card">
              <h3>Avg Unemployment (Baseline)</h3>
              <div className="value">{(results.baseline.summary.mean_unemployment * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <h3>Avg Unemployment (Scenario)</h3>
              <div className="value">{(results.scenario.summary.mean_unemployment * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <h3>Final Price Index (Scenario)</h3>
              <div className="value">{results.scenario.summary.mean_price_index.toFixed(2)}</div>
            </div>
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2e3340" />
                <XAxis dataKey="tick" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{backgroundColor: '#1a1d24', borderColor: '#2e3340', color: '#e2e8f0'}} />
                <Legend />
                <Line type="monotone" dataKey="baselineGini" stroke="#94a3b8" name="Baseline Gini" dot={false} strokeDasharray="5 5" />
                <Line type="monotone" dataKey="scenarioGini" stroke="#ef4444" name="Scenario Gini" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="baselineUnemployment" stroke="#3b82f6" name="Baseline Unemployment" dot={false} strokeDasharray="5 5" />
                <Line type="monotone" dataKey="scenarioUnemployment" stroke="#22c55e" name="Scenario Unemployment" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          <div className="explanation">
            <h3>LMM Policy Explanation</h3>
            <p style={{color: 'var(--text-secondary)', marginTop: '0.5rem'}}>
              The scenario shows that the tariff shock led to an increase in input costs, forcing firms to raise prices (Price Index: {results.scenario.summary.mean_price_index.toFixed(2)} vs {results.baseline.summary.mean_price_index.toFixed(2)}). 
              This caused aggregate demand to drop, increasing unemployment to {(results.scenario.summary.mean_unemployment * 100).toFixed(1)}%.
            </p>
          </div>
          
          <div className="export-buttons">
            <button className="export-btn" onClick={exportToCSV}>Export to CSV</button>
            <button className="export-btn" onClick={exportToJSON}>Export to JSON</button>
          </div>
        </div>
      )}
        </>
      )}
    </div>
  )
}
