import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Activity, Layers, Play, History } from 'lucide-react'
import { useAuth0 } from '@auth0/auth0-react'

export default function App() {
  const { isAuthenticated, loginWithRedirect, logout, isLoading, getAccessTokenSilently } = useAuth0()
  
  const [scenarios, setScenarios] = useState<string[]>([])
  const [profiles, setProfiles] = useState<string[]>([])
  
  const [scenario, setScenario] = useState('baseline')
  const [agents, setAgents] = useState(200)
  const [ticks, setTicks] = useState(120)
  
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<any>(null)

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
        body: JSON.stringify({ scenario, agents, ticks })
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
        baselineGini: results.baseline.metrics_history[i].gini_index,
        scenarioGini: results.scenario.metrics_history[i].gini_index,
        baselineUnemployment: results.baseline.metrics_history[i].unemployment_rate,
        scenarioUnemployment: results.scenario.metrics_history[i].unemployment_rate,
        baselinePrice: results.baseline.metrics_history[i].price_index,
        scenarioPrice: results.scenario.metrics_history[i].price_index,
      })
    }
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
        <h1><Activity style={{display: 'inline', verticalAlign: 'middle', marginRight: 8}}/> Utopia Scenario Builder</h1>
        <button onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })} style={{ background: 'transparent', border: '1px solid var(--border)', padding: '0.5rem 1rem', cursor: 'pointer' }}>
          Log Out
        </button>
      </header>

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
            <select>
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
          
          <button onClick={runSimulation} disabled={loading} style={{width: '100%', marginTop: '1rem'}}>
            {loading ? 'Running JAX Engine...' : <><Play size={16} style={{display:'inline', verticalAlign:'middle'}}/> Run Comparison</>}
          </button>
        </div>

        <div className="panel">
          <h2><History style={{display: 'inline', verticalAlign: 'middle', marginRight: 8}}/> History & Diff</h2>
          {results ? (
            <div>
              <p>Comparison complete.</p>
              <p>Baseline Final Gini: <strong>{results.baseline.summary.final_gini.toFixed(3)}</strong></p>
              <p>Scenario Final Gini: <strong>{results.scenario.summary.final_gini.toFixed(3)}</strong></p>
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
              <div className="value">{(results.baseline.summary.avg_unemployment * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <h3>Avg Unemployment (Scenario)</h3>
              <div className="value">{(results.scenario.summary.avg_unemployment * 100).toFixed(1)}%</div>
            </div>
            <div className="metric-card">
              <h3>Final Price Index (Scenario)</h3>
              <div className="value">{results.scenario.summary.final_price_index.toFixed(2)}</div>
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
            <h3>LMM Explanation (Mock)</h3>
            <p style={{color: 'var(--text-secondary)', marginTop: '0.5rem'}}>
              The scenario shows that the tariff shock led to an increase in input costs, forcing firms to raise prices (Price Index: {results.scenario.summary.final_price_index.toFixed(2)} vs {results.baseline.summary.final_price_index.toFixed(2)}). 
              This caused aggregate demand to drop, increasing unemployment to {(results.scenario.summary.avg_unemployment * 100).toFixed(1)}%.
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
