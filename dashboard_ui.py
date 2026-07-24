import jwt
import os

JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
TOKEN = jwt.encode({"sub": "admin", "tenant_id": "tenant_1"}, JWT_SECRET, algorithm="HS256")

DASHBOARD_HTML = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NexusAI Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0d1117;
            --panel-bg: rgba(22, 27, 34, 0.7);
            --text-color: #c9d1d9;
            --accent-color: #58a6ff;
            --border-color: #30363d;
            --hover-color: #1f6feb;
            --success: #2ea043;
            --error: #f85149;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, sans-serif;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 2rem;
            min-height: 100vh;
            background-image: 
                radial-gradient(circle at 15% 50%, rgba(88, 166, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 85% 30%, rgba(46, 160, 67, 0.08) 0%, transparent 50%);
        }}
        
        .header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .header h1 {{
            font-size: 2rem;
            font-weight: 600;
            background: linear-gradient(90deg, #58a6ff, #79c0ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .grid {{
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
        }}
        
        .panel {{
            background: var(--panel-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            margin-bottom: 1.5rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .panel:hover {{
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            border-color: #484f58;
        }}
        
        .panel h2 {{
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
            color: #fff;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.5rem;
        }}
        
        .form-group {{
            margin-bottom: 1rem;
        }}
        
        .form-group label {{
            display: block;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
            color: #8b949e;
        }}
        
        .form-group input, .form-group select {{
            width: 100%;
            padding: 0.75rem;
            background: #010409;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-color);
            outline: none;
            transition: border-color 0.2s;
        }}
        
        .form-group input:focus, .form-group select:focus {{
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1);
        }}
        
        button {{
            width: 100%;
            padding: 0.75rem;
            background-color: var(--border-color);
            color: #fff;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s, transform 0.1s;
            margin-top: 1rem;
        }}
        
        button.primary {{
            background-color: var(--success);
        }}
        
        button.primary:hover {{
            background-color: #3fb950;
        }}
        
        button.accent {{
            background-color: #238636;
        }}
        
        button.accent:hover {{
            background-color: #2ea043;
        }}
        
        button:hover {{
            background-color: #484f58;
        }}
        
        button:active {{
            transform: scale(0.98);
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            width: 100%;
        }}
        
        #status {{
            margin-top: 1rem;
            padding: 0.75rem;
            border-radius: 6px;
            font-size: 0.875rem;
            display: none;
        }}
        
        .status-success {{
            background-color: rgba(46, 160, 67, 0.1);
            color: var(--success);
            border: 1px solid rgba(46, 160, 67, 0.4);
        }}
        
        .status-error {{
            background-color: rgba(248, 81, 73, 0.1);
            color: var(--error);
            border: 1px solid rgba(248, 81, 73, 0.4);
        }}
        
        .loader {{
            display: none;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
            margin: 0 auto;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: rgba(1, 4, 9, 0.5);
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 1.5rem;
            font-weight: 600;
            color: #fff;
            margin: 0.5rem 0;
        }}
        
        .metric-label {{
            font-size: 0.75rem;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>NexusAI Dashboard</h1>
        <div style="color: #8b949e; font-size: 0.875rem;">Status: <span style="color: var(--success)">● Online</span></div>
    </div>
    
    <div id="status"></div>

    <div class="grid">
        <!-- Controls Column -->
        <div class="controls">
            <div class="panel">
                <h2>Simulation Parameters</h2>
                <form id="runForm">
                    <div class="form-group">
                        <label for="agents">Agents</label>
                        <input type="number" id="agents" value="200" min="10" max="1000">
                    </div>
                    <div class="form-group">
                        <label for="firms">Firms</label>
                        <input type="number" id="firms" value="5" min="1" max="50">
                    </div>
                    <div class="form-group">
                        <label for="ticks">Ticks (Duration)</label>
                        <input type="number" id="ticks" value="120" min="10" max="1000">
                    </div>
                    <button type="submit" class="primary" id="runBtn">
                        <span class="btn-text">Run Simulation</span>
                        <div class="loader" id="runLoader"></div>
                    </button>
                </form>
            </div>
            
            <div class="panel">
                <h2>Advanced Actions</h2>
                <button id="baselineBtn" style="margin-top: 0">
                    <span class="btn-text">Ingest Global Baseline</span>
                    <div class="loader" id="baselineLoader"></div>
                </button>
                
                <h3 style="margin-top: 1.5rem; margin-bottom: 1rem; font-size: 1rem; color: #fff; border-bottom: 1px solid var(--border-color); padding-bottom: 0.5rem;">Explain Firm Policy</h3>
                <form id="explainForm">
                    <button type="submit" id="explainBtn" style="margin-top: 0">
                        <span class="btn-text">Run Explanation</span>
                        <div class="loader" id="explainLoader"></div>
                    </button>
                </form>
                <div id="explainResult" style="margin-top: 1rem; font-size: 0.85rem; font-family: monospace; white-space: pre-wrap; color: #8b949e;"></div>
            </div>
        </div>
        
        <!-- Visualization Column -->
        <div class="visualization">
            <div class="panel">
                <h2>Simulation Results</h2>
                
                <div class="metrics-grid" id="metricsGrid" style="display: none;">
                    <div class="metric-card">
                        <div class="metric-label">Avg Gini</div>
                        <div class="metric-value" id="valGini">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Unemployment</div>
                        <div class="metric-value" id="valUnemp">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Price Index</div>
                        <div class="metric-value" id="valPrice">--</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Welfare</div>
                        <div class="metric-value" id="valWelfare">--</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <canvas id="resultsChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        let chartInstance = null;
        
        function showStatus(msg, isError=false) {{
            const el = document.getElementById('status');
            el.textContent = msg;
            el.style.display = 'block';
            el.className = isError ? 'status-error' : 'status-success';
            setTimeout(() => {{ el.style.display = 'none'; }}, 5000);
        }}
        
        function setLoading(btnId, loaderId, isLoading) {{
            const btn = document.getElementById(btnId);
            const text = btn.querySelector('.btn-text');
            const loader = document.getElementById(loaderId);
            
            if (isLoading) {{
                btn.disabled = true;
                text.style.display = 'none';
                loader.style.display = 'block';
            }} else {{
                btn.disabled = false;
                text.style.display = 'block';
                loader.style.display = 'none';
            }}
        }}

        const getAuthHeader = () => {{
            return {{'Authorization': 'Bearer {TOKEN}'}}; 
        }};

        document.getElementById('runForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            setLoading('runBtn', 'runLoader', true);
            
            try {{
                const req = {{
                    agents: parseInt(document.getElementById('agents').value),
                    firms: parseInt(document.getElementById('firms').value),
                    goods: 4,
                    ticks: parseInt(document.getElementById('ticks').value)
                }};
                
                const res = await fetch('/api/run', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json', ...getAuthHeader()}},
                    body: JSON.stringify(req)
                }});
                
                if (!res.ok) throw new Error(await res.text());
                
                const data = await res.json();
                showStatus('Simulation completed successfully');
                
                // Update metrics
                document.getElementById('metricsGrid').style.display = 'grid';
                document.getElementById('valGini').textContent = data.summary.mean_gini.toFixed(3);
                document.getElementById('valUnemp').textContent = (data.summary.mean_unemployment * 100).toFixed(1) + '%';
                document.getElementById('valPrice').textContent = data.summary.mean_price_index.toFixed(2);
                document.getElementById('valWelfare').textContent = data.summary.mean_welfare.toFixed(1);
                
                // Update Chart
                updateChart(data.metrics_history);
                
            }} catch (err) {{
                showStatus('Error: ' + err.message, true);
            }} finally {{
                setLoading('runBtn', 'runLoader', false);
            }}
        }});
        
        document.getElementById('baselineBtn').addEventListener('click', async () => {{
            setLoading('baselineBtn', 'baselineLoader', true);
            try {{
                const res = await fetch('/api/ingest_global_baseline', {{ 
                    method: 'POST',
                    headers: getAuthHeader()
                }});
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                showStatus(data.message);
            }} catch (err) {{
                showStatus('Error: ' + err.message, true);
            }} finally {{
                setLoading('baselineBtn', 'baselineLoader', false);
            }}
        }});
        
        document.getElementById('explainForm').addEventListener('submit', async (e) => {{
            e.preventDefault();
            setLoading('explainBtn', 'explainLoader', true);
            document.getElementById('explainResult').textContent = '';
            
            try {{
                const req = {{
                    demand_history: [10, 11, 10.5],
                    profit_history: [100, 105, 102],
                    price_history: [10, 10, 10],
                    macro_price_history: [10, 10.1, 10.2],
                    macro_rate_history: [0.05, 0.05, 0.05]
                }};
                
                const res = await fetch('/api/explain', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json', ...getAuthHeader()}},
                    body: JSON.stringify(req)
                }});
                
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                document.getElementById('explainResult').textContent = JSON.stringify(data, null, 2);
            }} catch (err) {{
                showStatus('Error: ' + err.message, true);
            }} finally {{
                setLoading('explainBtn', 'explainLoader', false);
            }}
        }});
        
        function updateChart(history) {{
            const ctx = document.getElementById('resultsChart').getContext('2d');
            
            const ticks = history.map(h => h.tick);
            const unemp = history.map(h => h.unemployment_rate);
            const gini = history.map(h => h.gini_coefficient);
            
            if (chartInstance) {{
                chartInstance.destroy();
            }}
            
            Chart.defaults.color = '#8b949e';
            Chart.defaults.borderColor = '#30363d';
            
            chartInstance = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: ticks,
                    datasets: [
                        {{
                            label: 'Unemployment Rate',
                            data: unemp,
                            borderColor: '#f85149',
                            backgroundColor: 'rgba(248, 81, 73, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true,
                            yAxisID: 'y'
                        }},
                        {{
                            label: 'Gini Coefficient',
                            data: gini,
                            borderColor: '#58a6ff',
                            backgroundColor: 'rgba(88, 166, 255, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            fill: true,
                            yAxisID: 'y1'
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false,
                    }},
                    plugins: {{
                        legend: {{
                            position: 'top',
                            labels: {{
                                usePointStyle: true,
                                boxWidth: 6
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{ display: false }}
                        }},
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{ display: true, text: 'Unemployment' }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{ display: true, text: 'Gini' }},
                            grid: {{ drawOnChartArea: false }}
                        }}
                    }}
                }}
            }});
        }}
        
        // Init empty chart
        updateChart([{{tick: 0, unemployment_rate: 0, gini_coefficient: 0}}]);
    </script>
</body>
</html>
"""
