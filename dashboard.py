import json
import math
import socketserver
import http.server
import urllib.parse
from http import HTTPStatus
import threading

from config import SimulationConfig
from simulation import Simulation
from experiment import Experiment
from scenario import create_scenario
from search import StrategySearch, firm_profit_objective, total_welfare_objective, price_stability_objective

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NexusAI — Agent Economy Simulator</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-start: #0a0e1a;
            --bg-end: #141b2d;
            --panel-bg: rgba(255, 255, 255, 0.04);
            --panel-border: rgba(255, 255, 255, 0.08);
            --blue: #3b82f6;
            --green: #10b981;
            --amber: #f59e0b;
            --rose: #f43f5e;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, var(--bg-start), var(--bg-end));
            background-attachment: fixed;
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
            line-height: 1.5;
        }

        header {
            margin-bottom: 2rem;
            text-align: center;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1rem;
            font-weight: 400;
        }

        .container {
            display: grid;
            grid-template-columns: 1fr 3fr;
            gap: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }

        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.5rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            margin-bottom: 1.5rem;
        }

        .panel-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--panel-border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .badge-simulated {
            font-size: 0.75rem;
            background: rgba(245, 158, 11, 0.2);
            color: var(--amber);
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-weight: 500;
        }

        .form-group {
            margin-bottom: 1rem;
        }

        label {
            display: block;
            font-size: 0.875rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        input, select {
            width: 100%;
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--panel-border);
            color: var(--text-main);
            padding: 0.75rem;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-size: 0.875rem;
            transition: all 0.2s;
        }

        input:focus, select:focus {
            outline: none;
            border-color: var(--blue);
            background: rgba(0, 0, 0, 0.4);
        }

        .btn-group {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            margin-top: 1.5rem;
        }

        button {
            width: 100%;
            padding: 0.875rem;
            border: none;
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
            color: white;
            position: relative;
            overflow: hidden;
        }

        button:active {
            transform: scale(0.98);
        }

        .btn-run {
            background: linear-gradient(90deg, var(--blue), #2563eb);
        }
        .btn-run:hover { background: linear-gradient(90deg, #60a5fa, var(--blue)); box-shadow: 0 0 15px rgba(59, 130, 246, 0.4); }

        .btn-exp {
            background: linear-gradient(90deg, var(--green), #059669);
        }
        .btn-exp:hover { background: linear-gradient(90deg, #34d399, var(--green)); box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }

        .btn-search {
            background: linear-gradient(90deg, var(--rose), #e11d48);
        }
        .btn-search:hover { background: linear-gradient(90deg, #fb7185, var(--rose)); box-shadow: 0 0 15px rgba(244, 63, 94, 0.4); }

        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 1280px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }

        .chart-container {
            position: relative;
            height: 250px;
            width: 100%;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
            margin-top: 1rem;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--panel-border);
        }

        th {
            color: var(--text-muted);
            font-weight: 500;
        }

        tr:last-child td {
            border-bottom: none;
        }
        
        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .delta-positive { color: var(--green); }
        .delta-negative { color: var(--rose); }

        /* Loader */
        .loader-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(10, 14, 26, 0.8);
            backdrop-filter: blur(4px);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }
        
        .loader-overlay.active {
            opacity: 1;
            pointer-events: all;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid var(--panel-border);
            border-top-color: var(--blue);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        }

        @keyframes spin { 100% { transform: rotate(360deg); } }

        .hidden { display: none; }
    </style>
</head>
<body>

    <div class="loader-overlay" id="loader">
        <div class="spinner"></div>
        <div id="loader-text">Running Simulation...</div>
    </div>

    <header>
        <h1>NexusAI — Agent Economy Simulator</h1>
        <p class="subtitle">(simulated) — All results are simulation outcomes, not real-world predictions</p>
    </header>

    <div class="container">
        <div class="sidebar">
            <div class="panel">
                <div class="panel-title">Configuration</div>
                <div class="form-group">
                    <label>Agents</label>
                    <input type="number" id="cfg-agents" value="100">
                </div>
                <div class="form-group">
                    <label>Firms</label>
                    <input type="number" id="cfg-firms" value="5">
                </div>
                <div class="form-group">
                    <label>Goods</label>
                    <input type="number" id="cfg-goods" value="4">
                </div>
                <div class="form-group">
                    <label>Ticks (Duration)</label>
                    <input type="number" id="cfg-ticks" value="60">
                </div>
                <div class="form-group">
                    <label>Base Seed</label>
                    <input type="number" id="cfg-seed" value="42">
                </div>
                <div class="form-group" style="display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem;">
                    <input type="checkbox" id="cfg-us-calibration" style="width: auto; margin: 0;">
                    <label style="margin: 0;">Use US Calibration</label>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Scenario</div>
                <div class="form-group">
                    <label>Scenario Type</label>
                    <select id="scenario-type" onchange="updateScenarioParams()">
                        <option value="none">None (Baseline)</option>
                        <option value="marketing">Marketing Campaign</option>
                        <option value="supply_disruption">Supply Disruption</option>
                        <option value="demand_shock">Demand Shock</option>
                        <option value="trade_disruption">Trade Disruption</option>
                        <option value="feature_change">Feature Change</option>
                    </select>
                </div>
                <div id="scenario-params">
                    <!-- Dynamic inputs injected here -->
                </div>
                <div class="panel-title" style="margin-top: 1.5rem; font-size: 1.1rem;">Demographic Targeting</div>
                <div class="form-group">
                    <label>Target Region</label>
                    <select id="target-region">
                        <option value="All">All Regions</option>
                        <option value="Northeast">Northeast</option>
                        <option value="Midwest">Midwest</option>
                        <option value="South">South</option>
                        <option value="West">West</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Target Age Group</label>
                    <select id="target-age-group">
                        <option value="All">All Ages</option>
                        <option value="18-35">18-35</option>
                        <option value="36-50">36-50</option>
                        <option value="51-65">51-65</option>
                        <option value="65+">65+</option>
                    </select>
                </div>
            </div>

            <div class="panel">
                <div class="panel-title">Actions</div>
                <div class="form-group">
                    <label>Experiment Seeds</label>
                    <input type="number" id="cfg-seeds" value="5">
                </div>
                <div class="btn-group">
                    <button class="btn-run" onclick="runSingle()">Run Single Simulation</button>
                    <button class="btn-exp" onclick="runExperiment()">Run Experiment (A/B)</button>
                    <button class="btn-search" onclick="runSearch()">Run Strategy Search</button>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div id="results-panel" class="panel hidden">
                <div class="panel-title">
                    <span id="results-title">Simulation Results</span>
                    <span class="badge-simulated">(simulated)</span>
                </div>
                
                <div class="charts-grid">
                    <div class="chart-container"><canvas id="chart-price"></canvas></div>
                    <div class="chart-container"><canvas id="chart-employment"></canvas></div>
                    <div class="chart-container"><canvas id="chart-gini"></canvas></div>
                    <div class="chart-container"><canvas id="chart-output"></canvas></div>
                </div>

                <div id="report-container" class="hidden" style="margin-top: 2rem;">
                    <div class="panel-title">
                        <span>Metrics Report</span>
                        <span class="badge-simulated">(simulated)</span>
                    </div>
                    <div id="report-content"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Chart instances
        let charts = {};

        function initCharts() {
            const ctxPrice = document.getElementById('chart-price').getContext('2d');
            const ctxEmp = document.getElementById('chart-employment').getContext('2d');
            const ctxGini = document.getElementById('chart-gini').getContext('2d');
            const ctxOutput = document.getElementById('chart-output').getContext('2d');

            Chart.defaults.color = '#94a3b8';
            Chart.defaults.font.family = "'Inter', sans-serif";

            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'top', labels: { boxWidth: 12 } }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                elements: {
                    line: { tension: 0.3, borderWidth: 2 },
                    point: { radius: 0, hitRadius: 10, hoverRadius: 4 }
                }
            };

            charts.price = new Chart(ctxPrice, { type: 'line', data: { datasets: [] }, options: { ...commonOptions, plugins: { ...commonOptions.plugins, title: { display: true, text: 'Price Index' } } } });
            charts.employment = new Chart(ctxEmp, { type: 'line', data: { datasets: [] }, options: { ...commonOptions, plugins: { ...commonOptions.plugins, title: { display: true, text: 'Employment Rate' } } } });
            charts.gini = new Chart(ctxGini, { type: 'line', data: { datasets: [] }, options: { ...commonOptions, plugins: { ...commonOptions.plugins, title: { display: true, text: 'Gini Coefficient' } } } });
            charts.output = new Chart(ctxOutput, { type: 'line', data: { datasets: [] }, options: { ...commonOptions, plugins: { ...commonOptions.plugins, title: { display: true, text: 'Total Output' } } } });
        }

        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            updateScenarioParams();
        });

        // Scenario param schemas that match create_scenario kwargs
        const SCENARIO_SCHEMA = {
            marketing: [
                {name: 'start_tick', type: 'number', default: 10},
                {name: 'duration', type: 'number', default: 30},
                {name: 'target_good', type: 'number', default: 0},
                {name: 'spend', type: 'number', default: 5000},
                {name: 'reach', type: 'number', step: 0.1, default: 0.6},
                {name: 'awareness_boost', type: 'number', step: 0.1, default: 0.5}
            ],
            supply_disruption: [
                {name: 'start_tick', type: 'number', default: 15},
                {name: 'duration', type: 'number', default: 20},
                {name: 'target_firm', type: 'number', default: 0},
                {name: 'capacity_reduction', type: 'number', step: 0.1, default: 0.5},
                {name: 'cost_increase', type: 'number', step: 0.1, default: 1.5}
            ],
            demand_shock: [
                {name: 'start_tick', type: 'number', default: 15},
                {name: 'duration', type: 'number', default: 30},
                {name: 'risk_aversion_delta', type: 'number', step: 0.05, default: 0.2},
                {name: 'savings_rate_delta', type: 'number', step: 0.01, default: 0.05}
            ],
            trade_disruption: [
                {name: 'start_tick', type: 'number', default: 15},
                {name: 'duration', type: 'number', default: 25},
                {name: 'tariff_rate', type: 'number', step: 0.1, default: 0.3},
                {name: 'affected_goods_fraction', type: 'number', step: 0.1, default: 0.5}
            ],
            feature_change: [
                {name: 'start_tick', type: 'number', default: 10},
                {name: 'duration', type: 'number', default: 40},
                {name: 'target_good', type: 'number', default: 0},
                {name: 'new_price', type: 'number', step: 1, default: 12},
                {name: 'new_quality', type: 'number', step: 0.1, default: 1.2}
            ]
        };

        function updateScenarioParams() {
            const type = document.getElementById('scenario-type').value;
            const container = document.getElementById('scenario-params');
            container.innerHTML = '';
            
            if (type === 'none' || !SCENARIO_SCHEMA[type]) return;

            SCENARIO_SCHEMA[type].forEach(param => {
                const step = param.step ? `step="${param.step}"` : '';
                container.innerHTML += `
                    <div class="form-group" style="margin-left: 1rem; border-left: 2px solid var(--panel-border); padding-left: 1rem;">
                        <label>${param.name.replace(/_/g, ' ').toUpperCase()}</label>
                        <input type="${param.type}" id="param-${param.name}" value="${param.default}" ${step} class="scenario-param" data-name="${param.name}">
                    </div>
                `;
            });
        }

        function getBaseConfig() {
            return {
                agents: parseInt(document.getElementById('cfg-agents').value),
                firms: parseInt(document.getElementById('cfg-firms').value),
                goods: parseInt(document.getElementById('cfg-goods').value),
                ticks: parseInt(document.getElementById('cfg-ticks').value),
                seed: parseInt(document.getElementById('cfg-seed').value),
                use_us_calibration: document.getElementById('cfg-us-calibration').checked,
                target_region: document.getElementById('target-region').value,
                target_age_group: document.getElementById('target-age-group').value
            };
        }

        function getScenarioConfig() {
            const type = document.getElementById('scenario-type').value;
            if (type === 'none') return null;
            
            const params = {};
            document.querySelectorAll('.scenario-param').forEach(input => {
                const val = parseFloat(input.value);
                params[input.dataset.name] = Number.isInteger(val) && !input.step ? parseInt(input.value) : val;
            });
            
            return { type, params };
        }

        function showLoader(text) {
            document.getElementById('loader-text').innerText = text;
            document.getElementById('loader').classList.add('active');
        }

        function hideLoader() {
            document.getElementById('loader').classList.remove('active');
        }

        // Extract arrays from list-of-dicts metrics_history
        function extractMetricArrays(history) {
            const result = { price_index: [], employment_rate: [], gini: [], total_output: [] };
            for (const tick of history) {
                result.price_index.push(tick.price_index || 0);
                result.employment_rate.push(tick.employment_rate || 0);
                result.gini.push(tick.gini || 0);
                result.total_output.push(tick.total_output || 0);
            }
            return result;
        }

        function renderSingleCharts(history) {
            const metrics = extractMetricArrays(history);
            const ticks = Array.from({length: metrics.price_index.length}, (_, i) => i);
            
            const updateChart = (chart, data, label, color) => {
                chart.data.labels = ticks;
                chart.data.datasets = [{
                    label: label,
                    data: data,
                    borderColor: color,
                    backgroundColor: color + '20',
                    fill: true
                }];
                chart.update();
            };

            updateChart(charts.price, metrics.price_index, 'Price Index', '#3b82f6');
            updateChart(charts.employment, metrics.employment_rate, 'Employment', '#10b981');
            updateChart(charts.gini, metrics.gini, 'Gini', '#f59e0b');
            updateChart(charts.output, metrics.total_output, 'Output', '#a855f7');
            
            // Show demographic cross-tabs for the final tick
            const finalTick = history[history.length - 1];
            if (finalTick.welfare_by_region && Object.keys(finalTick.welfare_by_region).length > 1) {
                let html = `<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem;">`;
                // Region Table
                html += `<div><div style="font-weight: 600; margin-bottom: 0.5rem;">By Region</div><table>
                    <tr><th>Region</th><th>Avg Welfare</th><th>Employment</th></tr>`;
                for (const [r, w] of Object.entries(finalTick.welfare_by_region)) {
                    const emp = finalTick.employment_by_region[r] || 0;
                    html += `<tr><td>${r}</td><td>$${w.toFixed(2)}</td><td>${(emp*100).toFixed(1)}%</td></tr>`;
                }
                html += `</table></div>`;
                // Age Table
                html += `<div><div style="font-weight: 600; margin-bottom: 0.5rem;">By Age Group</div><table>
                    <tr><th>Age Group</th><th>Avg Welfare</th><th>Employment</th></tr>`;
                for (const [a, w] of Object.entries(finalTick.welfare_by_age)) {
                    const emp = finalTick.employment_by_age[a] || 0;
                    html += `<tr><td>${a}</td><td>$${w.toFixed(2)}</td><td>${(emp*100).toFixed(1)}%</td></tr>`;
                }
                html += `</table></div></div>`;
                
                document.getElementById('report-content').innerHTML = html;
                document.getElementById('report-container').classList.remove('hidden');
            } else {
                document.getElementById('report-container').classList.add('hidden');
            }
            document.getElementById('results-panel').classList.remove('hidden');
        }

        function renderExperimentCharts(controlList, treatmentList) {
            const ctrl = extractMetricArrays(controlList);
            const treat = extractMetricArrays(treatmentList);
            const ticks = Array.from({length: ctrl.price_index.length}, (_, i) => i);
            
            const updateChart = (chart, dataC, dataT) => {
                chart.data.labels = ticks;
                chart.data.datasets = [
                    { label: 'Control', data: dataC, borderColor: '#94a3b8', borderDash: [5, 5] },
                    { label: 'Treatment', data: dataT, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true }
                ];
                chart.update();
            };

            updateChart(charts.price, ctrl.price_index, treat.price_index);
            updateChart(charts.employment, ctrl.employment_rate, treat.employment_rate);
            updateChart(charts.gini, ctrl.gini, treat.gini);
            updateChart(charts.output, ctrl.total_output, treat.total_output);
        }

        function renderDeltasTable(deltas) {
            let html = `<table>
                <tr><th>Metric</th><th>Delta</th><th>CI (95%)</th><th>Effect Size</th><th>Status</th></tr>`;
            
            for (const [metric, data] of Object.entries(deltas)) {
                const isSig = data.significant ? "Significant" : "Not Sig.";
                const sigColor = data.significant ? 'var(--green)' : 'var(--text-muted)';
                const colorCls = data.mean_delta > 0 ? "delta-positive" : "delta-negative";
                html += `<tr>
                    <td>${metric.replace(/_/g, ' ')}</td>
                    <td class="${colorCls}">${data.mean_delta.toFixed(3)}</td>
                    <td>[${data.ci_lower.toFixed(3)}, ${data.ci_upper.toFixed(3)}]</td>
                    <td>d = ${data.effect_size.toFixed(3)}</td>
                    <td style="color: ${sigColor}">${isSig}</td>
                </tr>`;
            }
            html += `</table>`;
            
            const rc = document.getElementById('report-container');
            document.getElementById('report-content').innerHTML = html;
            rc.classList.remove('hidden');
            document.getElementById('results-panel').classList.remove('hidden');
        }

        function renderSearchTable(candidates, robustness) {
            let html = `<table>
                <tr><th>Rank</th><th>Parameters</th><th>Objective</th><th>CI</th></tr>`;
            
            candidates.forEach(cand => {
                const paramStr = Object.entries(cand.params).map(([k,v]) => `${k}=${v}`).join(', ');
                html += `<tr>
                    <td>#${cand.rank}</td>
                    <td><code style="font-size:0.8rem">${paramStr}</code></td>
                    <td>${cand.objective_mean.toFixed(2)}</td>
                    <td>[${cand.ci_lower.toFixed(2)}, ${cand.ci_upper.toFixed(2)}]</td>
                </tr>`;
            });
            html += `</table>`;

            if (robustness && robustness.length > 0) {
                html += `<div style="margin-top:1.5rem"><div class="panel-title"><span>Robustness Validation</span><span class="badge-simulated">(simulated)</span></div>`;
                html += `<table><tr><th>Params</th><th>Search</th><th>Validation</th><th>Degradation</th><th>Status</th></tr>`;
                robustness.forEach(rc => {
                    const paramStr = Object.entries(rc.params).map(([k,v]) => `${k}=${v}`).join(', ');
                    const status = rc.is_overfit ? '<span style="color:var(--rose)">OVERFIT</span>' : '<span style="color:var(--green)">ROBUST</span>';
                    html += `<tr>
                        <td><code style="font-size:0.8rem">${paramStr}</code></td>
                        <td>${rc.search_mean.toFixed(2)}</td>
                        <td>${rc.validation_mean.toFixed(2)}</td>
                        <td>${rc.degradation_ratio.toFixed(2)}x</td>
                        <td>${status}</td>
                    </tr>`;
                });
                html += `</table></div>`;
            }
            
            const rc = document.getElementById('report-container');
            document.getElementById('report-content').innerHTML = html;
            rc.classList.remove('hidden');
            
            document.querySelector('.charts-grid').classList.add('hidden');
            document.getElementById('results-panel').classList.remove('hidden');
        }

        async function runSingle() {
            showLoader("Running Baseline Simulation...");
            try {
                const reqData = getBaseConfig();
                const res = await fetch('/api/run', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(reqData)
                });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                
                document.getElementById('results-title').innerText = "Baseline Results";
                document.querySelector('.charts-grid').classList.remove('hidden');
                renderSingleCharts(data.metrics_history);
            } catch (err) {
                console.error(err);
                alert("Error: " + err.message);
            }
            hideLoader();
        }

        async function runExperiment() {
            const sc = getScenarioConfig();
            if (!sc) {
                alert("Please select a scenario type for an experiment.");
                return;
            }
            
            showLoader("Running Experiment (Control vs Treatment)... This may take 30-60s");
            try {
                const reqData = {
                    ...getBaseConfig(),
                    seeds: parseInt(document.getElementById('cfg-seeds').value),
                    scenario_type: sc.type,
                    scenario_params: sc.params
                };
                
                const res = await fetch('/api/experiment', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(reqData)
                });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                
                document.getElementById('results-title').innerText = "Experiment: " + sc.type.replace(/_/g, ' ');
                document.querySelector('.charts-grid').classList.remove('hidden');
                
                renderExperimentCharts(data.control_metrics, data.treatment_metrics);
                renderDeltasTable(data.deltas);
            } catch (err) {
                console.error(err);
                alert("Error: " + err.message);
            }
            hideLoader();
        }

        async function runSearch() {
            const sc = getScenarioConfig();
            if (!sc) {
                alert("Please select a scenario to optimize.");
                return;
            }

            showLoader("Running Strategy Search... This may take 1-3 minutes");
            try {
                const reqData = {
                    ...getBaseConfig(),
                    scenario_type: sc.type,
                    scenario_params: sc.params,
                    seeds_per_eval: 4,
                    objective: 'welfare',
                    method: 'grid'
                };
                
                const res = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(reqData)
                });
                if (!res.ok) throw new Error(await res.text());
                const data = await res.json();
                
                document.getElementById('results-title').innerText = "Strategy Search: " + (data.objective || sc.type);
                renderSearchTable(data.candidates, data.robustness_checks);
            } catch (err) {
                console.error(err);
                alert("Error: " + err.message);
            }
            hideLoader();
        }
    </script>
</body>
</html>
"""

def sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

class DashboardHandler(http.server.BaseHTTPRequestHandler):
    def _set_headers(self, content_type='application/json'):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging

    def do_GET(self):
        if self.path == '/':
            self._set_headers('text/html')
            self.wfile.write(DASHBOARD_HTML.encode('utf-8'))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")

    def do_OPTIONS(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        
        try:
            req_json = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
            return

        if self.path == '/api/run':
            self.handle_api_run(req_json)
        elif self.path == '/api/experiment':
            self.handle_api_experiment(req_json)
        elif self.path == '/api/search':
            self.handle_api_search(req_json)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")

    def handle_api_run(self, data):
        try:
            config = SimulationConfig(
                num_agents=data.get('agents', 200),
                num_firms=data.get('firms', 5),
                num_goods=data.get('goods', 4),
                num_ticks=data.get('ticks', 120),
                use_us_calibration=data.get('use_us_calibration', False)
            )
            sim = Simulation(config=config, seed=data.get('seed', 42))
            result = sim.run()
            
            response = {
                'metrics_history': result.metrics_history,
                'summary': result.summary()
            }
            self._set_headers()
            self.wfile.write(json.dumps(sanitize_for_json(response)).encode('utf-8'))
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def handle_api_experiment(self, data):
        try:
            config = SimulationConfig(
                num_agents=data.get('agents', 200),
                num_firms=data.get('firms', 5),
                num_goods=data.get('goods', 4),
                num_ticks=data.get('ticks', 90),
                use_us_calibration=data.get('use_us_calibration', False)
            )
            scenario_type = data.get('scenario_type', 'marketing')
            scenario_params = data.get('scenario_params', {})
            # Add targeting params if present
            if 'target_region' in data:
                scenario_params['target_region'] = data['target_region']
            if 'target_age_group' in data:
                scenario_params['target_age_group'] = data['target_age_group']
                
            scenario = create_scenario(scenario_type, **scenario_params)
            
            exp = Experiment(
                config=config, 
                scenario=scenario, 
                num_seeds=data.get('seeds', 10), 
                base_seed=data.get('seed', 42)
            )
            result = exp.run()
            
            # Serialize control and treatment metric time series
            control_series = []
            treatment_series = []
            for cr in result.control_results:
                control_series.append(cr.metrics_history)
            for tr in result.treatment_results:
                treatment_series.append(tr.metrics_history)

            # Average metrics across seeds for charting
            num_ticks = len(control_series[0]) if control_series else 0
            avg_control = []
            avg_treatment = []
            for t in range(num_ticks):
                ctrl_tick = {}
                treat_tick = {}
                for key in ['gini', 'price_index', 'employment_rate', 'total_output', 'total_welfare']:
                    ctrl_vals = [cs[t].get(key, 0) for cs in control_series if t < len(cs)]
                    treat_vals = [ts[t].get(key, 0) for ts in treatment_series if t < len(ts)]
                    ctrl_tick[key] = sum(ctrl_vals) / len(ctrl_vals) if ctrl_vals else 0
                    treat_tick[key] = sum(treat_vals) / len(treat_vals) if treat_vals else 0
                ctrl_tick['tick'] = t
                treat_tick['tick'] = t
                avg_control.append(ctrl_tick)
                avg_treatment.append(treat_tick)

            # Serialize metric deltas
            deltas = {}
            for name, md in result.metric_deltas.items():
                deltas[name] = {
                    'mean_delta': md.mean_delta,
                    'ci_lower': md.ci_lower,
                    'ci_upper': md.ci_upper,
                    'effect_size': md.effect_size,
                    'p_value': md.p_value,
                    'significant': md.significant,
                }

            response = {
                'control_metrics': avg_control,
                'treatment_metrics': avg_treatment,
                'deltas': deltas,
                'scenario': result.scenario_description,
                'num_seeds': result.num_seeds,
                'runtime': result.runtime_seconds,
            }
            self._set_headers()
            self.wfile.write(json.dumps(sanitize_for_json(response)).encode('utf-8'))
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    def handle_api_search(self, data):
        try:
            config = SimulationConfig(
                num_agents=data.get('agents', 150),
                num_firms=data.get('firms', 5),
                num_goods=data.get('goods', 4),
                num_ticks=data.get('ticks', 90),
                use_us_calibration=data.get('use_us_calibration', False)
            )

            scenario_type = data.get('scenario_type', 'feature_change')
            
            # Use appropriate default param space depending on the scenario
            default_param_space = {}
            if scenario_type == 'marketing':
                default_param_space = {'spend': [1000, 5000, 10000], 'reach': [0.2, 0.5, 0.8]}
            elif scenario_type == 'demand_shock':
                default_param_space = {'risk_aversion_delta': [0.05, 0.1, 0.2]}
            elif scenario_type == 'supply_disruption':
                default_param_space = {'capacity_reduction': [0.2, 0.5, 0.8]}
            elif scenario_type == 'trade_disruption':
                default_param_space = {'cost_increase': [1.2, 1.5, 2.0]}
            else:
                default_param_space = {'price': [8, 10, 12, 14, 16, 18, 20]}

            param_space = data.get('param_space', default_param_space)

            def scenario_factory(params):
                sp = data.get('scenario_params', {}).copy()
                sp.update(params)
                
                if 'target_region' in data:
                    sp['target_region'] = data['target_region']
                if 'target_age_group' in data:
                    sp['target_age_group'] = data['target_age_group']
                if 'start_tick' not in sp:
                    sp['start_tick'] = 15
                if 'duration' not in sp:
                    sp['duration'] = 50
                    
                if scenario_type == 'feature_change':
                    if 'target_good' not in sp:
                        sp['target_good'] = 0
                    if 'price' in sp:
                        sp['new_price'] = sp.pop('price')
                
                return create_scenario(scenario_type, **sp)
            
            obj_name = data.get('objective', 'profit')
            if obj_name == 'profit':
                objective = firm_profit_objective(firm_id=0)
            elif obj_name == 'price_stability':
                objective = price_stability_objective()
            else:
                objective = total_welfare_objective()

            search = StrategySearch(
                config=config,
                scenario_factory=scenario_factory,
                param_space=param_space,
                objective=objective,
                method=data.get('method', 'grid'),
                num_seeds_per_eval=data.get('seeds_per_eval', 5),
                validation_num_seeds=data.get('validation_seeds', 10),
                top_k_validate=data.get('top_k', 3),
            )
            
            result = search.run()
            
            candidates = []
            for c in result.candidates[:20]:
                candidates.append({
                    'rank': c.rank,
                    'params': c.params,
                    'objective_mean': c.objective_mean,
                    'ci_lower': c.objective_ci_lower,
                    'ci_upper': c.objective_ci_upper,
                })

            robustness = []
            for rc in result.robustness_checks:
                robustness.append({
                    'params': rc.params,
                    'search_mean': rc.search_mean,
                    'validation_mean': rc.validation_mean,
                    'degradation_ratio': rc.degradation_ratio,
                    'robustness_score': rc.robustness_score,
                    'is_overfit': rc.is_overfit,
                    'ks_p_value': rc.ks_p_value,
                })
                
            response = {
                'candidates': candidates,
                'robustness_checks': robustness,
                'objective': result.objective_name,
                'method': result.method,
                'runtime': result.runtime_seconds,
            }
            self._set_headers()
            self.wfile.write(json.dumps(sanitize_for_json(response)).encode('utf-8'))
        except Exception as e:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))


def start_dashboard(port=8765):
    """Launch the interactive web dashboard."""
    server = http.server.HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f"==========================================================")
    print(f"  NexusAI Dashboard running at http://localhost:{port}")
    print(f"  All results are simulated, not real-world predictions")
    print(f"==========================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    start_dashboard(port)

