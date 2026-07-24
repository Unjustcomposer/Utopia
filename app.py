import streamlit as st
import pandas as pd
import jax

from config import SimulationConfig
from simulation_jax import run_simulation
from scenarios import SCENARIO_LIST
from report_generator import generate_pdf_report
from telematics_connectors import PhysicalShockCompiler

st.set_page_config(page_title="NexusAI Tariff Impact", page_icon="🚢", layout="wide")

st.title("NexusAI: Automated Tariff Impact Dashboard")
st.markdown("**(Simulated)** Enterprise Supply Chain Stress Tester with Native SAP/Oracle ERP Integration.")

# Sidebar Configuration
st.sidebar.header("Supply Chain Settings")

scenario_name = st.sidebar.selectbox("Stress Test Scenario", ["tariff_shock", "oil_shock", "pandemic", "baseline"], index=0)

agents = st.sidebar.slider("Global Supplier Nodes", min_value=10, max_value=1000, value=200, step=10)
firms = st.sidebar.slider("Retail Distribution Centers", min_value=2, max_value=50, value=5, step=1)
ticks = st.sidebar.slider("Forecast Horizon (Days)", min_value=20, max_value=200, value=60, step=10)
firm_mode = st.sidebar.selectbox("Inventory Policy AI", ["Heuristic", "LMM"], index=0)

st.sidebar.divider()
st.sidebar.header("Physical Telematics")
st.sidebar.markdown("Ground macroeconomic shocks in real-world physical reality.")

if 'telematics_multiplier' not in st.session_state:
    st.session_state.telematics_multiplier = 1.0
    st.session_state.telematics_summary = "Normal Operations"
    
if st.sidebar.button("📡 Ingest Live Weather & Shipping Delays", use_container_width=True):
    with st.spinner("Polling NOAA and Project44 APIs..."):
        compiler = PhysicalShockCompiler()
        mult, summary = compiler.compile_live_shock()
        st.session_state.telematics_multiplier = mult
        st.session_state.telematics_summary = summary
        
if st.session_state.telematics_multiplier > 1.0:
    st.sidebar.error(f"**Live Risk Detected!**\n\n{st.session_state.telematics_summary}\n\n*Applied Cost Multiplier: {st.session_state.telematics_multiplier:.2f}x*")
else:
    st.sidebar.success("All clear. No immediate weather or shipping risks.")

mode_map = {"LMM": 0, "Zero-Intelligence": 1, "Heuristic": 2}
config = SimulationConfig(
    num_agents=agents,
    num_firms=firms,
    num_ticks=ticks,
    firm_behavior_mode=mode_map[firm_mode]
)

if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = None

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Run Supply Chain Stress Test", type="primary", use_container_width=True):
        with st.spinner(f"Ingesting ERP State & Running '{scenario_name}' ..."):
            result = run_simulation(config=config, seed=42, scenario=scenario_name, telematics_multiplier=st.session_state.telematics_multiplier)
            st.session_state.metrics_history = result.metrics_history
            st.success("Simulation Complete!")

with col2:
    if st.session_state.metrics_history:
        # Generate PDF
        pdf_bytes = generate_pdf_report(scenario_name, config.__dict__, st.session_state.metrics_history)
        st.download_button(
            label="📄 Download 2-Page PDF Report",
            data=pdf_bytes,
            file_name=f"NexusAI_Report_{scenario_name}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# Display Results
if st.session_state.metrics_history:
    st.markdown("### Projected Supply Chain Impact")
    
    df = pd.DataFrame(st.session_state.metrics_history)
    df = df.set_index("tick")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("#### Projected Retail Output Volume")
        st.line_chart(df["total_output"], color="#3b82f6")
        
    with c2:
        st.markdown("#### Supplier Solvency Rate (%)")
        st.line_chart(df["employment_rate"], color="#10b981")
        
    with c3:
        st.markdown("#### Consumer Price Impact")
        st.line_chart(df["price_index"], color="#f43f5e")

    with st.expander("Raw Data Table"):
        st.dataframe(df)
else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to see results.")
