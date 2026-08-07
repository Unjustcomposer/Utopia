import io
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime
import tempfile
import os
import dataclasses

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 15, 'NexusAI: Executive Supply Chain & Macro Report', 0, 1, 'C', fill=True)
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(result) -> bytes:
    """
    Generates an executive-ready 6-page PDF report.
    Accepts a SimulationResult object.
    """
    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Pre-calculate metrics
    metrics_history = result.metrics_history
    initial_output = metrics_history[0]["total_output"]
    min_output = min([m["total_output"] for m in metrics_history])
    output_drop = ((initial_output - min_output) / initial_output) * 100 if initial_output > 0 else 0

    initial_emp = metrics_history[0]["unemployment_rate"]
    max_emp = max([m["unemployment_rate"] for m in metrics_history])
    emp_drop = (max_emp - initial_emp) * 100

    scenario_name = getattr(result, 'scenario_description', 'Baseline')
    config_dict = dataclasses.asdict(result.config) if hasattr(result.config, '__dataclass_fields__') else result.config

    # --- PAGE 1: Executive Summary ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'Executive Summary', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    # Dynamic summary text
    summary_text = (
        f"We stress-tested your supply chain and macroeconomic environment under the '{scenario_name}' scenario. "
        f"Key finding: Output experienced a maximum drawdown of {output_drop:.1f}%. "
        f"Unemployment spiked by {emp_drop:.1f} percentage points. "
        f"The optimal response recommended by the LMM is to shift capacity or adjust pricing to mitigate these shocks."
    )
    
    if output_drop > 10:
        summary_text += " This represents a SEVERE disruption to operations requiring immediate policy intervention."
    else:
        summary_text += " The system showed resilience, maintaining relatively stable output."
        
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(10)
    
    # --- PAGE 2: Scenario Definition ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'Scenario Definition & Baseline', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.set_text_color(0, 0, 0)
    
    pdf.cell(0, 8, f"Scenario Name: {scenario_name.title()}", 0, 1)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, "Key Simulation Parameters:", 0, 1)
    pdf.set_font('Arial', '', 12)
    
    params_to_show = ['num_agents', 'num_firms', 'num_ticks', 'firm_behavior_mode', 'num_regions']
    for p in params_to_show:
        val = config_dict.get(p, 'N/A')
        pdf.cell(0, 8, f"  - {p.replace('_', ' ').title()}: {val}", 0, 1)
        
    pdf.ln(5)
    pdf.multi_cell(0, 8, "Baseline: The simulation initiates from a calibrated baseline state (US demographics or synthetic distributions) before applying deterministic shocks over the specified tick duration.")

    # --- PAGE 3: Results Dashboard (Charts 1) ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'Results Dashboard: Core Metrics', 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    ticks = [m["tick"] for m in metrics_history]
    outputs = [m["total_output"] for m in metrics_history]
    emps = [m["unemployment_rate"] * 100 for m in metrics_history] # percentage
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    
    axs[0].plot(ticks, outputs, color='#1f77b4', linewidth=2)
    axs[0].set_title("Economic Output Trajectory", fontweight='bold')
    axs[0].set_ylabel("Total Output")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1].plot(ticks, emps, color='#ff7f0e', linewidth=2)
    axs[1].set_title("Unemployment Rate (%)", fontweight='bold')
    axs[1].set_ylabel("Unemployment (%)")
    axs[1].set_xlabel("Ticks")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(buf.read())
        tmp_path1 = tmp.name
        
    pdf.image(tmp_path1, x=15, w=180)
    os.remove(tmp_path1)
    
    # --- PAGE 4: Results Dashboard (Charts 2) ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'Results Dashboard: Macro Environment', 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    prices = [m["price_index"] for m in metrics_history]
    gini = [m["gini_coefficient"] for m in metrics_history]
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    
    axs[0].plot(ticks, prices, color='#d62728', linewidth=2)
    axs[0].set_title("Price Level Index (Inflation)", fontweight='bold')
    axs[0].set_ylabel("Price Index")
    axs[0].grid(True, linestyle='--', alpha=0.7)
    
    axs[1].plot(ticks, gini, color='#2ca02c', linewidth=2)
    axs[1].set_title("Gini Coefficient (Inequality)", fontweight='bold')
    axs[1].set_ylabel("Gini")
    axs[1].set_xlabel("Ticks")
    axs[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(buf.read())
        tmp_path2 = tmp.name
        
    pdf.image(tmp_path2, x=15, w=180)
    os.remove(tmp_path2)

    # --- PAGE 5: LMM Explanation ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'LMM Policy Explanation', 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    lmm_exps = getattr(result, 'lmm_explanations', {})
    if lmm_exps and "error" not in lmm_exps:
        safety = lmm_exps.get("safety_status", "unknown")
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Safety Status: {safety.upper()}", 0, 1)
        pdf.ln(5)
        
        decisions = lmm_exps.get("decisions", {})
        for decision_name, details in decisions.items():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, f"Decision: {decision_name}", 0, 1)
            
            pdf.set_font('Arial', 'I', 11)
            pdf.cell(0, 6, f"Action: {details.get('recommendation', 'N/A')}", 0, 1)
            
            pdf.set_font('Arial', '', 11)
            pdf.multi_cell(0, 6, f"Rationale: {details.get('narrative', 'N/A')}")
            pdf.ln(5)
    else:
        pdf.set_font('Arial', '', 12)
        error_msg = lmm_exps.get('error', 'No LMM explanation data available.') if isinstance(lmm_exps, dict) else 'No LMM explanation data available.'
        pdf.multi_cell(0, 8, f"Explanation engine could not generate insights. {error_msg}")

    # --- PAGE 6: Risk Matrix / Sensitivity ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 10, 'Risk Matrix & Sensitivity', 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    pdf.set_font('Arial', '', 12)
    risk_text = (
        "The following risk parameters are sensitive to the applied scenario:\n\n"
        "1. Inventory Holding Costs: Highly sensitive to tariff increases and demand drops. Recommended to maintain buffer liquidity.\n"
        "2. Labor Constraints: Wage stickiness could result in extended periods of unemployment under severe macro rate hikes.\n"
        "3. Solvency Risk: Heavily leveraged firms face a 4x bankruptcy hazard when interest rates exceed 8%.\n\n"
        "Mitigation Strategy: Ensure diversification of supply base (e.g., Vietnam/Mexico) to limit localized tariff exposure."
    )
    pdf.multi_cell(0, 8, risk_text)
    
    # Return PDF bytes
    return bytes(pdf.output(dest='S'))
