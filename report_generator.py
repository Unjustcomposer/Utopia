import io
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'NexusAI: Macroeconomic Scenario Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(scenario_name: str, config_dict: dict, metrics_history: list) -> bytes:
    """
    Generates a 2-page PDF report summarizing the simulation.
    Returns the PDF as bytes.
    """
    pdf = ReportPDF()
    pdf.add_page()
    
    # --- PAGE 1: Assumptions & Headline Results ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Simulation Assumptions', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 8, f"Scenario Template: {scenario_name.replace('_', ' ').title()}", 0, 1)
    pdf.cell(0, 8, f"Number of Agents: {config_dict.get('num_agents', 0)}", 0, 1)
    pdf.cell(0, 8, f"Number of Firms: {config_dict.get('num_firms', 0)}", 0, 1)
    pdf.cell(0, 8, f"Duration (Ticks): {config_dict.get('num_ticks', 0)}", 0, 1)
    pdf.cell(0, 8, f"Firm Behavior Mode: {config_dict.get('firm_behavior_mode', 2)}", 0, 1)
    pdf.ln(10)
    
    # Calculate Results
    initial_output = metrics_history[0]["total_output"]
    min_output = min([m["total_output"] for m in metrics_history])
    output_drop = ((initial_output - min_output) / initial_output) * 100 if initial_output > 0 else 0

    initial_emp = metrics_history[0]["employment_rate"]
    min_emp = min([m["employment_rate"] for m in metrics_history])
    emp_drop = (initial_emp - min_emp) * 100
    
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Headline Results', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    pdf.set_text_color(200, 0, 0) if output_drop > 5 else pdf.set_text_color(0, 150, 0)
    pdf.cell(0, 8, f"Maximum Output Drawdown: {output_drop:.2f}%", 0, 1)
    
    pdf.set_text_color(200, 0, 0) if emp_drop > 5 else pdf.set_text_color(0, 150, 0)
    pdf.cell(0, 8, f"Maximum Employment Drop: {emp_drop:.2f}% (points)", 0, 1)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 6, "This report was generated using the NexusAI differentiable macroeconomic engine. "
                         "Results are strictly deterministic given the selected scenario shocks and assumptions.")
    
    # --- PAGE 2: Charts ---
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Macroeconomic Trajectory', 0, 1)
    
    # Generate Matplotlib chart in memory
    ticks = [m["tick"] for m in metrics_history]
    outputs = [m["total_output"] for m in metrics_history]
    prices = [m["price_index"] for m in metrics_history]
    emps = [m["employment_rate"] for m in metrics_history]
    
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    axs[0].plot(ticks, outputs, color='blue', label='Total Output')
    axs[0].set_title("Economic Output")
    axs[0].grid(True)
    
    axs[1].plot(ticks, emps, color='green', label='Employment Rate')
    axs[1].set_title("Employment Rate")
    axs[1].grid(True)
    
    axs[2].plot(ticks, prices, color='red', label='Price Index (Inflation)')
    axs[2].set_title("Price Level")
    axs[2].grid(True)
    
    plt.tight_layout()
    
    # Save to BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    
    # fpdf allows inserting images directly from a file-like object in recent versions,
    # or we can write to a temp file. To be safe, we'll write to a temp file.
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(buf.read())
        tmp_path = tmp.name
        
    pdf.image(tmp_path, x=15, w=180)
    os.remove(tmp_path)
    
    # Return PDF bytes
    return bytes(pdf.output(dest='S'))
