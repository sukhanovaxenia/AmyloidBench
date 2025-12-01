"""
HTML report generator for amyloidogenicity prediction analysis.

This module creates interactive, self-contained HTML reports that
combine visualizations, tables, and interpretive text into a
comprehensive analysis document suitable for sharing and archiving.

Report Types
------------

**Single Sequence Report**
Comprehensive analysis of one protein including:
- Sequence information and statistics
- Per-residue score profiles from multiple predictors
- Consensus APR regions with annotations
- Polymorph classification with probabilities
- Structural features summary

**Benchmark Report**
Systematic comparison of predictors including:
- Performance metrics table with rankings
- ROC and PR curve comparisons
- Per-residue accuracy analysis
- Statistical significance tests

Design Philosophy
-----------------
Reports are generated as self-contained HTML files with:
- Embedded CSS for consistent styling
- Inline SVG/PNG figures (no external dependencies)
- Collapsible sections for complex data
- Responsive layout for different screen sizes
- Print-friendly formatting
- Accessible color schemes
"""

from __future__ import annotations

import base64
import html
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Check for visualization dependencies
try:
    from .plots import (
        HAS_MATPLOTLIB,
        plot_multi_predictor_profile,
        plot_polymorph_probabilities,
        plot_benchmark_comparison,
        figure_to_base64,
        PlotStyle,
    )
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# HTML Templates
# =============================================================================

HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --background-color: #f8f9fa;
            --card-background: #ffffff;
            --text-color: #333333;
            --border-color: #dee2e6;
            --success-color: #27ae60;
            --warning-color: #f39c12;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        
        header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        header .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        header .meta {{ margin-top: 15px; font-size: 0.9em; opacity: 0.8; }}
        
        .card {{
            background: var(--card-background);
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 25px;
        }}
        
        .card h2 {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .card h3 {{ color: var(--secondary-color); margin: 20px 0 15px 0; }}
        
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background-color: var(--primary-color); color: white; font-weight: 600; }}
        tr:hover {{ background-color: rgba(52, 152, 219, 0.1); }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-card .value {{ font-size: 2em; font-weight: bold; }}
        .metric-card .label {{ font-size: 0.9em; opacity: 0.9; margin-top: 5px; }}
        
        .sequence-box {{
            font-family: 'Courier New', monospace;
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            word-break: break-all;
            font-size: 0.9em;
            line-height: 1.8;
        }}
        
        .apr-highlight {{
            background-color: #ffeb3b;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
        
        .figure {{ text-align: center; margin: 20px 0; }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .figure-caption {{ font-style: italic; color: #666; margin-top: 10px; font-size: 0.9em; }}
        
        .region-list {{ list-style: none; }}
        
        .region-item {{
            background-color: #f8f9fa;
            border-left: 4px solid var(--secondary-color);
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        
        .region-item .position {{ font-weight: bold; color: var(--primary-color); }}
        .region-item .sequence {{ font-family: monospace; background: white; padding: 5px 10px; border-radius: 3px; margin: 5px 0; display: inline-block; }}
        
        .tag {{ display: inline-block; padding: 3px 10px; border-radius: 15px; font-size: 0.85em; margin: 2px; }}
        .tag-positive {{ background-color: #e8f5e9; color: var(--success-color); }}
        .tag-negative {{ background-color: #ffebee; color: var(--accent-color); }}
        .tag-neutral {{ background-color: #fff3e0; color: var(--warning-color); }}
        
        .progress-bar {{ height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, var(--success-color), var(--secondary-color)); border-radius: 10px; }}
        
        .collapsible {{
            cursor: pointer; padding: 15px; background-color: #f4f4f4;
            border: none; text-align: left; width: 100%; font-size: 1em;
            border-radius: 5px; margin: 10px 0; display: flex;
            justify-content: space-between; align-items: center;
        }}
        .collapsible:after {{ content: '+'; font-weight: bold; }}
        .collapsible.active:after {{ content: '-'; }}
        .collapsible-content {{ max-height: 0; overflow: hidden; transition: max-height 0.3s ease-out; }}
        
        footer {{ text-align: center; padding: 20px; color: #666; font-size: 0.9em; border-top: 1px solid var(--border-color); margin-top: 30px; }}
        
        @media print {{ body {{ background: white; padding: 0; }} .card {{ box-shadow: none; border: 1px solid #ddd; }} .collapsible-content {{ max-height: none !important; }} }}
    </style>
</head>
<body>
"""

HTML_FOOTER = """
    <footer>
        <p>Generated by AmyloidBench | {timestamp}</p>
        <p>For research use only</p>
    </footer>
    
    <script>
        document.querySelectorAll('.collapsible').forEach(button => {{
            button.addEventListener('click', function() {{
                this.classList.toggle('active');
                const content = this.nextElementSibling;
                if (content.style.maxHeight) {{ content.style.maxHeight = null; }}
                else {{ content.style.maxHeight = content.scrollHeight + 'px'; }}
            }});
        }});
    </script>
</body>
</html>
"""


@dataclass
class ReportConfig:
    """Configuration for HTML report generation."""
    title: str = "AmyloidBench Analysis Report"
    include_figures: bool = True
    include_raw_data: bool = True
    figure_format: str = "png"
    style: Optional[PlotStyle] = None


class SequenceReportGenerator:
    """Generate comprehensive HTML report for single sequence analysis."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
    
    def generate(
        self,
        sequence: str,
        sequence_id: str,
        prediction_results: dict[str, Any],
        consensus_result: Optional[dict] = None,
        polymorph_result: Optional[dict] = None,
        features: Optional[dict] = None,
    ) -> str:
        """Generate complete HTML report."""
        sections = []
        sections.append(self._generate_header(sequence_id, sequence))
        sections.append(self._generate_sequence_section(sequence, sequence_id))
        sections.append(self._generate_predictions_section(prediction_results, sequence, consensus_result))
        
        if consensus_result:
            sections.append(self._generate_regions_section(consensus_result, sequence))
        if polymorph_result:
            sections.append(self._generate_polymorph_section(polymorph_result))
        if features:
            sections.append(self._generate_features_section(features))
        
        report_html = HTML_HEAD.format(title=f"{self.config.title} - {sequence_id}")
        report_html += "\n".join(sections)
        report_html += HTML_FOOTER.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return report_html
    
    def _generate_header(self, sequence_id: str, sequence: str) -> str:
        return f"""
        <header>
            <h1>{html.escape(self.config.title)}</h1>
            <p class="subtitle">Sequence: {html.escape(sequence_id)}</p>
            <p class="meta">Length: {len(sequence)} residues | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </header>
        """
    
    def _generate_sequence_section(self, sequence: str, sequence_id: str) -> str:
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
        
        hydrophobic = sum(aa_counts.get(aa, 0) for aa in 'VILMFYW')
        charged = sum(aa_counts.get(aa, 0) for aa in 'DERK')
        polar = sum(aa_counts.get(aa, 0) for aa in 'STNQ')
        
        return f"""
        <div class="card">
            <h2>Sequence Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="value">{len(sequence)}</div>
                    <div class="label">Length (aa)</div>
                </div>
                <div class="metric-card">
                    <div class="value">{hydrophobic/len(sequence)*100:.1f}%</div>
                    <div class="label">Hydrophobic</div>
                </div>
                <div class="metric-card">
                    <div class="value">{charged/len(sequence)*100:.1f}%</div>
                    <div class="label">Charged</div>
                </div>
                <div class="metric-card">
                    <div class="value">{polar/len(sequence)*100:.1f}%</div>
                    <div class="label">Polar</div>
                </div>
            </div>
            <h3>Sequence</h3>
            <div class="sequence-box">{self._format_sequence(sequence)}</div>
        </div>
        """
    
    def _format_sequence(self, sequence: str, line_length: int = 60) -> str:
        lines = []
        for i in range(0, len(sequence), line_length):
            chunk = sequence[i:i+line_length]
            line = f"<span style='color:#999;'>{i+1:>4}</span>  {html.escape(chunk)}"
            lines.append(line)
        return "<br>".join(lines)
    
    def _generate_predictions_section(self, results: dict[str, Any], sequence: str, consensus: Optional[dict] = None) -> str:
        html_parts = ['<div class="card">', '<h2>Amyloidogenicity Predictions</h2>']
        
        if consensus:
            is_amyloid = consensus.get('is_amyloidogenic', False)
            prob = consensus.get('probability', 0)
            agreement = consensus.get('agreement', '?/?')
            
            tag_class = 'tag-positive' if is_amyloid else 'tag-negative'
            tag_text = 'Amyloidogenic' if is_amyloid else 'Non-amyloidogenic'
            
            html_parts.append(f"""
            <div style="text-align: center; margin: 20px 0;">
                <span class="tag {tag_class}" style="font-size: 1.2em; padding: 8px 20px;">{tag_text}</span>
                <p style="margin-top: 10px; color: #666;">Consensus probability: {prob*100:.1f}% | Agreement: {agreement}</p>
            </div>
            """)
        
        if self.config.include_figures and HAS_MATPLOTLIB and results:
            scores_dict = {}
            for name, result in results.items():
                if hasattr(result, 'per_residue_scores') and result.per_residue_scores:
                    scores_dict[name] = result.per_residue_scores.scores
            
            if scores_dict:
                fig = plot_multi_predictor_profile(scores_dict, sequence=sequence)
                if fig:
                    img_data = figure_to_base64(fig, format=self.config.figure_format)
                    html_parts.append(f"""
                    <div class="figure">
                        <img src="{img_data}" alt="Score profiles">
                        <p class="figure-caption">Per-residue amyloidogenicity scores from multiple predictors</p>
                    </div>
                    """)
                    import matplotlib.pyplot as plt
                    plt.close(fig)
        
        html_parts.append('<h3>Predictor Results</h3>')
        html_parts.append('<table><tr><th>Predictor</th><th>Amyloidogenic</th><th>Probability</th><th>APRs</th></tr>')
        
        for name, result in results.items():
            is_amy = getattr(result, 'is_amyloidogenic', False)
            prob = getattr(result, 'amyloid_probability', 0) or 0
            n_regions = len(getattr(result, 'predicted_regions', []))
            tag = '<span class="tag tag-positive">Yes</span>' if is_amy else '<span class="tag tag-negative">No</span>'
            html_parts.append(f"<tr><td>{html.escape(name)}</td><td>{tag}</td><td>{prob*100:.1f}%</td><td>{n_regions}</td></tr>")
        
        html_parts.append('</table></div>')
        return '\n'.join(html_parts)
    
    def _generate_regions_section(self, consensus: dict, sequence: str) -> str:
        regions = consensus.get('regions', [])
        html_parts = ['<div class="card">', '<h2>Aggregation-Prone Regions (APRs)</h2>']
        
        if not regions:
            html_parts.append('<p>No APR regions identified by consensus.</p>')
        else:
            html_parts.append(f'<p>Identified {len(regions)} consensus APR region(s):</p>')
            html_parts.append('<ul class="region-list">')
            
            for region in regions:
                start = region.get('start', 0)
                end = region.get('end', 0)
                seq = region.get('sequence', sequence[start:end])
                score = region.get('score', 0)
                html_parts.append(f"""
                <li class="region-item">
                    <span class="position">{start+1}-{end}</span>
                    <span class="tag tag-neutral">Score: {score:.2f}</span>
                    <div class="sequence">{html.escape(seq)}</div>
                </li>
                """)
            html_parts.append('</ul>')
            
            html_parts.append('<h3>Sequence with APRs Highlighted</h3>')
            highlighted = self._highlight_regions(sequence, regions)
            html_parts.append(f'<div class="sequence-box">{highlighted}</div>')
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _highlight_regions(self, sequence: str, regions: list[dict]) -> str:
        apr_positions = set()
        for r in regions:
            for i in range(r.get('start', 0), r.get('end', 0)):
                if i < len(sequence):
                    apr_positions.add(i)
        
        result = []
        in_apr = False
        
        for i, aa in enumerate(sequence):
            is_apr = i in apr_positions
            if is_apr and not in_apr:
                result.append('<span class="apr-highlight">')
                in_apr = True
            elif not is_apr and in_apr:
                result.append('</span>')
                in_apr = False
            result.append(html.escape(aa))
            if (i + 1) % 60 == 0:
                if in_apr:
                    result.append('</span><br><span class="apr-highlight">')
                else:
                    result.append('<br>')
        
        if in_apr:
            result.append('</span>')
        return ''.join(result)
    
    def _generate_polymorph_section(self, polymorph: dict) -> str:
        fold = polymorph.get('fold', 'unknown')
        geometry = polymorph.get('geometry', 'unknown')
        zipper_class = polymorph.get('zipper_class', 'unknown')
        confidence = polymorph.get('confidence', 0)
        fold_probs = polymorph.get('fold_probabilities', {})
        
        html_parts = ['<div class="card">', '<h2>Structural Classification</h2>']
        html_parts.append(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="value" style="font-size: 1.2em;">{fold.replace('_', ' ').title()}</div>
                <div class="label">Predicted Fold</div>
            </div>
            <div class="metric-card">
                <div class="value" style="font-size: 1.2em;">{geometry.replace('_', ' ').title()}</div>
                <div class="label">Cross-β Geometry</div>
            </div>
            <div class="metric-card">
                <div class="value">{zipper_class.replace('_', ' ').title()}</div>
                <div class="label">Steric Zipper</div>
            </div>
            <div class="metric-card">
                <div class="value">{confidence*100:.0f}%</div>
                <div class="label">Confidence</div>
            </div>
        </div>
        """)
        
        if fold_probs and self.config.include_figures and HAS_MATPLOTLIB:
            fig = plot_polymorph_probabilities(fold_probs, title="")
            if fig:
                img_data = figure_to_base64(fig, format=self.config.figure_format)
                html_parts.append(f"""
                <div class="figure">
                    <img src="{img_data}" alt="Fold probabilities">
                    <p class="figure-caption">Probability distribution over fold types</p>
                </div>
                """)
                import matplotlib.pyplot as plt
                plt.close(fig)
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def _generate_features_section(self, features: dict) -> str:
        html_parts = ['<div class="card">', '<h2>Extracted Features</h2>']
        
        key_features = ['hydrophobicity', 'beta_sheet_propensity', 'net_charge', 'aggregation_propensity', 'disorder_propensity']
        
        html_parts.append('<div class="metric-grid">')
        for feat in key_features:
            if feat in features:
                value = features[feat]
                html_parts.append(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #4a90d9 0%, #357abd 100%);">
                    <div class="value">{value:.2f}</div>
                    <div class="label">{feat.replace('_', ' ').title()}</div>
                </div>
                """)
        html_parts.append('</div>')
        
        html_parts.append('<button class="collapsible">View All Features</button>')
        html_parts.append('<div class="collapsible-content"><table><tr><th>Feature</th><th>Value</th></tr>')
        
        for name, value in sorted(features.items()):
            value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            html_parts.append(f'<tr><td>{html.escape(name)}</td><td>{value_str}</td></tr>')
        
        html_parts.append('</table></div></div>')
        return '\n'.join(html_parts)
    
    def save(self, html_content: str, path: Path) -> Path:
        path = Path(path)
        path.write_text(html_content, encoding='utf-8')
        logger.info(f"Report saved to {path}")
        return path


class BenchmarkReportGenerator:
    """Generate HTML report for benchmark evaluation results."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig(title="AmyloidBench Benchmark Report")
    
    def generate(self, results: list[dict], dataset_name: str, dataset_info: Optional[dict] = None) -> str:
        sections = []
        sections.append(self._generate_header(dataset_name))
        if dataset_info:
            sections.append(self._generate_dataset_section(dataset_info))
        sections.append(self._generate_performance_section(results))
        sections.append(self._generate_rankings_section(results))
        
        report_html = HTML_HEAD.format(title=f"{self.config.title} - {dataset_name}")
        report_html += "\n".join(sections)
        report_html += HTML_FOOTER.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return report_html
    
    def _generate_header(self, dataset_name: str) -> str:
        return f"""
        <header>
            <h1>{html.escape(self.config.title)}</h1>
            <p class="subtitle">Dataset: {html.escape(dataset_name)}</p>
            <p class="meta">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        </header>
        """
    
    def _generate_dataset_section(self, info: dict) -> str:
        return f"""
        <div class="card">
            <h2>Dataset Information</h2>
            <div class="metric-grid">
                <div class="metric-card"><div class="value">{info.get('n_samples', '?')}</div><div class="label">Samples</div></div>
                <div class="metric-card"><div class="value">{info.get('n_positive', '?')}</div><div class="label">Positive</div></div>
                <div class="metric-card"><div class="value">{info.get('n_negative', '?')}</div><div class="label">Negative</div></div>
            </div>
            <p>{html.escape(info.get('description', ''))}</p>
        </div>
        """
    
    def _generate_performance_section(self, results: list[dict]) -> str:
        html_parts = ['<div class="card">', '<h2>Performance Metrics</h2>']
        
        if self.config.include_figures and HAS_MATPLOTLIB:
            fig = plot_benchmark_comparison(results)
            if fig:
                img_data = figure_to_base64(fig, format=self.config.figure_format)
                html_parts.append(f"""
                <div class="figure">
                    <img src="{img_data}" alt="Performance comparison">
                    <p class="figure-caption">Comparison of key performance metrics</p>
                </div>
                """)
                import matplotlib.pyplot as plt
                plt.close(fig)
        
        html_parts.append('<table><tr><th>Predictor</th><th>Sensitivity</th><th>Specificity</th><th>Precision</th><th>F1</th><th>MCC</th><th>AUC</th></tr>')
        
        for r in sorted(results, key=lambda x: x.get('mcc', 0), reverse=True):
            html_parts.append(f"""
            <tr>
                <td>{html.escape(r.get('predictor', ''))}</td>
                <td>{r.get('sensitivity', 0):.3f}</td>
                <td>{r.get('specificity', 0):.3f}</td>
                <td>{r.get('precision', 0):.3f}</td>
                <td>{r.get('f1_score', 0):.3f}</td>
                <td><strong>{r.get('mcc', 0):.3f}</strong></td>
                <td>{r.get('auc_roc', 0):.3f}</td>
            </tr>
            """)
        
        html_parts.append('</table></div>')
        return '\n'.join(html_parts)
    
    def _generate_rankings_section(self, results: list[dict]) -> str:
        sorted_results = sorted(results, key=lambda x: x.get('mcc', 0), reverse=True)
        html_parts = ['<div class="card">', '<h2>Predictor Rankings</h2>']
        
        for i, r in enumerate(sorted_results):
            mcc = r.get('mcc', 0)
            name = r.get('predictor', 'Unknown')
            bar_width = max(0, (mcc + 1) / 2 * 100)
            medal = ['🥇', '🥈', '🥉'][i] if i < 3 else f'#{i+1}'
            
            html_parts.append(f"""
            <div style="margin: 15px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span><strong>{medal} {html.escape(name)}</strong></span>
                    <span>MCC: {mcc:.3f}</span>
                </div>
                <div class="progress-bar"><div class="progress-fill" style="width: {bar_width}%;"></div></div>
            </div>
            """)
        
        html_parts.append('</div>')
        return '\n'.join(html_parts)
    
    def save(self, html_content: str, path: Path) -> Path:
        path = Path(path)
        path.write_text(html_content, encoding='utf-8')
        logger.info(f"Report saved to {path}")
        return path


def generate_sequence_report(sequence: str, sequence_id: str, prediction_results: dict, output_path: Optional[Path] = None, **kwargs) -> str:
    """Generate HTML report for a single sequence analysis."""
    generator = SequenceReportGenerator()
    report_html = generator.generate(sequence=sequence, sequence_id=sequence_id, prediction_results=prediction_results, **kwargs)
    if output_path:
        generator.save(report_html, output_path)
    return report_html


def generate_benchmark_report(results: list[dict], dataset_name: str, output_path: Optional[Path] = None, **kwargs) -> str:
    """Generate HTML report for benchmark evaluation."""
    generator = BenchmarkReportGenerator()
    report_html = generator.generate(results=results, dataset_name=dataset_name, **kwargs)
    if output_path:
        generator.save(report_html, output_path)
    return report_html
