#!/usr/bin/env python3
"""Convert the MAPPO explainability JSON into a readable, filterable HTML report.

Run from the repository root:
    python make_explanation_report.py
Or select a particular input/output:
    python make_explanation_report.py --input evaluation/explanations/decision_trace_seed42.json
    python make_explanation_report.py --input path/to/report.json --output path/to/report.html

Uses only the Python standard library; no pip installation required.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def pp(value: Any, digits: int = 1) -> str:
    """Probability difference shown in percentage points."""
    try:
        number = float(value) * 100
        return f"{number:+.{digits}f} pp"
    except (TypeError, ValueError):
        return "—"


def num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def delta_class(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if value > 0.005:
        return "positive"
    if value < -0.005:
        return "negative"
    return "neutral"


def delta_label(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "Not available"
    if value > 0.005:
        return "Removal lowered the action probability"
    if value < -0.005:
        return "Removal raised the action probability"
    return "Little change in this test"


def progress_bar(value: Any, max_width: int = 100) -> str:
    try:
        width = max(0.0, min(float(value) * max_width, max_width))
    except (TypeError, ValueError):
        width = 0.0
    return f'<span class="bar"><span style="width:{width:.2f}%"></span></span>'


def render_action_table(actions: list[dict]) -> str:
    if not actions:
        return '<p class="muted">No alternative-action probabilities were recorded.</p>'
    rows = []
    for rank, item in enumerate(actions, 1):
        prob = item.get("probability", 0.0)
        rows.append(
            '<tr>'
            f'<td class="rank">{rank}</td>'
            f'<td class="action-label"><span class="action-index">[{esc(item.get("action_index"))}]</span> '
            f'{esc(item.get("label", "Unnamed action"))}</td>'
            f'<td class="prob">{pct(prob)}</td>'
            f'<td class="bar-cell">{progress_bar(prob)}</td>'
            '</tr>'
        )
    return ('<div class="table-wrap"><table class="actions-table"><thead><tr>'
            '<th>#</th><th>Available action</th><th>Policy probability</th><th></th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def render_messages(report: dict) -> str:
    incoming = report.get("incoming_messages") or []
    counterfactuals = {x.get("sender_agent_id"): x for x in report.get("message_counterfactuals", [])}
    if not incoming or not any(x.get("message_available") for x in incoming):
        return '<div class="empty-state">No previous-step messages were available for this decision. Outgoing messages generated now are available to agents on the next decision step.</div>'

    rows = []
    for item in incoming:
        sender = item.get("sender_agent_id")
        msg = item.get("structured_message") or {}
        cf = counterfactuals.get(sender, {})
        delta = cf.get("delta_probability_baseline_minus_ablation")
        if not msg or msg.get("event_type") == "NONE":
            message_text = '<span class="badge quiet">No event reported</span>'
            target = '—'
            threat = '—'
            confidence = '—'
            status = '—'
            priority = '—'
        else:
            message_text = f'<span class="badge event">{esc(msg.get("event_type", "Unknown"))}</span>'
            target = msg.get("target_name") or msg.get("target_type") or "—"
            threat = msg.get("threat_level", "—")
            confidence = pct(msg.get("confidence"))
            status = msg.get("status", "—")
            priority = msg.get("priority", "—")
        if delta is None:
            delta_html = '<span class="muted">Not tested</span>'
            effect_html = ''
        else:
            cls = delta_class(delta)
            delta_html = f'<span class="delta {cls}">{esc(pp(delta))}</span>'
            effect_html = f'<div class="tiny muted">{esc(delta_label(delta))}</div>'
        rows.append(
            '<tr>'
            f'<td><strong>Agent {esc(sender)}</strong></td>'
            f'<td>{message_text}<div class="tiny">{esc(target)}</div></td>'
            f'<td>{esc(threat)}<div class="tiny muted">Confidence: {esc(confidence)}</div></td>'
            f'<td>{esc(status)}<div class="tiny muted">Priority: {esc(priority)}</div></td>'
            f'<td>{num(item.get("trust_receiver_places_in_sender"))}</td>'
            f'<td>{num(item.get("communication_attention_weight"))}</td>'
            f'<td>{delta_html}{effect_html}</td>'
            '</tr>'
        )
    return ('<div class="table-wrap"><table><thead><tr>'
            '<th>Sender</th><th>Message / target</th><th>Threat</th><th>Status / priority</th>'
            '<th>Trust</th><th>Attention*</th><th>Effect of removing message</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            '<p class="footnote">*Attention is an internal model weight, not proof that a message caused the action. Trust is the receiver’s trust in the sender.</p>')


def render_feature_sensitivities(report: dict) -> str:
    features = report.get("observation_feature_counterfactuals") or {}
    rows = []
    nice_names = {
        "process_alert_features_removed": "Process-alert features",
        "connection_alert_features_removed": "Connection-alert features",
        "all_host_alert_features_removed": "All host-alert features",
    }
    for key, item in features.items():
        if not item.get("available"):
            continue
        delta = item.get("delta_probability_baseline_minus_ablation")
        rows.append(
            '<tr>'
            f'<td>{esc(nice_names.get(key, key.replace("_", " ").title()))}</td>'
            f'<td>{pct(item.get("selected_action_probability_after_ablation"))}</td>'
            f'<td><span class="delta {delta_class(delta)}">{esc(pp(delta))}</span></td>'
            f'<td>{esc(delta_label(delta))}</td>'
            '</tr>'
        )
    if not rows:
        return '<p class="muted">No feature ablation results were recorded.</p>'
    return ('<div class="table-wrap"><table><thead><tr><th>Removed input group</th>'
            '<th>Action probability after removal</th><th>Change vs. baseline</th><th>Interpretation</th>'
            '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def render_suppression_details(report: dict) -> str:
    action_mask = report.get("action_mask") or {}
    reasons = action_mask.get("adaptive_suppression_reasons") or []
    if not reasons:
        body = '<p class="muted">No adaptive suppression reasons were recorded for this decision.</p>'
    else:
        body = '<ul class="reason-list">' + ''.join(
            f'<li><strong>{esc(item.get("label", "Action"))}</strong><br>{esc(item.get("reason", ""))}</li>'
            for item in reasons
        ) + '</ul>'
    return (
        f'<details class="advanced"><summary>Action mask &amp; suppressed actions '
        f'<span class="count">{len(reasons)} reasons</span></summary>'
        f'<div class="details-body"><p><strong>Mask mode:</strong> {esc(action_mask.get("mode", "unknown"))} '
        f'&nbsp; · &nbsp; <strong>Available actions:</strong> {esc(action_mask.get("valid_action_count", "—"))}</p>'
        f'{body}</div></details>'
    )


def render_card(report: dict) -> str:
    step = report.get("timestep", 0)
    agent = report.get("agent_id", "?")
    name = report.get("agent_name", f"Agent {agent}")
    action = report.get("selected_action") or {}
    evidence = report.get("local_observation_evidence") or {}
    prob = action.get("policy_probability", 0.0)
    margin = action.get("probability_margin_over_second_choice")
    rank = action.get("rank_among_available_actions", "—")
    hosts = evidence.get("flagged_hosts") or []
    zones = evidence.get("flagged_zones") or []
    host_badges = ''.join(f'<span class="badge alert">{esc(h)}</span>' for h in hosts) or '<span class="badge quiet">No flagged hosts</span>'
    zone_badges = ''.join(f'<span class="badge zone">{esc(z)}</span>' for z in zones) or '<span class="badge quiet">No flagged zones</span>'
    action_label = action.get("label", "Unknown action")
    selected_label = esc(action_label)
    action_type = esc(action.get("action_type", "Action"))
    context = esc(evidence.get("action_context_note", "No additional local context was recorded."))
    top_actions = report.get("top_available_actions") or []
    top_json = ' '.join(esc(x.get("label", "")) for x in top_actions[:3])
    filter_text = esc(f'{step} {agent} {name} {action_label} {top_json}'.lower())
    margin_text = pp(margin) if margin is not None else "—"
    return f'''<article class="decision-card" data-step="{esc(step)}" data-agent="{esc(agent)}" data-search="{filter_text}">
      <div class="card-topline"><div><span class="step-pill">STEP {esc(step)}</span> <span class="agent-title">{esc(name)}</span></div><span class="rank-pill">Choice rank: {esc(rank)}</span></div>
      <h3>{selected_label}</h3>
      <div class="metrics">
        <div class="metric primary"><span>Chosen action probability</span><strong>{pct(prob)}</strong><div class="metric-bar">{progress_bar(prob)}</div></div>
        <div class="metric"><span>Gap over runner-up</span><strong>{esc(margin_text)}</strong><small>percentage points</small></div>
        <div class="metric"><span>Actions available</span><strong>{esc((report.get("action_mask") or {{}}).get("valid_action_count", "—"))}</strong><small>after action masking</small></div>
      </div>
      <section class="report-section"><h4>1. What information was visible?</h4>
        <div class="tag-row">{host_badges}</div><div class="tag-row">{zone_badges}</div>
        <p class="context-note">{context}</p>
      </section>
      <section class="report-section"><h4>2. Other actions the policy considered</h4>{render_action_table(top_actions)}</section>
      <section class="report-section"><h4>3. Incoming messages and trust</h4>{render_messages(report)}</section>
      <section class="report-section"><h4>4. What changed when alert features were removed?</h4>{render_feature_sensitivities(report)}</section>
      {render_suppression_details(report)}
      <p class="caveat"><strong>How to read this:</strong> a positive removal effect means the selected action became less likely when that input was removed; a negative effect means it became more likely. These are sensitivity tests, not proof of causation.</p>
    </article>'''


def build_html(reports: list[dict], source_path: Path) -> str:
    reports = sorted(reports, key=lambda x: (int(x.get("timestep", 0)), int(x.get("agent_id", 0))))
    steps = sorted({int(x.get("timestep", 0)) for x in reports})
    agents = sorted({(int(x.get("agent_id", 0)), str(x.get("agent_name", ""))) for x in reports})
    probs = [float((x.get("selected_action") or {}).get("policy_probability", 0.0)) for x in reports]
    avg_prob = mean(probs) if probs else 0
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    step_options = '<option value="all">All steps</option>' + ''.join(f'<option value="{s}">Step {s}</option>' for s in steps)
    agent_options = '<option value="all">All agents</option>' + ''.join(
        f'<option value="{aid}">Agent {aid} — {esc(aname)}</option>' for aid, aname in agents
    )
    cards = ''.join(render_card(report) for report in reports)
    title = f'MAPPO Explainability Report — {source_path.stem}'
    limits = [
        "The neural policy does not expose a definitive human-readable chain of reasoning.",
        "Removing a message or alert group measures model sensitivity under that test; it is not proof of real-world causation.",
        "Attention weights are descriptive indicators, not causal proof.",
        "The local-alert summary reflects the defender-visible alert signals used by this run; simulator ground truth is not used to justify decisions.",
    ]
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--ink:#172033;--muted:#5d6b82;--line:#dfe6f0;--blue:#2458d3;--blue-light:#eaf1ff;--green:#087f5b;--green-bg:#e4f7ef;--red:#b42318;--red-bg:#fff0ee;--amber:#8a5a00;--amber-bg:#fff5db;--shadow:0 8px 28px rgba(22,35,65,.07)}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}.hero{{background:linear-gradient(120deg,#14264a,#244fac);color:white;padding:32px max(22px,calc((100vw - 1240px)/2)) 28px}}.hero h1{{margin:0 0 8px;font-size:clamp(26px,3.3vw,38px);letter-spacing:-.025em}}.hero p{{margin:0;max-width:900px;color:#e2eaff}}.container{{max-width:1240px;margin:0 auto;padding:24px}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:22px}}.stat,.filter-panel,.decision-card,.limits{{background:var(--panel);border:1px solid var(--line);border-radius:15px;box-shadow:var(--shadow)}}.stat{{padding:18px 20px}}.stat span{{display:block;color:var(--muted);font-size:13px;font-weight:650}}.stat strong{{display:block;font-size:27px;margin-top:2px}}.filter-panel{{padding:16px 18px;margin-bottom:18px;display:flex;gap:14px;align-items:end;flex-wrap:wrap}}label{{display:flex;flex-direction:column;gap:5px;font-size:12px;font-weight:700;color:var(--muted)}}select,input{{border:1px solid #c9d4e4;border-radius:9px;background:white;color:var(--ink);padding:10px 12px;font:inherit;min-width:180px}}input{{min-width:240px;flex:1}}button{{border:0;border-radius:9px;background:var(--blue);color:white;font-weight:700;padding:11px 15px;cursor:pointer}}.result-count{{margin-left:auto;color:var(--muted);font-size:13px;padding-bottom:9px}}.decision-list{{display:grid;gap:22px}}.decision-card{{padding:22px;min-width:0}}.card-topline{{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}}.step-pill,.rank-pill{{font-size:12px;font-weight:800;border-radius:20px;padding:5px 9px;background:var(--blue-light);color:var(--blue)}}.rank-pill{{background:#eef1f6;color:#42516a}}.agent-title{{font-size:14px;font-weight:750;margin-left:7px}}.decision-card h3{{font-size:21px;line-height:1.35;margin:15px 0 17px;overflow-wrap:anywhere}}.metrics{{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:12px}}.metric{{border:1px solid var(--line);border-radius:12px;padding:13px 15px;background:#fbfcff;min-width:0}}.metric span,.metric small{{display:block;color:var(--muted);font-size:12px}}.metric strong{{display:block;font-size:24px;margin:4px 0;overflow-wrap:anywhere}}.metric.primary{{background:#f0f5ff;border-color:#cbdcff}}.metric-bar{{margin-top:8px}}.bar{{display:block;background:#e5eaf4;border-radius:99px;height:8px;overflow:hidden;min-width:65px}}.bar span{{display:block;background:var(--blue);height:100%;border-radius:99px}}.report-section{{margin-top:22px}}h4{{font-size:15px;margin:0 0 10px}}.tag-row{{display:flex;gap:7px;flex-wrap:wrap;margin:7px 0}}.badge{{display:inline-block;border-radius:6px;padding:4px 8px;font-size:12px;font-weight:750;overflow-wrap:anywhere}}.badge.alert{{background:var(--red-bg);color:var(--red)}}.badge.zone{{background:var(--amber-bg);color:var(--amber)}}.badge.quiet{{background:#eef1f5;color:#657188}}.badge.event{{background:var(--blue-light);color:var(--blue)}}.context-note{{border-left:3px solid #afc5fb;padding:8px 12px;background:#f8faff;border-radius:0 7px 7px 0;color:#43516a;margin:12px 0 0}}.table-wrap{{width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:10px}}table{{border-collapse:collapse;width:100%;font-size:13px;min-width:680px}}th{{text-align:left;background:#f4f7fb;color:#506078;font-size:11px;text-transform:uppercase;letter-spacing:.045em;padding:11px 12px;white-space:nowrap}}td{{border-top:1px solid var(--line);padding:11px 12px;vertical-align:top}}.actions-table{{min-width:540px}}.rank{{width:32px;color:var(--muted)}}.action-label{{overflow-wrap:anywhere;min-width:250px}}.action-index{{color:var(--muted);font-size:11px;font-weight:750;white-space:nowrap}}.prob{{white-space:nowrap;font-weight:800;width:110px}}.bar-cell{{width:24%;min-width:110px}}.delta{{display:inline-block;font-weight:800;white-space:nowrap}}.delta.positive{{color:var(--green)}}.delta.negative{{color:var(--red)}}.delta.neutral{{color:var(--muted)}}.tiny{{font-size:11px;color:#52627a;overflow-wrap:anywhere;margin-top:3px}}.muted{{color:var(--muted)}}.footnote,.caveat{{font-size:12px;color:var(--muted)}}.empty-state{{padding:14px;border:1px dashed #cbd5e1;border-radius:10px;color:var(--muted);background:#fafcff}}.advanced{{margin-top:18px;border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:#fbfcff}}summary{{cursor:pointer;font-weight:750}}.count{{font-size:12px;color:var(--muted);font-weight:500;margin-left:6px}}.details-body{{padding-top:10px}}.reason-list{{padding-left:22px}}.reason-list li{{margin:8px 0}}.caveat{{background:#fff8e5;border:1px solid #f3dfaa;border-radius:9px;padding:11px 13px;margin-top:18px;color:#79520b}}.limits{{margin-top:22px;padding:18px 22px}}.limits h2{{margin:0 0 9px;font-size:17px}}.limits li{{margin:4px 0;color:var(--muted)}}.no-results{{display:none;text-align:center;padding:35px;background:white;border:1px dashed var(--line);border-radius:12px;color:var(--muted)}}footer{{padding:20px 0 5px;color:var(--muted);font-size:12px;text-align:center}}@media(max-width:800px){{.container{{padding:14px}}.stats{{grid-template-columns:repeat(2,minmax(0,1fr))}}.metrics{{grid-template-columns:1fr}}.decision-card{{padding:16px}}.hero{{padding:25px 18px}}.result-count{{margin-left:0;width:100%}}}}@media print{{body{{background:#fff}}.hero{{background:#fff;color:#111;padding:15px 0}}.hero p{{color:#333}}.container{{max-width:none;padding:0}}.filter-panel,button{{display:none!important}}.decision-card,.stat,.limits{{box-shadow:none;break-inside:avoid}}.decision-list{{gap:12px}}details{{break-inside:avoid}}}}
</style></head><body>
<header class="hero"><h1>MAPPO Decision Explainability</h1><p>Readable review of what each agent chose, the policy’s alternatives, visible alert context, messages received, and input-removal sensitivity tests. This report does not alter the trained model.</p></header>
<main class="container">
  <section class="stats">
    <div class="stat"><span>Decisions explained</span><strong>{len(reports)}</strong></div>
    <div class="stat"><span>Decision steps</span><strong>{len(steps)}</strong></div>
    <div class="stat"><span>Agents</span><strong>{len(agents)}</strong></div>
    <div class="stat"><span>Average selected-action probability</span><strong>{pct(avg_prob)}</strong></div>
  </section>
  <section class="filter-panel">
    <label>Decision step<select id="stepFilter">{step_options}</select></label>
    <label>Agent<select id="agentFilter">{agent_options}</select></label>
    <label style="flex:1">Search action / host / event<input id="searchFilter" placeholder="e.g. Analyse, suspicious activity, Agent 2"></label>
    <button id="resetFilters" type="button">Reset filters</button>
    <span id="resultCount" class="result-count">Showing {len(reports)} decisions</span>
  </section>
  <section id="decisionList" class="decision-list">{cards}</section>
  <div id="noResults" class="no-results">No decisions match these filters.</div>
  <section class="limits"><h2>Interpretation notes</h2><ul>{''.join(f'<li>{esc(item)}</li>' for item in limits)}</ul><p class="muted">Generated {esc(generated)} · Source JSON: {esc(source_path.as_posix())}</p></section>
  <footer>Tip: use the filters to focus on one step or one agent. Use your browser’s Print command to save a clean PDF if needed.</footer>
</main>
<script>
(function() {{
 const step = document.getElementById('stepFilter');
 const agent = document.getElementById('agentFilter');
 const search = document.getElementById('searchFilter');
 const cards = Array.from(document.querySelectorAll('.decision-card'));
 const count = document.getElementById('resultCount');
 const none = document.getElementById('noResults');
 function update() {{
   const s = step.value, a = agent.value, q = search.value.trim().toLowerCase();
   let shown = 0;
   cards.forEach(card => {{
     const ok = (s === 'all' || card.dataset.step === s) &&
       (a === 'all' || card.dataset.agent === a) &&
       (!q || card.dataset.search.includes(q));
     card.style.display = ok ? '' : 'none';
     if (ok) shown++;
   }});
   count.textContent = `Showing ${{shown}} of ${{cards.length}} decisions`;
   none.style.display = shown ? 'none' : 'block';
 }}
 [step, agent, search].forEach(el => el.addEventListener('input', update));
 document.getElementById('resetFilters').addEventListener('click', () => {{ step.value='all'; agent.value='all'; search.value=''; update(); }});
 update();
}})();
</script></body></html>'''


def choose_input(input_path: str | None) -> Path:
    if input_path:
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input JSON not found: {path}")
        return path
    folder = Path("evaluation/explanations")
    candidates = [p for p in folder.glob("*.json") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No explanation JSON found in evaluation/explanations/. Run the decision trace first, "
            "or pass --input PATH."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to a decision_trace JSON file; defaults to the newest JSON in evaluation/explanations/")
    parser.add_argument("--output", help="Destination HTML path; defaults to the input file with .html extension")
    args = parser.parse_args()
    try:
        source = choose_input(args.input)
        with source.open("r", encoding="utf-8") as handle:
            reports = json.load(handle)
        if isinstance(reports, dict):
            reports = [reports]
        if not isinstance(reports, list) or not reports:
            raise ValueError(f"Expected a non-empty JSON list of decision reports in {source}")
        if not all(isinstance(item, dict) for item in reports):
            raise ValueError("The JSON must contain a list of decision-report objects.")
        output = Path(args.output) if args.output else source.with_suffix(".html")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_html(reports, source), encoding="utf-8")
        print(f"Created readable HTML report: {output}")
        print(f"Decisions included: {len(reports)}")
        print("Open the HTML file in your browser. No extra packages are needed.")
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
