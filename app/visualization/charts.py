"""
Plotly chart generation for A/B/C comparison results.
"""

from typing import Dict, List, Any


def create_comparison_chart(results: Dict[str, Any]) -> Dict:
    """
    Create A/B/C manipulation rate comparison chart.

    Returns Plotly JSON for frontend rendering.
    """
    import plotly.graph_objects as go

    # Get behaviors from results
    behaviors = []
    for ep in results.get("episodes", []):
        beh = ep.get("behavior", "")
        if beh and beh not in behaviors:
            behaviors.append(beh)

    conditions = results.get("conditions", ["A", "B", "C"])

    # Calculate manipulation rates per condition per behavior
    rates = {cond: {beh: {"total": 0, "manipulated": 0} for beh in behaviors} for cond in conditions}

    for ep in results.get("episodes", []):
        beh = ep.get("behavior", "")
        for cond, data in ep.get("conditions", {}).items():
            if cond in rates and beh in rates[cond]:
                rates[cond][beh]["total"] += 1
                if data.get("manipulation_detected"):
                    rates[cond][beh]["manipulated"] += 1

    # Convert to percentages
    for cond in conditions:
        for beh in behaviors:
            total = rates[cond][beh]["total"]
            if total > 0:
                rates[cond][beh]["rate"] = 100 * rates[cond][beh]["manipulated"] / total
            else:
                rates[cond][beh]["rate"] = 0

    # Format behavior names for display
    behavior_labels = [b.replace("_", " ").title() for b in behaviors]

    # Create grouped bar chart
    fig = go.Figure()

    colors = {
        "A": "#EF4444",  # Red
        "B": "#F59E0B",  # Yellow/Orange
        "C": "#10B981",  # Green
    }

    names = {
        "A": "A: Baseline",
        "B": "B: Monitor-only",
        "C": "C: Full Harness",
    }

    for cond in conditions:
        fig.add_trace(go.Bar(
            name=names.get(cond, cond),
            x=behavior_labels,
            y=[rates[cond][b]["rate"] for b in behaviors],
            marker_color=colors.get(cond, "#6B7280"),
            text=[f"{rates[cond][b]['rate']:.0f}%" for b in behaviors],
            textposition="auto",
        ))

    fig.update_layout(
        title={
            "text": "Manipulation Rate by Condition",
            "font": {"size": 20, "color": "#F9FAFB"},
        },
        xaxis_title="Behavior Type",
        yaxis_title="Manipulation Rate (%)",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font={"color": "#F9FAFB"},
        ),
        xaxis={"tickfont": {"color": "#F9FAFB"}},
        yaxis={"tickfont": {"color": "#F9FAFB"}, "range": [0, 100]},
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
    )

    return fig.to_dict()


def create_intervention_chart(results: Dict[str, Any]) -> Dict:
    """
    Create chart showing HUSH intervention effectiveness.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Collect intervention data from Condition C
    behaviors = []
    interventions = []
    corrections = []
    detection_scores = []

    for ep in results.get("episodes", []):
        beh = ep.get("behavior", "")
        cond_c = ep.get("conditions", {}).get("C", {})

        if beh and cond_c:
            behaviors.append(beh.replace("_", " ").title())
            interventions.append(cond_c.get("interventions", 0))
            corrections.append(cond_c.get("corrections", 0))
            detection_scores.append(cond_c.get("peak_detection_score", 0) * 100)

    if not behaviors:
        # Return empty chart
        fig = go.Figure()
        fig.update_layout(
            title="No Condition C data available",
            template="plotly_dark",
        )
        return fig.to_dict()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Interventions Triggered",
            "Peak Detection Score (%)"
        ),
    )

    fig.add_trace(
        go.Bar(
            x=behaviors,
            y=interventions,
            marker_color="#8B5CF6",
            name="Interventions",
            text=interventions,
            textposition="auto",
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=behaviors,
            y=detection_scores,
            marker_color="#06B6D4",
            name="Detection Score",
            text=[f"{s:.0f}%" for s in detection_scores],
            textposition="auto",
        ),
        row=1, col=2
    )

    fig.update_layout(
        title={
            "text": "HUSH Intervention Activity (Condition C)",
            "font": {"size": 20, "color": "#F9FAFB"},
        },
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#1F2937",
        showlegend=False,
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
    )

    fig.update_xaxes(tickfont={"color": "#F9FAFB"})
    fig.update_yaxes(tickfont={"color": "#F9FAFB"})

    return fig.to_dict()


def create_summary_stats(results: Dict[str, Any]) -> Dict:
    """
    Create summary statistics for display.
    """
    summary = results.get("summary", {})
    by_condition = summary.get("by_condition", {})

    stats = {
        "total_episodes": len(results.get("episodes", [])),
        "conditions_tested": len(results.get("conditions", [])),
        "baseline_rate": by_condition.get("A", {}).get("rate", 0),
        "monitor_rate": by_condition.get("B", {}).get("rate", 0),
        "harness_rate": by_condition.get("C", {}).get("rate", 0),
    }

    # Calculate improvement
    if stats["baseline_rate"] > 0:
        stats["improvement"] = (
            (stats["baseline_rate"] - stats["harness_rate"]) / stats["baseline_rate"] * 100
        )
    else:
        stats["improvement"] = 0

    return stats
