from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORWAY = ["#22c55e", "#38bdf8", "#f59e0b", "#ef4444", "#a78bfa", "#14b8a6", "#f97316"]


def cumulative_return_chart(df: pd.DataFrame, title: str = "Cumulative Return") -> go.Figure:
    fig = px.line(df, x="date_day", y="cumulative_return", color="ticker" if "ticker" in df else None, title=title)
    return style_figure(fig)


def performance_bar(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = px.bar(df, x=x, y=y, color=y, color_continuous_scale=["#ef4444", "#f8fafc", "#22c55e"], title=title)
    return style_figure(fig)


def risk_return_scatter(df: pd.DataFrame, title: str = "Risk / Return Map") -> go.Figure:
    fig = px.scatter(
        df,
        x="annualized_volatility",
        y="annualized_return",
        color="sector",
        size="average_volume",
        hover_name="ticker",
        facet_col=None,
        title=title,
    )
    return style_figure(fig)


def correlation_heatmap(corr: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", zmin=-1, zmax=1, title=title)
    return style_figure(fig)


def drawdown_chart(df: pd.DataFrame, title: str = "Drawdown") -> go.Figure:
    fig = px.area(df, x="date_day", y="drawdown", color="ticker" if "ticker" in df else None, title=title)
    return style_figure(fig)


def cluster_pca_chart(assignments: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        assignments,
        x="pca_1",
        y="pca_2",
        color="cluster",
        symbol="region" if "region" in assignments else None,
        hover_name="ticker",
        title="PCA Cluster Map",
    )
    return style_figure(fig)


def forecast_chart(history: pd.DataFrame, forecast: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date_day"], y=history["adjusted_close"], name="Historical", line=dict(color="#38bdf8")))
    if not forecast.empty:
        fig.add_trace(go.Scatter(x=forecast["date_day"], y=forecast["forecast"], name="Forecast", line=dict(color="#22c55e")))
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast["date_day"], forecast["date_day"].iloc[::-1]]),
                y=pd.concat([forecast["upper"], forecast["lower"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(34,197,94,0.18)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Uncertainty band",
            )
        )
    fig.update_layout(title=f"{ticker} forecast with uncertainty")
    return style_figure(fig)


def style_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        margin=dict(l=20, r=20, t=60, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
