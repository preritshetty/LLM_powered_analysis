# query_handler.py
import pandas as pd
import io
import re
import streamlit as st
import numpy as np
from datetime import datetime

# -------------------------------
# Utilities
# -------------------------------

def parse_markdown_table(response: str) -> pd.DataFrame | None:
    """Try to parse a markdown-style table from an LLM response string."""
    try:
        lines = [l.strip() for l in response.splitlines() if "|" in l]
        if len(lines) < 2:
            return None
        # Remove separator lines like ----|----
        lines = [l for l in lines if "---" not in l]
        table_str = "\n".join(lines)
        df = pd.read_csv(io.StringIO(table_str), sep="|").dropna(axis=1, how="all")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Generic cleaning: fill missing values and normalize datatypes."""
    if df is None or df.empty:
        return df

    for col in df.columns:
        if df[col].dtype == "O":  # categorical/string
            df[col] = df[col].fillna("N/A").replace("", "N/A").astype(str)
        else:  # numeric
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def pick_axes(result_df: pd.DataFrame):
    """Pick sensible default X and Y axes based on column types."""
    cols = result_df.columns.tolist()
    if len(cols) == 2:
        return cols[0], cols[1]

    numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in cols if c not in numeric_cols]

    x = non_numeric_cols[0] if non_numeric_cols else cols[0]
    y = numeric_cols[0] if numeric_cols else (cols[1] if len(cols) > 1 else cols[0])
    return x, y


def pick_chart_type(x: str, query: str) -> str:
    """Decide the best chart type based on query text and X-axis."""
    q = query.lower()
    x_lower = x.lower()
    if any(word in q for word in ["trend", "over time", "progression"]) or any(
        k in x_lower for k in ["date", "month", "day", "week", "year", "time"]
    ):
        return "Line"
    if any(word in q for word in ["share", "proportion", "percentage", "rate", "distribution"]):
        return "Pie"
    return "Bar"


# -------------------------------
# Conversational context helpers
# -------------------------------

def _init_context(max_turns: int = 3):
    """Ensure session state keys exist."""
    st.session_state.setdefault("conv_context", [])  # list of dicts: {"q": str, "a": str}
    st.session_state.setdefault("conv_max_turns", max_turns)


def _push_context(q: str, a: str):
    """Append a Q/A turn and clip to max turns."""
    _init_context()
    st.session_state.conv_context.insert(0, {"q": q.strip(), "a": a.strip()})
    # keep only last N
    st.session_state.conv_context = st.session_state.conv_context[: st.session_state.conv_max_turns]


def _build_context_block() -> str:
    """
    Convert recent Q&A into a concise instruction block for the LLM.
    We keep it short and structured to reduce prompt bloat.
    """
    _init_context()
    if not st.session_state.conv_context:
        return ""

    lines = ["Previous Q&A (most recent first):"]
    for i, turn in enumerate(st.session_state.conv_context, start=1):
        q = re.sub(r"\s+", " ", turn["q"]).strip()
        a = re.sub(r"\s+", " ", turn["a"]).strip()
        a_short = a if len(a) <= 300 else (a[:297] + "…")
        lines.append(f"- Q{i}: {q}")
        lines.append(f"  A{i}: {a_short}")
    lines.append(
        "When the new question uses pronouns like 'this/that/these/it' or says 'same airline',"
        " interpret them using the most recent relevant Q&A above."
    )
    return "\n".join(lines)


# -------------------------------
# Core Query
# -------------------------------

def format_table_for_query(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """Reformat results if query implies ranking/ordering."""
    if df is None or df.empty:
        return df

    q = query.lower()
    if any(word in q for word in ["most", "highest", "top", "largest", "biggest", "max"]):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            df = df.sort_values(by=numeric_cols[0], ascending=False)
    elif any(word in q for word in ["least", "lowest", "smallest", "min"]):
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            df = df.sort_values(by=numeric_cols[0], ascending=True)

    return df


def run_query(agent, query: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame | None]:
    """
    Run query with the LLM agent and return results.

    Returns
    -------
    response : str
        Natural language answer from the agent
    result_df : pd.DataFrame | None
        Parsed and cleaned DataFrame with proof data
    """
    try:
        # 🔗 Inject recent conversational context
        context_block = _build_context_block()

        # 🧾 Provide a compact schema so the LLM uses the real df (no toy data).
        # Keep it short to avoid prompt bloat.
        cols_preview = ", ".join([f"{c}({str(t)})" for c, t in df.dtypes.items()])
        row_count = len(df)
        schema_block = (
            "DATA CONTRACT (MUST FOLLOW):\n"
            f"- Use ONLY the provided pandas DataFrame named `df` (rows={row_count}).\n"
            "- DO NOT create synthetic/sample DataFrames in memory.\n"
            "- DO NOT hardcode answers. Compute with pandas on `df`.\n"
            "- Use only these columns (exact spelling):\n"
            f"  {cols_preview}\n"
            "- If a date-like field is needed, convert with `pd.to_datetime(..., errors='coerce')`.\n"
            "- If user asks about month/weekday, derive from a date column and report month/weekday NAMES in the proof.\n"
            "- Always return a full grouped table as proof (not just a single row), then highlight the top item(s) in text.\n"
            "- Never use the `flag_status` column in outputs.\n"
        )

        # 🧠 Task directive
        task_block = (
            "TASK FORMAT:\n"
            "1) A short explanation with exact numbers based on `df`.\n"
            "2) A markdown table with the supporting data (no code blocks).\n"
        )

        formatted_query = (
            (context_block + "\n\n") if context_block else ""
        ) + schema_block + "\n" + task_block + f"\nUSER QUESTION:\n{query}\n"

        # ✅ Ensure df + helpers are available in the tool sandbox
        if hasattr(agent, "tools") and agent.tools:
            try:
                agent.tools[0].locals = {
                    "df": df,
                    "pd": pd,
                    "np": np,
                    "re": re,
                    "datetime": datetime
                }
            except Exception:
                pass

        # Run the agent
        result = agent.invoke({"input": formatted_query})
        response = result.get("output", "").strip()
        result_df = None

        # Prefer a DataFrame (or Series/dict) that the tool actually produced
        for step in result.get("intermediate_steps", []):
            out = step[1]
            if isinstance(out, pd.DataFrame):
                result_df = out
                break
            # Accept common shapes and normalize to DataFrame
            if isinstance(out, pd.Series):
                result_df = out.reset_index()
                result_df.columns = ["Index", "Value"]
                break
            if isinstance(out, dict):
                try:
                    result_df = pd.DataFrame(list(out.items()), columns=["Key", "Value"])
                    break
                except Exception:
                    pass

        # If none, try to parse a markdown table from the LLM text
        if result_df is None:
            result_df = parse_markdown_table(response)

        # Clean + format proof table if present; otherwise fall back to a single-cell DF
        if result_df is not None and not result_df.empty:
            result_df = format_table_for_query(query, result_df)
            result_df = clean_dataframe(result_df)
        else:
            result_df = pd.DataFrame({"Answer": [response]})

        # 🧠 Update conversational memory
        _push_context(query, response)

        return str(response), result_df

    except Exception as e:
        error_msg = f"Error during query: {str(e)}"
        st.error(error_msg)
        return error_msg, None
