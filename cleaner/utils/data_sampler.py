import re
import random
from itertools import combinations
from collections import defaultdict
from typing import Dict, Any, List

import numpy as np
import pandas as pd


class DataSampler:
    """Enhanced data sampler for LLM analysis with better issue detection (dataset-agnostic)."""

    def __init__(self, max_rows: int = 150):
        self.max_rows = max_rows
        random.seed(42)  # reproducible sampling

    # ---------------------------
    # Public API
    # ---------------------------
    def create_sample(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create an intelligent, position-safe sample of the dataset."""
        n = len(df)
        if n <= self.max_rows:
            return {
                'sampled_data': df.copy(),
                'sampling_strategy': 'full_dataset',
                'original_rows': n,
                'sampled_rows': n,
                'sampling_ratio': 1.0
            }

        chosen_positions: List[int] = []

        # 1) Targeted: potential issues (~45% of budget)
        target_issues = max(1, int(self.max_rows * 0.45))
        issue_pos = self._find_potential_issues_positions(df, target_issues)
        self._extend_positions(chosen_positions, issue_pos, self.max_rows)

        # 2) Targeted: contradictory patterns (fill until ~85% total)
        target_contra = max(0, int(self.max_rows * 0.85) - len(chosen_positions))
        if target_contra > 0:
            contra_pos = self._find_contradictory_positions(df, target_contra)
            self._extend_positions(chosen_positions, contra_pos, self.max_rows)

        # 3) Deterministic random fill for diversity
        remaining = self.max_rows - len(chosen_positions)
        if remaining > 0:
            pool = [i for i in range(n) if i not in set(chosen_positions)]
            random_positions = random.sample(pool, k=min(remaining, len(pool)))
            self._extend_positions(chosen_positions, random_positions, self.max_rows)

        # Build sample by **position** so iloc is safe
        sampled_df = df.iloc[chosen_positions].copy().reset_index(drop=True)
        return {
            'sampled_data': sampled_df,
            'sampling_strategy': 'smart_sampling',
            'original_rows': n,
            'sampled_rows': len(sampled_df),
            'sampling_ratio': len(sampled_df) / max(1, n)
        }

    def extract_column_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract simple column summaries for LLMs."""
        column_info: Dict[str, Any] = {}
        for col in df.columns:
            s = df[col]
            info = {
                'dtype': str(s.dtype),
                'null_count': int(s.isnull().sum()),
                'unique_count': int(s.nunique(dropna=True))
            }
            if pd.api.types.is_numeric_dtype(s) and not s.dropna().empty:
                with np.errstate(all='ignore'):
                    info.update({
                        'min': float(np.nanmin(s)),
                        'max': float(np.nanmax(s)),
                        'mean': float(np.nanmean(s))
                    })
            elif s.dtype == 'object':
                vals = s.dropna().astype(str).unique()
                if len(vals) > 0:
                    info['sample_values'] = [vals[i] for i in range(min(3, len(vals)))]
            column_info[col] = info
        return column_info

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _extend_positions(dest: List[int], src: List[int], limit: int) -> None:
        """Append from src into dest keeping order, uniqueness, and <= limit length."""
        seen = set(dest)
        for p in src:
            if p not in seen:
                dest.append(p)
                seen.add(p)
                if len(dest) >= limit:
                    break

    # ---------- Issue finder (positions) ----------
    def _find_potential_issues_positions(self, df: pd.DataFrame, count: int) -> List[int]:
        """Find likely-problematic rows by position: missingness + numeric outliers + fallback."""
        n = len(df)
        if n == 0 or count <= 0:
            return []

        positions: List[int] = []

        # A) Many missing values (top priority)
        nulls_per_row = df.isna().sum(axis=1)
        if len(df.columns) > 0:
            threshold = max(1, int(0.2 * len(df.columns)))  # >20% missing
            high_missing_pos = nulls_per_row[nulls_per_row > threshold].index
            # convert index labels to positions
            high_missing_pos = self._labels_to_positions(df, list(high_missing_pos))
            self._extend_positions(positions, high_missing_pos, count)

        if len(positions) >= count:
            return positions[:count]

        # B) Numeric IQR outliers (robust)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if len(positions) >= count:
                break
            s = df[col]
            s_valid = s.dropna()
            if s_valid.empty:
                continue
            q1, q3 = s_valid.quantile(0.25), s_valid.quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr <= 0:
                continue
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (s < low) | (s > high)
            idx = df.index[mask]
            pos = self._labels_to_positions(df, list(idx))
            # take a slice proportional to remaining budget
            take = max(1, (count - len(positions)) // max(1, len(numeric_cols)))
            self._extend_positions(positions, pos[:take], count)

        if len(positions) >= count:
            return positions[:count]

        # C) Deterministic random fallback
        remaining = count - len(positions)
        if remaining > 0:
            pool = [i for i in range(n) if i not in set(positions)]
            random_pos = random.sample(pool, k=min(remaining, len(pool)))
            self._extend_positions(positions, random_pos, count)

        return positions[:count]

    # ---------- Contradiction finder (positions, dataset-agnostic) ----------
    def _find_contradictory_positions(self, df: pd.DataFrame, count: int) -> List[int]:
        """
        Generic contradiction finder by **position**:
          1) One-hot exclusivity violations among boolean-like groups inferred from column-name tokens.
          2) Text columns that are usually single-choice but contain multiple categories in one cell.
          3) Numeric contradictions: negatives when the distribution is largely >= 0, and 3*IQR outliers.
          4) Fallback: unusual text patterns (mixed casing, repeated spaces, special chars).
        No dataset-specific keywords.
        """
        if count <= 0 or len(df) == 0:
            return []

        positions: List[int] = []

        # 1) Boolean-like groups & one-hot exclusivity
        boolish_map = self._get_boolish_columns(df)
        groups = self._group_binary_onehots(list(boolish_map.keys()))
        for grp in groups:
            if len(positions) >= count:
                break
            mat = pd.DataFrame({c: boolish_map[c].astype(bool) for c in grp})
            per_row = mat.sum(axis=1)  # by original index labels
            # Exclusivity if rows usually have <= 1 active
            if per_row.mean() <= 1.2 and per_row.quantile(0.95) <= 1.0 + 1e-9:
                violated_labels = per_row[per_row > 1].index.tolist()
                violated_pos = self._labels_to_positions(df, violated_labels)
                self._extend_positions(positions, violated_pos, count)

        if len(positions) >= count:
            return positions[:count]

        # 2) Single-choice text columns that contain multiple categories
        obj_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        for col in obj_cols:
            if len(positions) >= count:
                break
            s = df[col]
            if not self._column_is_small_cardinality(s):
                continue
            if not self._likely_single_choice_text_column(s):
                continue

            sv = s.dropna().astype(str).str.strip()
            if sv.empty:
                continue
            # learn top categories
            top = sv.value_counts().head(12).index.tolist()
            if len(top) < 2:
                continue

            sep_pattern = r'[;,|/]|->'
            mask_multi = s.astype(str).str.contains(sep_pattern, na=False)
            candidate = s[mask_multi].astype(str)
            hits_labels = []
            top_lower = {t.lower() for t in top}
            for idx_label, val in candidate.items():
                parts = [p.strip() for p in re.split(sep_pattern, val) if p.strip()]
                if len(parts) <= 1:
                    continue
                distinct_hits = len({p.lower() for p in parts if p.lower() in top_lower})
                if distinct_hits >= 2:
                    hits_labels.append(idx_label)

            self._extend_positions(positions, self._labels_to_positions(df, hits_labels), count)

        if len(positions) >= count:
            return positions[:count]

        # 3) Numeric negatives (when median >= 0) and 3*IQR outliers
        for col in df.select_dtypes(include=[np.number]).columns:
            if len(positions) >= count:
                break
            s = df[col]
            s_valid = s.dropna()
            if s_valid.empty:
                continue

            # Negatives if distribution is predominantly non-negative
            if s_valid.median() >= 0:
                neg_labels = s[s < 0].index.tolist()
                self._extend_positions(positions, self._labels_to_positions(df, neg_labels), count)
                if len(positions) >= count:
                    break

            # 3*IQR robust outliers
            q1, q3 = s_valid.quantile(0.25), s_valid.quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                low, high = q1 - 3 * iqr, q3 + 3 * iqr
                out_labels = s[(s < low) | (s > high)].index.tolist()
                self._extend_positions(positions, self._labels_to_positions(df, out_labels), count)

        if len(positions) >= count:
            return positions[:count]

        # 4) Fallback unusual text patterns
        remaining = count - len(positions)
        if remaining > 0 and obj_cols:
            special_chars = set('@#$%&*<>\\{}[]^~`')
            unusual_labels = []
            for col in obj_cols:
                if len(unusual_labels) >= remaining:
                    break
                ts = df[col].astype(str)
                for i in range(len(ts)):
                    if len(unusual_labels) >= remaining:
                        break
                    val = ts.iloc[i]
                    label = ts.index[i]
                    if (
                        (len(val) > 1 and not (val.islower() or val.isupper() or val.istitle()))
                        or ('  ' in val)
                        or any(ch in special_chars for ch in val)
                    ):
                        unusual_labels.append(label)
            self._extend_positions(positions, self._labels_to_positions(df, unusual_labels), count)

        return positions[:count]

    # ---------- Internals used by contradiction logic ----------
    @staticmethod
    def _normalize_boolish_series(s: pd.Series) -> pd.Series:
        """Map common truthy/falsey encodings to {True, False, NaN}."""
        if s.dtype == bool:
            return s
        sv = s.astype(str).str.strip().str.lower()
        truthy = {'true', 't', '1', 'y', 'yes'}
        falsey = {'false', 'f', '0', 'n', 'no'}
        out = pd.Series(index=s.index, dtype='float')
        out.loc[sv.isin(truthy)] = 1.0
        out.loc[sv.isin(falsey)] = 0.0
        # numeric 0/1 passthrough
        num_mask = s.apply(lambda x: isinstance(x, (int, float, np.integer, np.floating)))
        out.loc[num_mask & s.notna()] = (pd.to_numeric(s[num_mask], errors='coerce') == 1).astype(float)
        return out.map({1.0: True, 0.0: False})

    def _get_boolish_columns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Return columns that are mostly boolean-encodable."""
        out: Dict[str, pd.Series] = {}
        for col in df.columns:
            norm = self._normalize_boolish_series(df[col])
            if len(norm) and norm.notna().mean() >= 0.8:
                out[col] = norm.fillna(False)
        return out

    @staticmethod
    def _tokenize_name(name: str) -> set:
        return {t for t in re.split(r'[_\W]+', name.lower()) if t}

    def _group_binary_onehots(self, boolean_cols: List[str]) -> List[List[str]]:
        """
        Group boolean-like columns likely representing mutually exclusive categories
        by shared non-trivial name tokens.
        """
        tokens_map: Dict[str, set] = defaultdict(set)
        for c in boolean_cols:
            for t in self._tokenize_name(c):
                if len(t) >= 3:
                    tokens_map[t].add(c)

        groups = []
        for t, cols in tokens_map.items():
            if len(cols) >= 2:
                grp = sorted(cols)
                if not any(set(grp).issubset(set(g)) for g in groups):
                    groups.append(grp)

        groups = sorted(groups, key=lambda g: (-len(g), g))
        final, used = [], set()
        for g in groups:
            gg = [c for c in g if c not in used]
            if len(gg) >= 2:
                final.append(gg)
                used.update(gg)
        return final

    @staticmethod
    def _column_is_small_cardinality(s: pd.Series, max_unique: int = 15) -> bool:
        return 1 < s.dropna().astype(str).nunique() <= max_unique

    @staticmethod
    def _likely_single_choice_text_column(s: pd.Series) -> bool:
        sv = s.dropna().astype(str)
        if sv.empty:
            return False
        # Few delimiter-joined entries -> column is likely single-choice
        delim_frac = (sv.str.contains(r'[;,|/]|->')).mean()
        return delim_frac < 0.1

    @staticmethod
    def _labels_to_positions(df: pd.DataFrame, labels: List) -> List[int]:
        """Map index labels to positional indices; keeps order and drops missing."""
        if not labels:
            return []
        idxr = df.index.get_indexer(labels)
        return [int(p) for p in idxr if p != -1]
