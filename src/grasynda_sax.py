import numpy as np
import pandas as pd
from statistics import NormalDist
from typing import Dict, List, Optional

try:
    from pyts.approximation import (
        PiecewiseAggregateApproximation,
        SymbolicAggregateApproximation,
    )
except Exception:
    PiecewiseAggregateApproximation = None
    SymbolicAggregateApproximation = None

from src.grasynda_unified import GrasyndaUnified


class GrasyndaSAX(GrasyndaUnified):
    """
    Grasynda variant that keeps the GrasyndaUnified pipeline but replaces
    quantile discretization with SAX symbolization for transition learning.
    """

    def __init__(
        self,
        period: int,
        n_symbols: int = 25,
        components_to_model: Optional[List[str]] = None,
        component_params: Optional[Dict[str, Dict]] = None,
        sampling_type: str = 'discrete',
        graph_type: str = 'quantile',
        visibility_type: str = 'horizontal',
        apply_differentiation: bool = False,
        ensemble_transitions: bool = False,
        ensemble_size: int = 5,
        n_sax_windows: int = 1,
        sax_normalize: bool = True,
        # Backward-compatibility aliases
        n_quantiles: Optional[int] = None,
        sax_window_size: Optional[int] = None,
    ):
        if n_quantiles is not None:
            n_symbols = int(n_quantiles)
        if sax_window_size is not None:
            n_sax_windows = int(sax_window_size)

        super().__init__(
            period=period,
            n_quantiles=n_symbols,
            components_to_model=components_to_model,
            component_params=component_params,
            sampling_type=sampling_type,
            graph_type=graph_type,
            visibility_type=visibility_type,
            apply_differentiation=apply_differentiation,
            ensemble_transitions=ensemble_transitions,
            ensemble_size=ensemble_size,
        )
        self.alias = 'GrasyndaSAX'
        self.n_symbols = int(n_symbols)
        self.n_sax_windows = max(1, int(n_sax_windows))
        self.sax_normalize = bool(sax_normalize)

    @staticmethod
    def _sax_breakpoints(alphabet_size: int) -> np.ndarray:
        if alphabet_size <= 1:
            return np.array([], dtype=float)
        normal = NormalDist()
        probs = np.arange(1, alphabet_size) / alphabet_size
        return np.array([normal.inv_cdf(float(p)) for p in probs], dtype=float)

    def _get_sax_param(self, component: str, param: str):
        if component in self.component_params:
            comp_cfg = self.component_params[component]
            if param in comp_cfg:
                return comp_cfg[param]
            # Support old key names in component overrides.
            if param == 'n_sax_windows' and 'sax_window_size' in comp_cfg:
                return comp_cfg['sax_window_size']
            if param == 'n_symbols' and 'n_quantiles' in comp_cfg:
                return comp_cfg['n_quantiles']
        return getattr(self, param)

    def _sax_symbolize_series(
        self,
        values: np.ndarray,
        n_symbols: int,
        window_size: int,
        normalize: bool,
    ) -> np.ndarray:
        n = len(values)
        if n == 0:
            return np.array([], dtype=int)
        if n_symbols <= 1:
            return np.zeros(n, dtype=int)

        # pyts with ordinal alphabet is bounded by 26 bins.
        n_symbols = max(2, min(int(n_symbols), 26))

        arr = np.asarray(values, dtype=float)
        finite_mask = np.isfinite(arr)
        if not finite_mask.any():
            return np.zeros(n, dtype=int)
        if not finite_mask.all():
            fill_value = float(np.mean(arr[finite_mask]))
            arr = arr.copy()
            arr[~finite_mask] = fill_value

        if normalize:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
            if sigma > 0:
                arr = (arr - mu) / sigma
            else:
                arr = np.zeros_like(arr)

        w = max(1, int(window_size))
        X = arr.reshape(1, -1)

        # Preferred backend: pyts
        if (
            PiecewiseAggregateApproximation is not None
            and SymbolicAggregateApproximation is not None
        ):
            sax = SymbolicAggregateApproximation(
                n_bins=n_symbols,
                strategy='normal',
                alphabet='ordinal',
            )

            if w == 1:
                return sax.transform(X)[0].astype(int)

            paa = PiecewiseAggregateApproximation(window_size=w)
            X_paa = paa.transform(X)
            paa_symbols = sax.transform(X_paa)[0].astype(int)

            starts = np.arange(0, n, w)
            ends = np.minimum(starts + w, n)
            seg_sizes = ends - starts

            # Keep behavior robust if pyts segment count differs on edge cases.
            if len(paa_symbols) != len(seg_sizes):
                n_seg = len(paa_symbols)
                base = n // n_seg
                rem = n % n_seg
                seg_sizes = np.array(
                    [base + (1 if i < rem else 0) for i in range(n_seg)],
                    dtype=int,
                )

            expanded = np.repeat(paa_symbols, seg_sizes)
            if len(expanded) > n:
                expanded = expanded[:n]
            elif len(expanded) < n:
                expanded = np.pad(expanded, (0, n - len(expanded)), mode='edge')
            return expanded.astype(int)

        # Fallback: previous manual implementation if pyts is unavailable.
        if w == 1:
            paa_values = arr
            seg_sizes = np.ones(n, dtype=int)
        else:
            starts = np.arange(0, n, w)
            ends = np.minimum(starts + w, n)
            seg_sizes = (ends - starts).astype(int)
            paa_values = np.array([arr[s:e].mean() for s, e in zip(starts, ends)], dtype=float)

        breakpoints = self._sax_breakpoints(n_symbols)
        paa_symbols = np.digitize(paa_values, breakpoints, right=False).astype(int)

        if w == 1:
            return paa_symbols

        expanded = np.repeat(paa_symbols, seg_sizes)
        if len(expanded) > n:
            expanded = expanded[:n]
        elif len(expanded) < n:
            expanded = np.pad(expanded, (0, n - len(expanded)), mode='edge')
        return expanded.astype(int)

    def _get_quantiles(self, df: pd.DataFrame, target_col: str, component: str):
        n_symbols = self._get_sax_param(component, 'n_symbols')
        if n_symbols is None:
            n_symbols = self._get_param(component, 'n_quantiles')
        n_sax_windows = self._get_sax_param(component, 'n_sax_windows')
        sax_normalize = self._get_sax_param(component, 'sax_normalize')

        symbols = pd.Series(index=df.index, dtype='float64')
        for _, group in df.groupby('unique_id', sort=False):
            uid_symbols = self._sax_symbolize_series(
                values=group[target_col].to_numpy(),
                n_symbols=int(n_symbols),
                window_size=int(n_sax_windows),
                normalize=bool(sax_normalize),
            )
            symbols.loc[group.index] = uid_symbols

        return symbols.reindex(df.index).fillna(0).astype(int)
