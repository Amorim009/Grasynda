
import copy
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from itertools import combinations

from statsmodels.tsa.seasonal import STL
from scipy.stats import gaussian_kde
import networkx as nx

from metaforecast.synth.generators.base import SemiSyntheticGenerator

class GrasyndaUnified(SemiSyntheticGenerator):
    """
    Unified Grasynda Generator capable of replicating all variants:
    - Standard (Discrete Remainder)
    - Trend (Continuous Differentiated Trend)
    - Continuous (Original Series)
    - Hybrid (Trend + Remainder)
    - Visibility Graph (Quantile or Visibility based)
    """
    
    def __init__(self,
                 period: int,
                 n_quantiles: int = 25,
                 # Configuration
                 components_to_model: List[str] = None, # Default: ['remainder']
                 component_params: Dict[str, Dict] = None,
                 # Global Defaults (applied if not in component_params)
                 sampling_type: str = 'discrete', # 'discrete', 'continuous_uniform', 'kde'
                 graph_type: str = 'quantile', # 'quantile', 'visibility'
                 visibility_type: str = 'horizontal', # 'horizontal', 'natural'
                 apply_differentiation: bool = False, # For trend
                 ensemble_transitions: bool = False,
                 ensemble_size: int = 5,
                 robust: bool = False):
        
        super().__init__(alias='GrasyndaUnified')
        
        self.period = period
        self.n_quantiles = n_quantiles
        self.robust = robust
        
        # Defaults
        self.components_to_model = components_to_model if components_to_model else ['remainder']
        self.component_params = component_params if component_params else {}
        
        # Global Settings
        self.global_settings = {
            'n_quantiles': n_quantiles,
            'sampling_type': sampling_type,
            'graph_type': graph_type,
            'visibility_type': visibility_type,
            'apply_differentiation': apply_differentiation,
            'ensemble_transitions': ensemble_transitions,
            'ensemble_size': ensemble_size
        }
        
        # State storage
        self.transition_mats = {}
        self.ensemble_transition_mats = {}
        self.uid_pw_distance = {}
        
        # Visibility Graph specific storage
        self.visibility_graphs = {}
        self.degree_distributions = {}
        self.degree_transitions = {}
        self.value_degree_map = {}

    def _create_synthetic_ts(self, df: pd.DataFrame) -> Dict:
        """
        Implementation of abstract method from SemiSyntheticGenerator.
        This acts as a wrapper to delegate to the specific generation logic
        based on the FIRST component in components_to_model (or default).
        """
        component = self.components_to_model[0]
        graph_type = self._get_param(component, 'graph_type')
        
        if graph_type == 'quantile':
            return self._create_synthetic_ts_quantile(
                df, component, 
                sampling_type=self._get_param(component, 'sampling_type')
            )
        elif graph_type == 'visibility':
            return self._generate_synthetic_series_vis(df, component)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")

    def _get_param(self, component: str, param: str):
        """Get parameter for a specific component, falling back to global settings."""
        if component in self.component_params and param in self.component_params[component]:
            return self.component_params[component][param]
        return self.global_settings.get(param)

    def transform(self, df: pd.DataFrame, return_components: bool = False, **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        
        # 1. Determine if decomposition is needed
        # If we are modeling 'y' (original), we skip decomposition
        skip_decomposition = 'y' in self.components_to_model
        
        df_ = df.copy()
        
        if not skip_decomposition:
            df_ = self.decompose_tsd(df_, period=self.period, robust=self.robust)
        
        # Dictionary to store generated components
        synth_components = {}
        
        # 2. Iterate through components to model
        for component in self.components_to_model:
            # Prepare data for this component
            comp_df = df_.copy()
            
            # Handle Differentiation (e.g., for Trend)
            is_diff = False
            if self._get_param(component, 'apply_differentiation'):
                comp_df = self._differentiate_component(comp_df, component)
                target_col = f'diff_{component}'
                is_diff = True
            else:
                target_col = component
            
            # Check if target column exists
            if target_col not in comp_df.columns:
                # If modeling 'y' but it's not in columns (should be there), or other error
                if component == 'y' and 'y' in comp_df.columns:
                     pass # y is there
                elif component in ['trend', 'seasonal', 'remainder'] and not skip_decomposition:
                    pass # components are there
                else:
                    raise ValueError(f"Component '{target_col}' not found in DataFrame.")

            # --- Graph & Transition Logic ---
            graph_type = self._get_param(component, 'graph_type')
            
            if graph_type == 'quantile':
                # Quantile-based Logic
                comp_df['Quantile'] = self._get_quantiles(comp_df, target_col, component=component)
                self._calc_transition_matrix(comp_df, component=component)
                
                if self._get_param(component, 'ensemble_transitions'):
                    self.ensemble_transition_mats[component] = self._get_ensemble_transition_mats(
                        component=component,
                        ensemble_size=self._get_param(component, 'ensemble_size')
                    )
                
                # Generate
                synth_dict = self._create_synthetic_ts_quantile(
                    comp_df, component, target_col,
                    sampling_type=self._get_param(component, 'sampling_type')
                )
                
            elif graph_type == 'visibility':
                # Visibility Graph Logic
                vis_type = self._get_param(component, 'visibility_type')
                self._learn_visibility_patterns(comp_df, target_col, vis_type)
                
                if self._get_param(component, 'ensemble_transitions'):
                    self.ensemble_transition_mats[component] = self._get_ensemble_visibility_mats(
                        component=component,
                        ensemble_size=self._get_param(component, 'ensemble_size')
                    )
                
                synth_dict = self._generate_synthetic_series_vis(comp_df, target_col)
            
            else:
                raise ValueError(f"Unknown graph_type: {graph_type}")
            
            # --- Integration (if differentiated) ---
            if is_diff:
                synth_dict = self._integrate_component(comp_df, component, synth_dict)
            
            synth_components[component] = synth_dict
            
        # 3. Reconstruction
        synth_df = self._reconstruct(df_, synth_components, skip_decomposition)
        
        if return_components:
            return synth_df, synth_components
            
        return synth_df

    # =========================================================================
    # DECOMPOSITION & RECONSTRUCTION
    # =========================================================================
    
    @staticmethod
    def decompose_tsd(df: pd.DataFrame, period: int, robust: bool):
        seasonal_components = []
        trend_components = []
        remainder_components = []

        for unique_id, group in df.groupby('unique_id'):
            stl = STL(group['y'], period=period, robust=robust)
            result = stl.fit()

            seasonal_components.append(pd.DataFrame({
                'unique_id': unique_id, 'ds': group['ds'], 'seasonal': result.seasonal
            }))
            trend_components.append(pd.DataFrame({
                'unique_id': unique_id, 'ds': group['ds'], 'trend': result.trend
            }))
            remainder_components.append(pd.DataFrame({
                'unique_id': unique_id, 'ds': group['ds'], 'remainder': result.resid
            }))

        seasonal_df = pd.concat(seasonal_components)
        trend_df = pd.concat(trend_components)
        remainder_df = pd.concat(remainder_components)

        decomposed_df = pd.merge(seasonal_df, trend_df, on=['unique_id', 'ds'])
        decomposed_df = pd.merge(decomposed_df, remainder_df, on=['unique_id', 'ds'])
        # Merge back y if needed, though usually we start with df that has y
        if 'y' in df.columns:
             decomposed_df = pd.merge(decomposed_df, df[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'])

        return decomposed_df

    def _reconstruct(self, df: pd.DataFrame, synth_components: Dict, skip_decomposition: bool):
        synth_list = []
        
        for uid, uid_df in df.groupby('unique_id'):
            row = uid_df.copy()
            
            # Replace components with synthetic ones
            for comp, synth_dict in synth_components.items():
                if uid in synth_dict:
                    row[comp] = synth_dict[uid].values
            
            # Calculate final y
            if skip_decomposition:
                # If we modeled 'y' directly, that is our result
                if 'y' not in synth_components:
                     # This shouldn't happen if skip_decomposition is True unless config is wrong
                     pass 
            else:
                # Sum components
                # Use synthetic if available, else original
                trend = row['trend']
                seasonal = row['seasonal']
                remainder = row['remainder']
                row['y'] = trend + seasonal + remainder
            
            synth_list.append(row)
            
        synth_df = pd.concat(synth_list)
        
        # Cleanup columns
        cols_to_keep = ['ds', 'unique_id', 'y']
        synth_df = synth_df[cols_to_keep]
        
        # Update unique_id
        synth_df['unique_id'] = synth_df['unique_id'].apply(lambda x: f'{self.alias}_{x}')
        
        return synth_df

    # =========================================================================
    # DIFFERENTIATION LOGIC
    # =========================================================================

    def _differentiate_component(self, df: pd.DataFrame, component: str) -> pd.DataFrame:
        """Add a 'diff_{component}' column."""
        diff_list = []
        for uid, group in df.groupby('unique_id'):
            series = group[component].values
            # First-order difference, preserving length
            first_diff = series[1] - series[0] if len(series) > 1 else 0
            diff_series = np.diff(series, prepend=series[0] - first_diff)
            
            diff_list.append(pd.DataFrame({
                'unique_id': uid,
                'ds': group['ds'],
                f'diff_{component}': diff_series
            }))
        
        diff_df = pd.concat(diff_list)
        return pd.merge(df, diff_df, on=['unique_id', 'ds'])

    def _integrate_component(self, df: pd.DataFrame, component: str, synth_diff_dict: Dict) -> Dict:
        """Integrate synthetic diff component back to original scale."""
        synth_integrated_dict = {}
        for uid, uid_df in df.groupby('unique_id'):
            synth_diff = synth_diff_dict[uid].values
            original_start = uid_df[component].values[0]
            
            synth_vals = np.zeros(len(synth_diff))
            synth_vals[0] = original_start
            synth_vals[1:] = original_start + np.cumsum(synth_diff[1:])
            
            synth_integrated_dict[uid] = pd.Series(synth_vals, index=uid_df.index)
        return synth_integrated_dict

    # =========================================================================
    # QUANTILE GRAPH LOGIC
    # =========================================================================

    def _get_quantiles(self, df: pd.DataFrame, target_col: str, component: str):
        n_q = self._get_param(component, 'n_quantiles')
        return df.groupby('unique_id')[target_col].transform(
            lambda x: pd.qcut(x, n_q, labels=False, duplicates='drop')
        ).fillna(0).astype(int)

    def _calc_transition_matrix(self, df: pd.DataFrame, component: str):
        if component not in self.transition_mats:
            self.transition_mats[component] = {}
        n_q = self._get_param(component, 'n_quantiles')
        for unique_id, group in df.groupby('unique_id'):
            quantiles = group['Quantile'].values
            t_count_matrix = np.zeros((n_q, n_q))
            
            for i in range(len(quantiles) - 1):
                t_count_matrix[quantiles[i], quantiles[i + 1]] += 1
                
            # Normalize
            with np.errstate(divide='ignore', invalid='ignore'):
                t_prob_matrix = t_count_matrix / t_count_matrix.sum(axis=1, keepdims=True)
            t_prob_matrix = np.nan_to_num(t_prob_matrix)
            
            # Handle empty rows (uniform)
            for row in range(n_q):
                if np.sum(t_prob_matrix[row]) == 0:
                    t_prob_matrix[row] = np.ones(n_q) / n_q
            
            self.transition_mats[component][unique_id] = t_prob_matrix

    def _get_ensemble_transition_mats(self, component: str, ensemble_size: int):
        mats = copy.deepcopy(self.transition_mats[component])
        uid_pairs = combinations([*mats], 2)
        
        # Calculate distances
        for uid in mats: self.uid_pw_distance[(uid, uid)] = 0.0
        for uid1, uid2 in uid_pairs:
            dist = np.linalg.norm(mats[uid1] - mats[uid2])
            self.uid_pw_distance[(uid1, uid2)] = dist
            self.uid_pw_distance[(uid2, uid1)] = dist
            
        ensemble_mats = {}
        for uid in mats:
            uid_dists = pd.Series({other: self.uid_pw_distance[(uid, other)] for other in mats})
            similar_uids = uid_dists.sort_values().head(ensemble_size).index.tolist()
            ensemble_mats[uid] = np.sum([mats[u] for u in similar_uids], axis=0) / len(similar_uids)
            
        return ensemble_mats

    def _get_ensemble_visibility_mats(self, component: str, ensemble_size: int):
        """Find similar UIDs based on degree transitions and average their transition maps."""
        uids = list(self.degree_transitions.keys())
        
        # 1. Convert probability maps to matrices for distance calculation
        # Use max degree across all series to define common matrix size
        all_degs = []
        for d_t in self.degree_transitions.values():
            all_degs.extend(d_t['unique'])
        max_deg = int(max(all_degs)) if all_degs else 20
        n_d = max_deg + 1
        
        mats = {}
        for uid in uids:
            mat = np.zeros((n_d, n_d))
            for deg, trans in self.degree_transitions[uid]['probs'].items():
                for ndeg, prob in trans.items():
                    if deg < n_d and ndeg < n_d:
                        mat[int(deg), int(ndeg)] = prob
            mats[uid] = mat
            
        # 2. Pairwise distances between transition matrices
        for uid in uids: self.uid_pw_distance[(uid, uid)] = 0.0
        for uid1, uid2 in combinations(uids, 2):
            dist = np.linalg.norm(mats[uid1] - mats[uid2])
            self.uid_pw_distance[(uid1, uid2)] = dist
            self.uid_pw_distance[(uid2, uid1)] = dist
            
        ensemble_mats = {}
        for uid in uids:
            uid_dists = pd.Series({other: self.uid_pw_distance[(uid, other)] for other in uids})
            similar_uids = uid_dists.sort_values().head(ensemble_size).index.tolist()
            
            # Average transition probabilities
            merged_probs = {}
            all_degrees = set()
            for u in similar_uids:
                all_degrees.update(self.degree_transitions[u]['probs'].keys())
                
            for deg in all_degrees:
                deg_trans = {}
                count = 0
                for u in similar_uids:
                    if deg in self.degree_transitions[u]['probs']:
                        count += 1
                        for ndeg, prob in self.degree_transitions[u]['probs'][deg].items():
                            deg_trans[ndeg] = deg_trans.get(ndeg, 0) + prob
                if count > 0:
                    total = sum(deg_trans.values())
                    merged_probs[deg] = {k: v / total for k, v in deg_trans.items()}
            
            ensemble_mats[uid] = {
                'unique': np.unique(list(merged_probs.keys())),
                'probs': merged_probs
            }
            
        return ensemble_mats

    def _generate_quantile_series(self, df: pd.DataFrame, component: str):
        quantile_series = {}
        uids = df['unique_id'].unique().tolist()
        
        for uid in uids:
            if component in self.ensemble_transition_mats and uid in self.ensemble_transition_mats[component]:
                mat = self.ensemble_transition_mats[component][uid]
            else:
                mat = self.transition_mats[component][uid]
            uid_df = df.query(f'unique_id=="{uid}"')
            
            q_series = np.zeros(len(uid_df), dtype=int)
            q_series[0] = uid_df['Quantile'].values[0]
            
            n_q = self._get_param(component, 'n_quantiles')
            for t in range(1, len(q_series)):
                probs = mat[q_series[t-1]]
                # Normalize if needed (should be already, but safety check)
                if np.sum(probs) == 0: probs = np.ones(n_q) / n_q
                else: probs = probs / np.sum(probs)
                
                q_series[t] = np.random.choice(np.arange(n_q), p=probs)
            
            quantile_series[uid] = q_series
        return quantile_series

    def _create_synthetic_ts_quantile(self, df: pd.DataFrame, component: str, target_col: str, sampling_type: str) -> Dict:
        quantile_series = self._generate_quantile_series(df, component)
        generated_ts = {}
        
        uids = df['unique_id'].unique().tolist()
        for uid in uids:
            uid_df = df.query(f'unique_id=="{uid}"')
            uid_vals = uid_df[target_col]
            uid_quantiles = uid_df['Quantile']
            
            # Pre-calculate bin properties
            bin_props = {}
            n_q = self._get_param(component, 'n_quantiles')
            for q in range(n_q):
                vals = uid_vals[uid_quantiles == q].values
                if len(vals) > 0:
                    bin_props[q] = {
                        'vals': vals,
                        'min': vals.min(),
                        'max': vals.max(),
                        'kde': gaussian_kde(vals) if len(vals) > 1 and sampling_type == 'kde' else None
                    }
                else:
                    bin_props[q] = None

            synth_vals = np.zeros(len(uid_vals))
            synth_vals[0] = uid_vals.values[0]
            
            for i in range(1, len(uid_vals)):
                q = quantile_series[uid][i]
                props = bin_props.get(q)
                
                if props is None:
                    synth_vals[i] = synth_vals[i-1]
                    continue
                
                if sampling_type == 'discrete':
                    synth_vals[i] = np.random.choice(props['vals'])
                elif sampling_type == 'continuous_uniform':
                    if props['min'] == props['max']:
                        synth_vals[i] = props['min']
                    else:
                        synth_vals[i] = np.random.uniform(props['min'], props['max'])
                elif sampling_type == 'kde':
                    if props['kde']:
                        synth_vals[i] = props['kde'].resample(1)[0][0]
                    else:
                        synth_vals[i] = np.random.choice(props['vals']) # Fallback
                else:
                    raise ValueError(f"Unknown sampling_type: {sampling_type}")
            
            generated_ts[uid] = pd.Series(synth_vals, index=uid_df.index)
            
        return generated_ts

    # =========================================================================
    # VISIBILITY GRAPH LOGIC
    # =========================================================================

    
    # (Re-implementing VG logic briefly)
    @staticmethod
    def _horizontal_visibility(series: np.ndarray) -> np.ndarray:
        n = len(series)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                min_h = min(series[i], series[j])
                visible = True
                for k in range(i + 1, j):
                    if series[k] >= min_h:
                        visible = False; break
                if visible: adj[i, j] = adj[j, i] = 1
        return adj

    @staticmethod
    def _natural_visibility(series: np.ndarray) -> np.ndarray:
        n = len(series)
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                visible = True
                for k in range(i + 1, j):
                    if series[k] >= series[i] + (series[j] - series[i]) * (k - i) / (j - i):
                        visible = False; break
                if visible: adj[i, j] = adj[j, i] = 1
        return adj

    def _learn_visibility_patterns(self, df: pd.DataFrame, target_col: str, vis_type: str):
        for uid, group in df.groupby('unique_id'):
            series = group[target_col].values
            if vis_type == 'horizontal': adj = self._horizontal_visibility(series)
            else: adj = self._natural_visibility(series)
            
            degrees = adj.sum(axis=1)
            self.degree_distributions[uid] = degrees
            
            # Map degrees -> values
            self.value_degree_map[uid] = {}
            for v, d in zip(series, degrees):
                if d not in self.value_degree_map[uid]: self.value_degree_map[uid][d] = []
                self.value_degree_map[uid][d].append(v)
            
            # Transition Matrix
            u_deg = np.unique(degrees)
            counts = {d: {} for d in u_deg}
            for i in range(len(degrees)-1):
                c, n = degrees[i], degrees[i+1]
                counts[c][n] = counts[c].get(n, 0) + 1
            
            probs = {}
            for d, trans in counts.items():
                total = sum(trans.values())
                if total > 0: probs[d] = {k: v/total for k,v in trans.items()}
                else: probs[d] = {k: 1/len(u_deg) for k in u_deg}
            
            self.degree_transitions[uid] = {'unique': u_deg, 'probs': probs}

    def _generate_synthetic_series_vis(self, df: pd.DataFrame, target_col: str) -> Dict:
        gen_series = {}
        # Get sampling type for this component
        # Always strip 'diff_' to find the true component config key
        base_component = target_col.replace('diff_', '')
        sampling_type = self._get_param(base_component, 'sampling_type')
        if not sampling_type: sampling_type = 'discrete'

        for uid, group in df.groupby('unique_id'):
            n = len(group)
            
            # Use ensemble if available
            if base_component in self.ensemble_transition_mats and uid in self.ensemble_transition_mats[base_component]:
                trans = self.ensemble_transition_mats[base_component][uid]
            else:
                trans = self.degree_transitions[uid]
                
            curr_deg = np.random.choice(self.degree_distributions[uid])
            
            synth_vals = np.zeros(n)
            
            for i in range(n):
                # Sample value for current degree
                vals_in_bin = []
                if curr_deg in self.value_degree_map[uid]:
                    vals_in_bin = self.value_degree_map[uid][curr_deg]
                else:
                    # Nearest neighbor fallback
                    avail = list(self.value_degree_map[uid].keys())
                    if avail:
                        nearest = min(avail, key=lambda x: abs(x - curr_deg))
                        vals_in_bin = self.value_degree_map[uid][nearest]
                
                if not vals_in_bin:
                    val = 0 # Should not happen if map populated
                else:
                    if sampling_type == 'discrete':
                        val = np.random.choice(vals_in_bin)
                    elif sampling_type == 'continuous_uniform':
                        # Mix-Max of that degree bin
                        min_v, max_v = min(vals_in_bin), max(vals_in_bin)
                        if min_v == max_v: val = min_v
                        else: val = np.random.uniform(min_v, max_v)
                    else:
                        # Fallback
                        val = np.random.choice(vals_in_bin)
                        
                synth_vals[i] = val
                
                # Transition
                if i < n-1:
                    if curr_deg in trans['probs']:
                        opts = list(trans['probs'][curr_deg].keys())
                        p = list(trans['probs'][curr_deg].values())
                        curr_deg = np.random.choice(opts, p=p)
                    else:
                        curr_deg = np.random.choice(trans['unique'])
            
            gen_series[uid] = pd.Series(synth_vals, index=group.index)
        return gen_series
