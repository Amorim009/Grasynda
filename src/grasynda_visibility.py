

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import networkx as nx
from statsmodels.tsa.seasonal import STL
from metaforecast.synth.generators.base import SemiSyntheticGenerator


class VisibilityGraph:
  
    @staticmethod
    def natural_visibility(series: np.ndarray) -> np.ndarray:
        """
        Construct Natural Visibility Graph adjacency matrix.
        
        Two points (t_a, y_a) and (t_b, y_b) are connected if all intermediate
        points satisfy: y_c < y_a + (y_b - y_a) * (t_c - t_a) / (t_b - t_a)
        """
        n = len(series)
        adj_matrix = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                visible = True
                for k in range(i + 1, j):
                    y_line = series[i] + (series[j] - series[i]) * (k - i) / (j - i)
                    if series[k] >= y_line:
                        visible = False
                        break
                
                if visible:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        return adj_matrix
    
    @staticmethod
    def horizontal_visibility(series: np.ndarray) -> np.ndarray:
        """
        Construct Horizontal Visibility Graph adjacency matrix.
        
        Two points are connected if all intermediate points are below min(y_a, y_b).
        """
        n = len(series)
        adj_matrix = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(i + 1, n):
                min_height = min(series[i], series[j])
                visible = True
                
                for k in range(i + 1, j):
                    if series[k] >= min_height:
                        visible = False
                        break
                
                if visible:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1
        
        return adj_matrix
    
    @staticmethod
    def extract_degree_sequence(adj_matrix: np.ndarray) -> np.ndarray:
        """Extract degree sequence from adjacency matrix."""
        return adj_matrix.sum(axis=1)
    



class GrasyndaVisibilityGraph(SemiSyntheticGenerator):


    
    def __init__(self,
                 period: int = None,
                 visibility_type: str = 'horizontal',
                 quantile_on: str = 'remainder',
                 use_decomposition: bool = True,
                 robust: bool = False):
        """
        Initialize GrasyndaVisibilityGraph.
        
        Args:
            period: Seasonal period for STL decomposition (only needed if use_decomposition=True)
            visibility_type: 'natural' or 'horizontal'
            quantile_on: Which component to apply VG to: 'trend', 'remainder', or 'raw'
            use_decomposition: If True, use STL decomposition. If False, work directly on raw data
            robust: Use robust STL decomposition
        """
        super().__init__(alias='GrasyndaVG')
        
        self.period = period
        self.visibility_type = visibility_type
        self.quantile_on = quantile_on
        self.use_decomposition = use_decomposition
        self.robust = robust
        
        if use_decomposition and period is None:
            raise ValueError("period must be specified when use_decomposition=True")
        
        # Storage for learned patterns
        self.visibility_graphs = {}
        self.degree_distributions = {}
        self.degree_transitions = {}  # Transition matrices: P(degree_t+1 | degree_t)
        self.value_degree_map = {}  # Maps degrees to values with those degrees
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generate synthetic time series using visibility graphs."""
        
        # Decompose or use raw
        if self.use_decomposition:
            df_decomp = self._decompose(df)
            target_column = self.quantile_on  # 'trend' or 'remainder'
        else:
            df_decomp = df.copy()
            df_decomp['raw_y'] = df_decomp['y']
            target_column = 'raw_y'
        
        # Step 2: Build visibility graphs and learn patterns
        self._learn_visibility_patterns(df_decomp, target_column)
        
        # Step 3: Generate synthetic series
        synth_ts_dict = self._generate_synthetic_series(df_decomp, target_column)
        
        # Step 4: Reconstruct and format
        if self.use_decomposition:
            synth_df = self._reconstruct_decomposed(df_decomp, synth_ts_dict)
        else:
            synth_df = self._reconstruct_raw(df_decomp, synth_ts_dict)
        
        return synth_df
    
    def _decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """STL decomposition of time series."""
        components = []
        
        for unique_id, group in df.groupby('unique_id'):
            stl = STL(group['y'], period=self.period, robust=self.robust)
            result = stl.fit()
            
            components.append(pd.DataFrame({
                'unique_id': unique_id,
                'ds': group['ds'],
                'y': group['y'],
                'trend': result.trend,
                'seasonal': result.seasonal,
                'remainder': result.resid
            }))
        
        return pd.concat(components, ignore_index=True)
    
    def _learn_visibility_patterns(self, df: pd.DataFrame, target_column: str):
        """Build visibility graphs and extract patterns."""
        
        print(f"Building {self.visibility_type} visibility graphs on '{target_column}'...")
        
        for unique_id, group in df.groupby('unique_id'):
            series = group[target_column].values
            
            # Build visibility graph
            if self.visibility_type == 'horizontal':
                adj_matrix = VisibilityGraph.horizontal_visibility(series)
            else:
                adj_matrix = VisibilityGraph.natural_visibility(series)
            
            self.visibility_graphs[unique_id] = adj_matrix
            
            # Extract degree sequence
            degrees = VisibilityGraph.extract_degree_sequence(adj_matrix)
            self.degree_distributions[unique_id] = degrees
            
            # Map degrees→values
            self.value_degree_map[unique_id] = {}
            for i, (val, deg) in enumerate(zip(series, degrees)):
                if deg not in self.value_degree_map[unique_id]:
                    self.value_degree_map[unique_id][deg] = []
                self.value_degree_map[unique_id][deg].append(val)
            
            # Build transition matrix
            self.degree_transitions[unique_id] = self._build_transition_matrix(degrees)
    
    def _create_synthetic_ts(self, df: pd.DataFrame) -> Dict:
        """
        Implementation of abstract method from SemiSyntheticGenerator.
        """
        if self.use_decomposition:
            target_column = 'remainder'
        else:
            target_column = 'raw_y'
            if target_column not in df.columns and 'y' in df.columns:
                df[target_column] = df['y']
                
        return self._generate_synthetic_series(df, target_column)

    def _generate_synthetic_series(self, df: pd.DataFrame, target_column: str) -> Dict:
        """Generate new series using visibility graph degree matching."""
        
        generated_series = {}
        
        for unique_id, group in df.groupby('unique_id'):
            series = group[target_column].values
            n = len(series)
            
            # Use degree matching to generate synthetic series
            synth = self._generate_by_degree_matching(unique_id, n)
            generated_series[unique_id] = pd.Series(synth, index=group.index)
        
        return generated_series
    
    def _build_transition_matrix(self, degrees: np.ndarray) -> Dict:
        """
        Build transition matrix for degree sequence.
        
        Returns a dictionary where:
        - transition_matrix[degree_from][degree_to] = count
        - transition_probs[degree_from] = probability distribution over next degrees
        """
        unique_degrees = np.unique(degrees)
        transition_counts = {deg: {} for deg in unique_degrees}
        
        # Count transitions
        for i in range(len(degrees) - 1):
            current_deg = degrees[i]
            next_deg = degrees[i + 1]
            
            if next_deg not in transition_counts[current_deg]:
                transition_counts[current_deg][next_deg] = 0
            transition_counts[current_deg][next_deg] += 1
        
        # Counts→probabilities
        transition_probs = {}
        for deg_from, trans_dict in transition_counts.items():
            if trans_dict:
                total = sum(trans_dict.values())
                transition_probs[deg_from] = {
                    deg_to: count / total 
                    for deg_to, count in trans_dict.items()
                }
            else:
                # Fallback: uniform
                transition_probs[deg_from] = {
                    deg_to: 1.0 / len(unique_degrees) 
                    for deg_to in unique_degrees
                }
        
        return {
            'unique_degrees': list(unique_degrees),
            'transition_probs': transition_probs
        }
    
    def _generate_by_degree_matching(self, uid: str, length: int) -> np.ndarray:
        """
        Generate synthetic series using degree transition matrix.
        
        Algorithm:
        1. Start with a random degree from the original distribution
        2. Use transition matrix to generate next degree probabilistically
        3. For each degree, select a value that had that degree in the original series
        
        Returns:
            Synthetic time series of specified length
        """
        synth = np.zeros(length)
        
        # Get transition data
        transition_info = self.degree_transitions[uid]
        unique_degrees = transition_info['unique_degrees']
        transition_probs = transition_info['transition_probs']
        
        # Initialize
        original_degrees = self.degree_distributions[uid]
        current_degree = np.random.choice(original_degrees)
        
        
        sampled_degrees = [current_degree]
        
        for i in range(1, length):
            # Next degree
            if current_degree in transition_probs:
                next_degrees = list(transition_probs[current_degree].keys())
                next_probs = list(transition_probs[current_degree].values())
                current_degree = np.random.choice(next_degrees, p=next_probs)
            else:
                # Fallback sampling
                current_degree = np.random.choice(unique_degrees)
            
            sampled_degrees.append(current_degree)
        
        # Degrees→values mapping
        for i, target_degree in enumerate(sampled_degrees):
            if target_degree in self.value_degree_map[uid]:
                possible_values = self.value_degree_map[uid][target_degree]
                synth[i] = np.random.choice(possible_values)
            else:
                # Find nearest
                available_degrees = list(self.value_degree_map[uid].keys())
                nearest = min(available_degrees, key=lambda x: abs(x - target_degree))
                synth[i] = np.random.choice(self.value_degree_map[uid][nearest])
        
        return synth
    

    
    def _reconstruct_decomposed(self, df_decomp: pd.DataFrame, synth_dict: Dict) -> pd.DataFrame:
        """Reconstruct synthetic time series from decomposed components."""
        
        synth_list = []
        
        for unique_id, group in df_decomp.groupby('unique_id'):
            synth_component = synth_dict[unique_id].values
            
            synth_row = group.copy()
            
            # Replace target component
            if self.quantile_on == 'trend':
                synth_row['trend'] = synth_component
            else:  # 'remainder'
                synth_row['remainder'] = synth_component
            
            synth_row['y'] = (synth_row['trend'] + 
                             synth_row['seasonal'] + 
                             synth_row['remainder'])
            
            synth_row['unique_id'] = f'{self.alias}_{unique_id}'
            synth_row = synth_row[['ds', 'unique_id', 'y']]
            
            synth_list.append(synth_row)
        
        return pd.concat(synth_list, ignore_index=True)
    
    def _reconstruct_raw(self, df: pd.DataFrame, synth_dict: Dict) -> pd.DataFrame:
        """Reconstruct synthetic time series from raw generated values."""
        
        synth_list = []
        
        for unique_id, group in df.groupby('unique_id'):
            synth_y = synth_dict[unique_id].values
            
            synth_row = group[['ds']].copy()
            synth_row['unique_id'] = f'{self.alias}_{unique_id}'
            synth_row['y'] = synth_y
            
            synth_list.append(synth_row)
        
        return pd.concat(synth_list, ignore_index=True)
    
    def get_graph_statistics(self, uid: str) -> Dict:
        """Get visibility graph statistics for a time series."""
        if uid not in self.visibility_graphs:
            return {}
        
        adj_matrix = self.visibility_graphs[uid]
        G = nx.from_numpy_array(adj_matrix)
        degrees = self.degree_distributions[uid]
        
        stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': degrees.mean(),
            'std_degree': degrees.std(),
            'max_degree': degrees.max(),
            'min_degree': degrees.min(),
            'avg_clustering': nx.average_clustering(G),
        }
        
        if nx.is_connected(G):
            stats['avg_path_length'] = nx.average_shortest_path_length(G)
        else:
            stats['num_components'] = nx.number_connected_components(G)
        
        return stats
