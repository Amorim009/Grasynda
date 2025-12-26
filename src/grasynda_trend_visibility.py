
import numpy as np
import pandas as pd
from typing import Dict

from src.grasynda_trend import GrasyndaTrend
from src.grasynda_visibility import GrasyndaVisibilityGraph, VisibilityGraph


class GrasyndaTrendVisibility(GrasyndaTrend):
    """
    Grasynda variant that applies Visibility Graph generation to the differentiated trend.
    
    Pipeline:
    1. Decompose (STL)
    2. Differentiate Trend -> diff_trend
    3. Generate synthetic diff_trend using Visibility Graph (Horizontal/Natural)
    4. Integrate back -> synthetic trend
    5. Reconstruct
    """
    
    def __init__(self,
                 period: int,
                 visibility_type: str = 'horizontal',
                 robust: bool = False):
        
        # Initialize parent (GrasyndaTrend)
        super().__init__(
            n_quantiles=0,  # Not used
            period=period,
            ensemble_transitions=False,
            robust=robust
        )
        
        self.visibility_type = visibility_type
        self.alias = 'GrasyndaTrendVG'
    
    def _create_synthetic_diff_trend(self, df: pd.DataFrame) -> Dict:
        """
        Override: Generate synthetic differentiated trend using Visibility Graph.
        """
        generated_diff_trend = {}
        
        # Build visibility graphs directly for diff_trend
        visibility_graphs = {}
        degree_distributions = {}
        value_degree_map = {}
        degree_transitions = {}
        
        for uid, group in df.groupby('unique_id'):
            series = group['diff_trend'].values
            
            # Build visibility graph
            if self.visibility_type == 'horizontal':
                adj_matrix = VisibilityGraph.horizontal_visibility(series)
            else:
                adj_matrix = VisibilityGraph.natural_visibility(series)
            
            visibility_graphs[uid] = adj_matrix
            
            # Extract degree sequence
            degrees = VisibilityGraph.extract_degree_sequence(adj_matrix)
            degree_distributions[uid] = degrees
            
            # Map degrees→values
            value_degree_map[uid] = {}
            for i, (val, deg) in enumerate(zip(series, degrees)):
                deg_int = int(deg)
                if deg_int not in value_degree_map[uid]:
                    value_degree_map[uid][deg_int] = []
                value_degree_map[uid][deg_int].append(val)
            
            # Build transition matrix
            degree_transitions[uid] = self._build_transition_matrix(degrees)
        
        # Generate synthetic series
        for uid, group in df.groupby('unique_id'):
            n = len(group)
            synth_diff = self._generate_by_degree_matching(
                uid, n, degree_distributions, degree_transitions, value_degree_map
            )
            generated_diff_trend[uid] = pd.Series(synth_diff, index=group.index)
                
        return generated_diff_trend
    
    def _build_transition_matrix(self, degrees: np.ndarray) -> Dict:
        """Build transition matrix for degree sequence."""
        unique_degrees = [int(d) for d in np.unique(degrees)]
        transition_counts = {deg: {} for deg in unique_degrees}
        
        # Count transitions
        for i in range(len(degrees) - 1):
            current_deg = int(degrees[i])
            next_deg = int(degrees[i + 1])
            
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
                transition_probs[deg_from] = {
                    deg_to: 1.0 / len(unique_degrees) 
                    for deg_to in unique_degrees
                }
        
        return {
            'unique_degrees': unique_degrees,
            'transition_probs': transition_probs
        }
    
    def _generate_by_degree_matching(self, uid: str, length: int, 
                                    degree_distributions, degree_transitions, 
                                    value_degree_map) -> np.ndarray:
        """Generate synthetic series using degree transition matrix."""
        synth = np.zeros(length)
        
        # Get transition data
        transition_info = degree_transitions[uid]
        unique_degrees = transition_info['unique_degrees']
        transition_probs = transition_info['transition_probs']
        
        # Initialize
        original_degrees = degree_distributions[uid]
        current_degree = int(np.random.choice(original_degrees))
        
        sampled_degrees = [current_degree]
        
        for i in range(1, length):
            # Next degree
            if current_degree in transition_probs:
                next_degrees = list(transition_probs[current_degree].keys())
                next_probs = list(transition_probs[current_degree].values())
                current_degree = int(np.random.choice(next_degrees, p=next_probs))
            else:
                current_degree = int(np.random.choice(unique_degrees))
            
            sampled_degrees.append(current_degree)
        
        # Degrees→values mapping
        for i, target_degree in enumerate(sampled_degrees):
            if target_degree in value_degree_map[uid]:
                possible_values = value_degree_map[uid][target_degree]
                synth[i] = np.random.choice(possible_values)
            else:
                # Find nearest
                available_degrees = list(value_degree_map[uid].keys())
                nearest = min(available_degrees, key=lambda x: abs(x - target_degree))
                synth[i] = np.random.choice(value_degree_map[uid][nearest])
        
        return synth
