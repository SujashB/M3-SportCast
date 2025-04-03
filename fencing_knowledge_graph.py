import json
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend BEFORE pyplot import
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Tuple, Optional, Any, Set
import os

class FencingKnowledgeGraph:
    """
    Knowledge graph for modeling fencing tactical knowledge and transitions
    """
    def __init__(self):
        """Initialize the knowledge graph"""
        # Create directed graph
        self.graph = nx.DiGraph()
        
        # Define core techniques
        self.core_techniques = [
            "neutral", "attack", "defense", "lunge", "parry", "riposte", 
            "advance", "retreat", "feint", "flick", "counter-attack", "bind"
        ]
        
        # Define technique attributes
        self.technique_attributes = {
            "neutral": {
                "type": "position",
                "distance": "medium",
                "blade_position": "middle",
                "risk_level": "low",
                "complexity": "low"
            },
            "attack": {
                "type": "offensive",
                "distance": "closing",
                "blade_position": "extending",
                "risk_level": "medium",
                "complexity": "medium"
            },
            "defense": {
                "type": "defensive",
                "distance": "maintaining",
                "blade_position": "protecting",
                "risk_level": "low",
                "complexity": "medium"
            },
            "lunge": {
                "type": "offensive",
                "distance": "long",
                "blade_position": "extended",
                "risk_level": "high",
                "complexity": "medium"
            },
            "parry": {
                "type": "defensive",
                "distance": "short",
                "blade_position": "blocking",
                "risk_level": "medium",
                "complexity": "medium"
            },
            "riposte": {
                "type": "offensive",
                "distance": "medium",
                "blade_position": "extending",
                "risk_level": "medium",
                "complexity": "high"
            },
            "advance": {
                "type": "movement",
                "distance": "closing",
                "blade_position": "variable",
                "risk_level": "low",
                "complexity": "low"
            },
            "retreat": {
                "type": "movement",
                "distance": "increasing",
                "blade_position": "variable",
                "risk_level": "low",
                "complexity": "low"
            },
            "feint": {
                "type": "offensive",
                "distance": "variable",
                "blade_position": "deceptive",
                "risk_level": "medium",
                "complexity": "high"
            },
            "flick": {
                "type": "offensive",
                "distance": "medium",
                "blade_position": "bent",
                "risk_level": "high",
                "complexity": "high"
            },
            "counter-attack": {
                "type": "offensive",
                "distance": "medium",
                "blade_position": "extending",
                "risk_level": "high",
                "complexity": "high"
            },
            "bind": {
                "type": "control",
                "distance": "close",
                "blade_position": "contact",
                "risk_level": "medium",
                "complexity": "high"
            }
        }
        
        # Define common transitions between techniques
        self.common_transitions = [
            ("neutral", "attack", 0.8),
            ("neutral", "advance", 0.9),
            ("neutral", "retreat", 0.9),
            ("neutral", "feint", 0.6),
            
            ("attack", "lunge", 0.85),
            ("attack", "feint", 0.7),
            ("attack", "flick", 0.5),
            
            ("defense", "parry", 0.9),
            ("defense", "retreat", 0.8),
            ("defense", "counter-attack", 0.6),
            
            ("lunge", "neutral", 0.7),
            ("lunge", "retreat", 0.5),
            
            ("parry", "riposte", 0.85),
            ("parry", "retreat", 0.6),
            
            ("riposte", "lunge", 0.7),
            ("riposte", "advance", 0.5),
            
            ("advance", "attack", 0.8),
            ("advance", "lunge", 0.7),
            ("advance", "feint", 0.6),
            
            ("retreat", "defense", 0.8),
            ("retreat", "parry", 0.7),
            ("retreat", "counter-attack", 0.5),
            
            ("feint", "attack", 0.8),
            ("feint", "lunge", 0.7),
            
            ("flick", "neutral", 0.6),
            ("flick", "retreat", 0.5),
            
            ("counter-attack", "retreat", 0.7),
            ("counter-attack", "defense", 0.6),
            
            ("bind", "attack", 0.7),
            ("bind", "feint", 0.6),
            ("bind", "retreat", 0.5)
        ]
        
        # Initialize the graph with core techniques
        self._build_graph()
    
    def _build_graph(self):
        """Build the knowledge graph with core techniques and transitions"""
        # Add nodes (techniques) with attributes
        for technique in self.core_techniques:
            self.graph.add_node(technique, **self.technique_attributes.get(technique, {}))
        
        # Add edges (transitions) with probabilities
        for source, target, probability in self.common_transitions:
            self.graph.add_edge(source, target, probability=probability)
    
    def add_technique(self, technique: str, attributes: Dict[str, Any]) -> None:
        """
        Add a new technique to the knowledge graph
        
        Args:
            technique: Name of the technique
            attributes: Dictionary of technique attributes
        """
        if technique not in self.graph:
            self.graph.add_node(technique, **attributes)
            self.core_techniques.append(technique)
            self.technique_attributes[technique] = attributes
    
    def add_transition(self, source: str, target: str, probability: float) -> None:
        """
        Add a transition between techniques to the knowledge graph
        
        Args:
            source: Source technique
            target: Target technique
            probability: Transition probability
        """
        if source in self.graph and target in self.graph:
            self.graph.add_edge(source, target, probability=probability)
            self.common_transitions.append((source, target, probability))
    
    def get_next_likely_techniques(self, current_technique: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the most likely next techniques given the current technique
        
        Args:
            current_technique: Current technique
            top_k: Number of top techniques to return
            
        Returns:
            List of (technique, probability) tuples sorted by probability
        """
        if current_technique not in self.graph:
            return []
        
        # Get all outgoing edges
        edges = self.graph.out_edges(current_technique, data=True)
        
        # Sort by probability
        sorted_edges = sorted(edges, key=lambda x: x[2]['probability'], reverse=True)
        
        # Return top-k techniques with probabilities
        return [(edge[1], edge[2]['probability']) for edge in sorted_edges[:top_k]]
    
    def get_technique_sequence_probability(self, technique_sequence: List[str]) -> float:
        """
        Calculate the probability of a sequence of techniques
        
        Args:
            technique_sequence: List of techniques in sequence
            
        Returns:
            Probability of the sequence
        """
        if len(technique_sequence) < 2:
            return 1.0
        
        total_probability = 1.0
        
        for i in range(len(technique_sequence) - 1):
            source = technique_sequence[i]
            target = technique_sequence[i + 1]
            
            if source in self.graph and target in self.graph and self.graph.has_edge(source, target):
                transition_prob = self.graph[source][target]['probability']
                total_probability *= transition_prob
            else:
                # If transition doesn't exist, assign a very low probability
                total_probability *= 0.01
        
        return total_probability
    
    def find_common_sequence(self, length: int = 3) -> List[List[str]]:
        """
        Find common sequences of techniques of specified length
        
        Args:
            length: Length of sequences to find
            
        Returns:
            List of technique sequences sorted by probability
        """
        sequences = []
        
        # Generate all paths of specified length
        for source in self.graph.nodes():
            self._generate_paths(source, [], length, sequences)
        
        # Sort sequences by probability
        sorted_sequences = sorted(
            [(seq, self.get_technique_sequence_probability(seq)) for seq in sequences],
            key=lambda x: x[1],
            reverse=True
        )
        
        return [seq for seq, _ in sorted_sequences]
    
    def _generate_paths(self, current: str, path: List[str], length: int, result: List[List[str]]) -> None:
        """
        Helper function to generate all paths of specified length
        
        Args:
            current: Current technique
            path: Current path
            length: Maximum path length
            result: List to store results
        """
        # Add current technique to path
        current_path = path + [current]
        
        # If we reached desired length, add to result
        if len(current_path) == length:
            result.append(current_path)
            return
        
        # Continue with next techniques
        for neighbor in self.graph.neighbors(current):
            self._generate_paths(neighbor, current_path, length, result)
    
    def predict_next_technique(self, technique_history: List[str]) -> str:
        """
        Predict the next technique based on history
        
        Args:
            technique_history: List of techniques observed so far
            
        Returns:
            Predicted next technique
        """
        if not technique_history:
            # If no history, return most common starting technique
            start_techniques = []
            for source, target, _ in self.common_transitions:
                start_techniques.append(source)
            
            # Count occurrences
            counts = {}
            for tech in start_techniques:
                counts[tech] = counts.get(tech, 0) + 1
            
            # Return most common
            return max(counts.items(), key=lambda x: x[1])[0]
        
        # Use the last technique to predict the next
        current = technique_history[-1]
        next_techniques = self.get_next_likely_techniques(current, top_k=1)
        
        if next_techniques:
            return next_techniques[0][0]
        else:
            return "neutral"  # Default fallback
    
    def visualize(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the knowledge graph
        
        Args:
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 10))
        
        # Define node colors based on technique type
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'unknown')
            if node_type == 'offensive':
                node_colors.append('red')
            elif node_type == 'defensive':
                node_colors.append('blue')
            elif node_type == 'movement':
                node_colors.append('green')
            elif node_type == 'position':
                node_colors.append('yellow')
            elif node_type == 'control':
                node_colors.append('purple')
            else:
                node_colors.append('gray')
        
        # Get edge weights (probabilities)
        edge_weights = [self.graph[u][v]['probability'] * 5 for u, v in self.graph.edges()]
        
        # Create layout
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, alpha=0.8, node_size=800)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, width=edge_weights, alpha=0.6, arrows=True, arrowsize=15, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_family='sans-serif')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Offensive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Defensive'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Movement'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=10, label='Position'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='Control')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title('Fencing Techniques Knowledge Graph')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save(self, filepath: str) -> None:
        """
        Save the knowledge graph to a JSON file
        
        Args:
            filepath: Path to save the file
        """
        data = {
            'techniques': {node: dict(self.graph.nodes[node]) for node in self.graph.nodes()},
            'transitions': [(u, v, self.graph[u][v]['probability']) for u, v in self.graph.edges()]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'FencingKnowledgeGraph':
        """
        Load a knowledge graph from a JSON file
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Loaded knowledge graph
        """
        kg = cls()
        kg.graph = nx.DiGraph()
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Add nodes
        for technique, attributes in data['techniques'].items():
            kg.graph.add_node(technique, **attributes)
        
        # Add edges
        for u, v, probability in data['transitions']:
            kg.graph.add_edge(u, v, probability=probability)
        
        # Update core techniques and attributes
        kg.core_techniques = list(kg.graph.nodes())
        kg.technique_attributes = {node: dict(kg.graph.nodes[node]) for node in kg.graph.nodes()}
        kg.common_transitions = [(u, v, kg.graph[u][v]['probability']) for u, v in kg.graph.edges()]
        
        return kg


class FencingTacticalAnalyzer:
    """
    Tactical analyzer for fencing that provides coaching feedback based on observed techniques
    using the knowledge graph
    """
    def __init__(self, knowledge_graph):
        """
        Initialize the tactical analyzer with a knowledge graph
        
        Args:
            knowledge_graph: FencingKnowledgeGraph instance
        """
        self.knowledge_graph = knowledge_graph
        self.coaching_templates = {
            # Templates for single techniques
            'technique': [
                "Focus on your {technique} technique, maintain proper form.",
                "Your {technique} could be improved by maintaining better distance control.",
                "When executing {technique}, remember to keep your balance.",
                "Good {technique}, but be mindful of your opponent's potential responses.",
                "Consider varying your timing on the {technique} to be less predictable."
            ],
            
            # Templates for sequential patterns
            'sequence': [
                "The {sequence} sequence you're using could be countered by {counter}.",
                "After {prev_technique}, consider {next_option} instead of {current} for better surprise.",
                "Your {sequence} pattern is becoming predictable. Try mixing in {alternative}.",
                "Good combination with {sequence}, but watch for {weakness}.",
                "When you perform {sequence}, focus on maintaining proper tempo through the entire action."
            ],
            
            # Templates for tactical situations
            'tactical': [
                "You're repeatedly using {common_technique}. Mix in {alternative} to be less predictable.",
                "Your opponent favors {opponent_pattern}. Try {counter_suggestion} to exploit this.",
                "After your {setup}, you have a good opportunity for {follow_up}.",
                "Consider using more {tactical_concept} in your approach.",
                "Be careful when attempting {risky_technique} against this opponent's style."
            ]
        }
    
    def analyze_sequence(self, technique_sequence):
        """
        Analyze a sequence of techniques using the knowledge graph
        
        Args:
            technique_sequence: List of technique names
            
        Returns:
            analysis: Dictionary with analysis results
        """
        if not technique_sequence or len(technique_sequence) < 1:
            return {'error': 'Empty technique sequence'}
        
        # Initialize analysis
        analysis = {
            'sequence': technique_sequence,
            'sequence_probability': 0.0,
            'common_patterns': [],
            'next_likely_techniques': [],
            'tactical_suggestions': []
        }
        
        # Get sequence probability
        if len(technique_sequence) > 1:
            try:
                analysis['sequence_probability'] = self.knowledge_graph.get_technique_sequence_probability(technique_sequence)
            except:
                analysis['sequence_probability'] = 0.0
        
        # Get next likely techniques
        current_technique = technique_sequence[-1]
        try:
            analysis['next_likely_techniques'] = self.knowledge_graph.get_next_likely_techniques(current_technique, top_n=3)
        except:
            analysis['next_likely_techniques'] = []
        
        # Find common patterns that include parts of this sequence
        try:
            subsequence = technique_sequence[-3:] if len(technique_sequence) >= 3 else technique_sequence
            analysis['common_patterns'] = self.knowledge_graph.find_common_sequence(subsequence, min_length=2, max_length=4)
        except:
            analysis['common_patterns'] = []
        
        # Generate tactical suggestions
        analysis['tactical_suggestions'] = self._generate_tactical_suggestions(technique_sequence, analysis)
        
        return analysis
    
    def _generate_tactical_suggestions(self, technique_sequence, analysis):
        """Generate tactical suggestions based on the sequence analysis"""
        suggestions = []
        
        # Get technique attributes for context
        current_technique = technique_sequence[-1]
        current_attrs = {}
        
        try:
            for node_id, attrs in self.knowledge_graph.graph.nodes(data=True):
                if node_id == current_technique:
                    current_attrs = attrs
                    break
        except:
            pass
        
        # Suggestion 1: Based on technique properties
        if current_attrs:
            technique_type = current_attrs.get('type', 'neutral')
            risk_level = current_attrs.get('risk_level', 'medium')
            
            if risk_level == 'high':
                suggestions.append(f"Be cautious with this high-risk {technique_type}. Consider a safer option if your score advantage is significant.")
            elif technique_type == 'attack' and len(technique_sequence) > 2:
                suggestions.append(f"You're using multiple attacks in sequence. Mix in some defensive actions to be less predictable.")
            elif technique_type == 'defense' and len(technique_sequence) > 2:
                suggestions.append(f"You're relying heavily on defensive actions. Look for counterattack opportunities.")
        
        # Suggestion 2: Based on common patterns
        if analysis['common_patterns']:
            common_pattern = analysis['common_patterns'][0]
            alternatives = self._find_alternative_techniques(current_technique)
            if alternatives:
                suggestions.append(f"You've used the pattern {' → '.join(common_pattern)} frequently. Consider {alternatives[0]} as an alternative.")
        
        # Suggestion 3: Based on next likely techniques
        if analysis['next_likely_techniques']:
            next_likely = analysis['next_likely_techniques'][0][0]
            suggestions.append(f"Your opponent might anticipate {next_likely} next with {analysis['next_likely_techniques'][0][1]:.2f} probability. Consider an alternative.")
        
        # Add a general suggestion if we don't have enough
        if len(suggestions) < 2:
            suggestions.append(f"Focus on maintaining proper distance and timing after your {current_technique}.")
        
        return suggestions
    
    def _find_alternative_techniques(self, technique):
        """Find alternative techniques with similar function but different attributes"""
        alternatives = []
        technique_attrs = {}
        
        # Get attributes of the current technique
        try:
            for node_id, attrs in self.knowledge_graph.graph.nodes(data=True):
                if node_id == technique:
                    technique_attrs = attrs
                    break
                    
            # Find techniques with same type but different attributes
            if technique_attrs:
                technique_type = technique_attrs.get('type', 'neutral')
                for node_id, attrs in self.knowledge_graph.graph.nodes(data=True):
                    if node_id != technique and attrs.get('type') == technique_type:
                        if attrs.get('distance') != technique_attrs.get('distance') or \
                           attrs.get('blade_position') != technique_attrs.get('blade_position'):
                            alternatives.append(node_id)
        except:
            pass
            
        return alternatives[:3]  # Return up to 3 alternatives
    
    def generate_coaching_feedback(self, technique_sequence):
        """
        Generate coaching feedback based on observed technique sequence
        
        Args:
            technique_sequence: List of technique names
            
        Returns:
            feedback: Coaching feedback string
        """
        if not technique_sequence:
            return "Maintain proper form and watch your distance."
        
        # Analyze the sequence
        analysis = self.analyze_sequence(technique_sequence)
        
        # Choose a feedback template based on sequence length
        if len(technique_sequence) == 1:
            template_key = 'technique'
            current_technique = technique_sequence[0]
            
            template = self._select_random_template(template_key)
            feedback = template.format(technique=current_technique)
            
        elif len(technique_sequence) >= 2:
            # For sequences, use more complex feedback
            template_key = 'sequence' if len(technique_sequence) <= 3 else 'tactical'
            
            # Get sequence representation
            sequence_str = " → ".join(technique_sequence[-3:]) if len(technique_sequence) > 3 else " → ".join(technique_sequence)
            
            # Get additional context
            context = {
                'sequence': sequence_str,
                'current': technique_sequence[-1],
                'prev_technique': technique_sequence[-2] if len(technique_sequence) > 1 else "",
                'common_technique': technique_sequence[-1],
            }
            
            # Add alternative and next options if available
            if analysis['next_likely_techniques']:
                context['next_option'] = analysis['next_likely_techniques'][0][0]
                context['counter'] = analysis['next_likely_techniques'][-1][0] if len(analysis['next_likely_techniques']) > 1 else "a faster retreat"
            else:
                context['next_option'] = "a different action"
                context['counter'] = "a timely parry"
            
            # Add alternatives
            alternatives = self._find_alternative_techniques(technique_sequence[-1])
            context['alternative'] = alternatives[0] if alternatives else "varying your timing"
            
            # Add tactical concepts based on sequence
            attack_count = sum(1 for t in technique_sequence if self._get_technique_type(t) == 'attack')
            defense_count = sum(1 for t in technique_sequence if self._get_technique_type(t) == 'defense')
            
            if attack_count > defense_count * 2:
                context['tactical_concept'] = "defensive actions"
                context['weakness'] = "vulnerability to counterattacks"
            else:
                context['tactical_concept'] = "offensive pressure"
                context['weakness'] = "missed attack opportunities"
            
            # Get template and format
            template = self._select_random_template(template_key)
            try:
                feedback = template.format(**context)
            except KeyError:
                # Fallback if template has missing keys
                feedback = f"Work on varying your {sequence_str} sequence with different timing and distance."
            
            # Add a tactical suggestion if available
            if analysis['tactical_suggestions']:
                feedback += " " + analysis['tactical_suggestions'][0]
        else:
            feedback = "Maintain proper form and watch your distance."
        
        return feedback
    
    def _select_random_template(self, template_key):
        """Select a random template from the given category"""
        import random
        templates = self.coaching_templates.get(template_key, self.coaching_templates['technique'])
        return random.choice(templates)
    
    def _get_technique_type(self, technique):
        """Get the type of a technique from the knowledge graph"""
        try:
            for node_id, attrs in self.knowledge_graph.graph.nodes(data=True):
                if node_id == technique:
                    return attrs.get('type', 'neutral')
        except:
            pass
        return 'neutral'


def generate_coaching_tips(knowledge_graph, technique_sequence):
    """
    Helper function to generate coaching tips based on a sequence of techniques
    
    Args:
        knowledge_graph: FencingKnowledgeGraph instance
        technique_sequence: List of technique names
        
    Returns:
        tips: List of coaching tips
    """
    analyzer = FencingTacticalAnalyzer(knowledge_graph)
    analysis = analyzer.analyze_sequence(technique_sequence)
    
    tips = [
        analyzer.generate_coaching_feedback(technique_sequence)
    ]
    
    # Add additional tips based on analysis
    if analysis['tactical_suggestions']:
        tips.extend(analysis['tactical_suggestions'])
    
    # Add next technique suggestion
    if analysis['next_likely_techniques']:
        next_technique, prob = analysis['next_likely_techniques'][0]
        tips.append(f"Your opponent may expect {next_technique} next with {prob:.2f} probability. Consider an alternative.")
    
    return tips


def create_default_knowledge_graph(save_path: Optional[str] = None) -> FencingKnowledgeGraph:
    """
    Create and save a default fencing knowledge graph
    
    Args:
        save_path: Optional path to save the knowledge graph
        
    Returns:
        Created knowledge graph
    """
    kg = FencingKnowledgeGraph()
    
    if save_path:
        kg.save(save_path)
        print(f"Default knowledge graph saved to {save_path}")
    
    return kg


if __name__ == "__main__":
    # Create and visualize a default knowledge graph
    kg = create_default_knowledge_graph("fencing_knowledge_graph.json")
    kg.visualize("fencing_knowledge_graph.png")
    
    # Example usage
    analyzer = FencingTacticalAnalyzer(kg)
    
    # Analyze a sample sequence
    sequence = ["neutral", "advance", "attack", "lunge", "retreat"]
    analysis = analyzer.analyze_sequence(sequence)
    
    print("Sequence:", sequence)
    print("Tactical Style:", analysis['tactical_style'])
    print("Recommendations:", analysis['recommendations'])
    
    # Generate coaching feedback
    feedback = analyzer.generate_coaching_feedback(sequence)
    print("\nCoaching Feedback:", feedback) 