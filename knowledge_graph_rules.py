import networkx as nx
import numpy as np
import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from collections import defaultdict
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class FencingLogicalRules:
    """
    Implements logical rules for fencing technique analysis based on
    knowledge graph data and first-order logic rules
    """
    def __init__(self):
        """Initialize the logical rule engine"""
        # Define standard fencing techniques
        self.standard_techniques = [
            'attack', 'defense', 'parry', 'riposte', 'lunge',
            'advance', 'retreat', 'feint'
        ]
        
        # Define logical implications between techniques
        # Format: (precondition, postcondition, confidence_score)
        self.technique_implications = [
            # Sequential implications (A typically followed by B)
            ('lunge', 'retreat', 0.8),  # After a lunge, a retreat often follows
            ('parry', 'riposte', 0.9),  # A parry is typically followed by a riposte
            ('feint', 'attack', 0.7),   # A feint is typically followed by a real attack
            ('advance', 'lunge', 0.6),  # An advance is often followed by a lunge
            
            # Logical negations (A makes B unlikely)
            ('retreat', 'lunge', -0.8),  # Can't lunge while retreating
            ('lunge', 'advance', -0.7),  # Can't advance during a lunge
            ('retreat', 'attack', -0.6), # Difficult to attack while retreating
            
            # Conditional requirements (A required for B)
            ('advance', 'attack', 0.4),  # Advance often required before attack
            ('parry', 'defense', 0.5),   # Parry is a form of defense
        ]
        
        # Define positional constraints
        self.positional_rules = {
            'lunge': {
                'right_leg': ('angle', '<', 130),  # Right leg bent
                'torso': ('angle', '>', 15),       # Torso leaning forward
                'right_arm': ('angle', '>', 160)   # Right arm extended
            },
            'parry': {
                'right_arm': ('angle', '<', 140),  # Right arm bent
                'torso': ('angle', '>', -10)       # Torso relatively upright
            },
            'retreat': {
                'movement': ('direction', '<', 0)  # Moving backward
            },
            'advance': {
                'movement': ('direction', '>', 0)  # Moving forward
            }
        }
        
        # Define complex rule patterns that combine multiple observations
        self.complex_rules = [
            # Format: ([conditions], conclusion, confidence)
            
            # Rule 1: If we see arm extension followed by forward movement and bent leg, it's a lunge
            ([('right_arm', '>', 160), ('movement', '>', 5), ('right_leg', '<', 140)], 'lunge', 0.9),
            
            # Rule 2: If parry is detected and then arm extends, it's a riposte
            ([('parry', 'detected'), ('right_arm', '>', 150)], 'riposte', 0.85),
            
            # Rule 3: If retreat is detected and then arm extends, it's a counterattack
            ([('retreat', 'detected'), ('right_arm', '>', 150)], 'counterattack', 0.7),
            
            # Rule 4: If arm movement is lateral and then arm is bent, it's a parry
            ([('arm_movement', 'lateral'), ('right_arm', '<', 140)], 'parry', 0.8)
        ]
        
        # Initialize local knowledge graph
        self.knowledge_graph = nx.DiGraph()
        self._initialize_local_knowledge_graph()
        
        # Try to connect to Neo4j if available
        try:
            self.neo4j_uri = "bolt://localhost:7687"
            self.neo4j_user = "neo4j"
            self.neo4j_password = NEO4J_PASSWORD
            self.driver = GraphDatabase.driver(
                self.neo4j_uri, 
                auth=(self.neo4j_user, self.neo4j_password)
            )
            self.neo4j_available = True
            print("Successfully connected to Neo4j database")
            
            # Load knowledge from Neo4j to local graph
            self._load_knowledge_from_neo4j()
        except Exception as e:
            print(f"Neo4j connection failed: {str(e)}")
            print("Using local knowledge graph only")
            self.neo4j_available = False
    
    def _initialize_local_knowledge_graph(self):
        """Initialize the local knowledge graph with basic fencing knowledge"""
        # Add technique nodes
        for technique in self.standard_techniques:
            self.knowledge_graph.add_node(technique, type='technique')
        
        # Add relationships based on implications
        for precond, postcond, confidence in self.technique_implications:
            if confidence > 0:
                self.knowledge_graph.add_edge(
                    precond, postcond, 
                    type='leads_to', 
                    confidence=confidence
                )
        
        # Add key positions/movements
        key_positions = ['forward_stance', 'backward_stance', 'neutral_stance',
                        'extended_arm', 'bent_arm', 'forward_lunge']
        
        for position in key_positions:
            self.knowledge_graph.add_node(position, type='position')
        
        # Add position to technique relationships
        position_technique_relations = [
            ('forward_stance', 'advance', 0.8),
            ('backward_stance', 'retreat', 0.9),
            ('extended_arm', 'attack', 0.7),
            ('extended_arm', 'lunge', 0.8),
            ('bent_arm', 'parry', 0.9),
            ('forward_lunge', 'lunge', 1.0)
        ]
        
        for position, technique, confidence in position_technique_relations:
            self.knowledge_graph.add_edge(
                position, technique,
                type='indicates',
                confidence=confidence
            )
    
    def _load_knowledge_from_neo4j(self):
        """Load knowledge from Neo4j database into local graph"""
        if not self.neo4j_available:
            return
            
        try:
            with self.driver.session() as session:
                # Load techniques and their relationships
                result = session.run("""
                    MATCH (t1:Entity)-[r:RELATED_TO]->(t2:Entity)
                    WHERE EXISTS((t1)-[:MENTIONED_IN]->(:Book))
                    AND EXISTS((t2)-[:MENTIONED_IN]->(:Book))
                    RETURN t1.name as source, t2.name as target, r.context as context
                """)
                
                for record in result:
                    source = record["source"]
                    target = record["target"]
                    context = record["context"] if record["context"] else "related_to"
                    
                    # Add nodes if they don't exist
                    if not self.knowledge_graph.has_node(source):
                        self.knowledge_graph.add_node(source, type='technique')
                    if not self.knowledge_graph.has_node(target):
                        self.knowledge_graph.add_node(target, type='technique')
                    
                    # Add relationship
                    self.knowledge_graph.add_edge(
                        source, target,
                        type=context, 
                        confidence=0.7  # Default confidence
                    )
                
                print(f"Loaded {len(self.knowledge_graph.edges)} relationships from Neo4j")
                
        except Exception as e:
            print(f"Error loading data from Neo4j: {str(e)}")
    
    def save_knowledge_graph(self, output_file="fencing_knowledge_graph.json"):
        """Save the knowledge graph to a JSON file"""
        # Convert NetworkX graph to data structure suitable for JSON
        data = {
            "nodes": [],
            "edges": []
        }
        
        for node, attrs in self.knowledge_graph.nodes(data=True):
            node_data = {"id": node, "type": attrs.get("type", "unknown")}
            data["nodes"].append(node_data)
        
        for source, target, attrs in self.knowledge_graph.edges(data=True):
            edge_data = {
                "source": source,
                "target": target,
                "type": attrs.get("type", "related_to"),
                "confidence": attrs.get("confidence", 0.5)
            }
            data["edges"].append(edge_data)
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Knowledge graph saved to {output_file}")
    
    def add_rule(self, precondition, postcondition, confidence):
        """Add a new logical rule to the system"""
        # Add to implications list
        self.technique_implications.append((precondition, postcondition, confidence))
        
        # Add to graph if confidence is positive
        if confidence > 0:
            # Ensure nodes exist
            if not self.knowledge_graph.has_node(precondition):
                self.knowledge_graph.add_node(precondition, type='technique')
            if not self.knowledge_graph.has_node(postcondition):
                self.knowledge_graph.add_node(postcondition, type='technique')
                
            # Add relationship
            self.knowledge_graph.add_edge(
                precondition, postcondition,
                type='leads_to',
                confidence=confidence
            )
    
    def add_complex_rule(self, conditions, conclusion, confidence):
        """Add a new complex rule that depends on multiple conditions"""
        self.complex_rules.append((conditions, conclusion, confidence))
    
    def evaluate_pose_with_rules(self, pose_data):
        """
        Evaluate pose data using logical rules
        
        Args:
            pose_data: Dict containing pose angles and movements
            
        Returns:
            technique_scores: Dict of technique scores based on rules
        """
        technique_scores = {tech: 0.0 for tech in self.standard_techniques}
        triggered_rules = []
        
        # Apply positional rules
        for technique, rules in self.positional_rules.items():
            score = 0.0
            rule_count = 0
            
            for body_part, (measurement_type, operator, threshold) in rules.items():
                # Check if this body part and measurement are available
                if body_part in pose_data:
                    value = pose_data[body_part]
                    
                    # Evaluate the rule condition
                    if operator == '>' and value > threshold:
                        score += 1.0
                        triggered_rules.append(f"{body_part} {operator} {threshold} → {technique}")
                    elif operator == '<' and value < threshold:
                        score += 1.0
                        triggered_rules.append(f"{body_part} {operator} {threshold} → {technique}")
                    elif operator == '==' and value == threshold:
                        score += 1.0
                        triggered_rules.append(f"{body_part} {operator} {threshold} → {technique}")
                    
                    rule_count += 1
            
            # Calculate average score if any rules were evaluated
            if rule_count > 0:
                technique_scores[technique] = score / rule_count
        
        # Apply complex rules
        observed_facts = []
        for body_part, value in pose_data.items():
            observed_facts.append((body_part, value))
        
        for conditions, conclusion, confidence in self.complex_rules:
            # Check if all conditions are met
            conditions_met = True
            for condition in conditions:
                if len(condition) == 3:
                    attr, op, val = condition
                    if attr in pose_data:
                        if op == '>' and not (pose_data[attr] > val):
                            conditions_met = False
                            break
                        elif op == '<' and not (pose_data[attr] < val):
                            conditions_met = False
                            break
                        elif op == '==' and not (pose_data[attr] == val):
                            conditions_met = False
                            break
                elif len(condition) == 2:
                    tech, state = condition
                    # Check if technique was previously detected
                    if tech in technique_scores:
                        if state == 'detected' and technique_scores[tech] < 0.5:
                            conditions_met = False
                            break
            
            # If all conditions met, add to score
            if conditions_met and conclusion in technique_scores:
                technique_scores[conclusion] += confidence
                triggered_rules.append(f"Complex rule → {conclusion} ({confidence})")
        
        # Normalize scores to be between 0 and 1
        for technique in technique_scores:
            technique_scores[technique] = min(1.0, technique_scores[technique])
        
        return technique_scores, triggered_rules
    
    def apply_logical_constraints(self, initial_scores):
        """
        Apply logical constraints to refine technique scores
        
        Args:
            initial_scores: Dict of initial technique scores
            
        Returns:
            refined_scores: Dict of scores after logical constraints
        """
        refined_scores = initial_scores.copy()
        
        # Apply implication rules
        for precond, postcond, confidence in self.technique_implications:
            if precond in initial_scores and postcond in initial_scores:
                # Positive implication: if precond likely, postcond more likely
                if confidence > 0 and initial_scores[precond] > 0.5:
                    boost = initial_scores[precond] * confidence
                    refined_scores[postcond] = min(1.0, refined_scores[postcond] + boost * 0.3)
                
                # Negative implication: if precond likely, postcond less likely
                elif confidence < 0 and initial_scores[precond] > 0.5:
                    penalty = initial_scores[precond] * abs(confidence)
                    refined_scores[postcond] = max(0.0, refined_scores[postcond] - penalty * 0.3)
        
        # Apply mutual exclusivity rules (can't do certain techniques simultaneously)
        mutual_exclusions = [
            ('lunge', 'retreat'),
            ('lunge', 'parry'),
            ('advance', 'retreat')
        ]
        
        for tech1, tech2 in mutual_exclusions:
            if tech1 in initial_scores and tech2 in initial_scores:
                # If both have high scores, reduce the lower one
                if initial_scores[tech1] > 0.4 and initial_scores[tech2] > 0.4:
                    if initial_scores[tech1] > initial_scores[tech2]:
                        refined_scores[tech2] *= 0.5
                    else:
                        refined_scores[tech1] *= 0.5
        
        return refined_scores
    
    def infer_from_sequence(self, techniques_sequence, current_scores):
        """
        Infer likely next techniques from a sequence of recently observed techniques
        
        Args:
            techniques_sequence: List of recently detected techniques
            current_scores: Dict of current technique scores
            
        Returns:
            updated_scores: Dict of updated scores after sequence inference
        """
        if not techniques_sequence:
            return current_scores
        
        updated_scores = current_scores.copy()
        
        # Get the most recent technique
        most_recent = techniques_sequence[-1]
        
        # Find outgoing edges in knowledge graph
        if self.knowledge_graph.has_node(most_recent):
            for _, next_tech, data in self.knowledge_graph.out_edges(most_recent, data=True):
                if next_tech in updated_scores:
                    confidence = data.get('confidence', 0.5)
                    updated_scores[next_tech] += 0.2 * confidence
                    
                    # Cap at 1.0
                    updated_scores[next_tech] = min(1.0, updated_scores[next_tech])
        
        # Check for common patterns in fencing
        common_patterns = {
            ('retreat', 'retreat'): {'advance': 0.7},  # Double retreat often followed by advance
            ('parry', 'riposte'): {'retreat': 0.6},    # After parry-riposte, often retreat
            ('advance', 'lunge'): {'retreat': 0.8},    # After advance-lunge, usually retreat
            ('feint', 'attack'): {'retreat': 0.5}      # After feint-attack, maybe retreat
        }
        
        # Check if last two techniques match a pattern
        if len(techniques_sequence) >= 2:
            last_two = tuple(techniques_sequence[-2:])
            if last_two in common_patterns:
                for tech, boost in common_patterns[last_two].items():
                    if tech in updated_scores:
                        updated_scores[tech] += 0.3 * boost
                        updated_scores[tech] = min(1.0, updated_scores[tech])
        
        return updated_scores
    
    def query_knowledge_graph(self, query_type, query_params):
        """
        Query the knowledge graph for specific information
        
        Args:
            query_type: Type of query (e.g., 'related_techniques', 'prerequisites')
            query_params: Parameters for the query
            
        Returns:
            results: Query results
        """
        results = []
        
        if query_type == 'related_techniques':
            technique = query_params.get('technique')
            if technique and self.knowledge_graph.has_node(technique):
                # Find directly related techniques
                for _, related, data in self.knowledge_graph.out_edges(technique, data=True):
                    results.append({
                        'technique': related,
                        'relationship': data.get('type', 'related'),
                        'confidence': data.get('confidence', 0.5)
                    })
        
        elif query_type == 'prerequisites':
            technique = query_params.get('technique')
            if technique and self.knowledge_graph.has_node(technique):
                # Find prerequisites (incoming edges)
                for prereq, _, data in self.knowledge_graph.in_edges(technique, data=True):
                    if data.get('type') == 'leads_to':
                        results.append({
                            'prerequisite': prereq,
                            'confidence': data.get('confidence', 0.5)
                        })
        
        elif query_type == 'neo4j_custom':
            # Direct Neo4j query if available
            if self.neo4j_available:
                cypher_query = query_params.get('cypher')
                if cypher_query:
                    try:
                        with self.driver.session() as session:
                            result = session.run(cypher_query)
                            results = [dict(record) for record in result]
                    except Exception as e:
                        print(f"Error executing Neo4j query: {str(e)}")
        
        return results
    
    def visualize_knowledge_graph(self, output_file="knowledge_graph.png"):
        """Visualize the knowledge graph"""
        plt.figure(figsize=(12, 10))
        
        # Create position layout
        pos = nx.spring_layout(self.knowledge_graph, seed=42)
        
        # Draw nodes with different colors for different types
        node_colors = []
        for node in self.knowledge_graph.nodes():
            node_type = self.knowledge_graph.nodes[node].get('type', 'unknown')
            if node_type == 'technique':
                node_colors.append('skyblue')
            elif node_type == 'position':
                node_colors.append('lightgreen')
            else:
                node_colors.append('lightgray')
        
        nx.draw_networkx_nodes(self.knowledge_graph, pos, 
                              node_color=node_colors, 
                              node_size=700, 
                              alpha=0.8)
        
        # Draw edges with different styles for different relationships
        for source, target, data in self.knowledge_graph.edges(data=True):
            edge_type = data.get('type', 'related_to')
            confidence = data.get('confidence', 0.5)
            
            # Edge style based on type
            if edge_type == 'leads_to':
                style = 'solid'
                color = 'blue'
            elif edge_type == 'indicates':
                style = 'dashed'
                color = 'green'
            else:
                style = 'dotted'
                color = 'gray'
            
            # Edge width based on confidence
            width = 1 + 2 * confidence
            
            # Draw single edge
            nx.draw_networkx_edges(
                self.knowledge_graph, pos,
                edgelist=[(source, target)],
                width=width,
                edge_color=color,
                style=style,
                alpha=0.7
            )
        
        # Draw node labels
        nx.draw_networkx_labels(self.knowledge_graph, pos, font_size=10)
        
        # Add a title
        plt.title("Fencing Knowledge Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        
        # Save to file
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Knowledge graph visualization saved to {output_file}")
        
        return output_file
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'driver') and self.neo4j_available:
            self.driver.close()

class FencingReasoningEngine:
    """
    Advanced reasoning engine for fencing analysis that combines
    knowledge graph rules with pose estimation and VideoMAE predictions
    """
    def __init__(self):
        """Initialize the reasoning engine"""
        self.rules_engine = FencingLogicalRules()
        self.confidence_thresholds = {
            'high': 0.7,
            'medium': 0.4,
            'low': 0.2
        }
        
        # Weight factors for combining different sources
        self.weight_factors = {
            'videomae': 0.3,
            'pose': 0.4,
            'rules': 0.3
        }
        
        # Tracking sequences for temporal reasoning
        self.recent_techniques = []
        self.max_sequence_length = 5
    
    def analyze_frame(self, pose_data, videomae_scores=None):
        """
        Analyze a single frame using pose data and optional VideoMAE scores
        
        Args:
            pose_data: Dict of pose information (angles, movements)
            videomae_scores: Optional dict of VideoMAE technique scores
            
        Returns:
            technique_scores: Dict of refined technique scores
            explanations: List of rule explanations
        """
        # Convert VideoMAE scores to dict if provided as array
        if videomae_scores is not None and not isinstance(videomae_scores, dict):
            standard_techniques = self.rules_engine.standard_techniques
            videomae_dict = {
                standard_techniques[i]: score 
                for i, score in enumerate(videomae_scores)
                if i < len(standard_techniques)
            }
            videomae_scores = videomae_dict
        
        # Apply rules to pose data
        rule_scores, triggered_rules = self.rules_engine.evaluate_pose_with_rules(pose_data)
        
        # Initialize combined scores with rule scores
        combined_scores = rule_scores.copy()
        
        # Add VideoMAE influence if available
        if videomae_scores:
            for technique in combined_scores:
                if technique in videomae_scores:
                    # Weighted combination
                    combined_scores[technique] = (
                        self.weight_factors['rules'] * rule_scores[technique] +
                        self.weight_factors['videomae'] * videomae_scores[technique]
                    )
        
        # Apply logical constraints
        refined_scores = self.rules_engine.apply_logical_constraints(combined_scores)
        
        # Update temporal sequence and apply sequence reasoning
        if refined_scores:
            # Find top technique from refined scores
            top_technique = max(refined_scores.items(), key=lambda x: x[1])
            
            # Only add to sequence if confidence is medium or higher
            if top_technique[1] >= self.confidence_thresholds['medium']:
                self.recent_techniques.append(top_technique[0])
                # Keep sequence to maximum length
                if len(self.recent_techniques) > self.max_sequence_length:
                    self.recent_techniques.pop(0)
            
            # Apply sequence inference
            if self.recent_techniques:
                refined_scores = self.rules_engine.infer_from_sequence(
                    self.recent_techniques, refined_scores
                )
        
        # Generate explanations
        explanations = []
        for rule in triggered_rules:
            explanations.append(f"Rule triggered: {rule}")
        
        if self.recent_techniques:
            explanations.append(f"Recent sequence: {' → '.join(self.recent_techniques)}")
        
        return refined_scores, explanations
    
    def get_sequence_predictions(self, num_steps=3):
        """
        Predict future techniques based on recent sequence
        
        Args:
            num_steps: Number of future steps to predict
            
        Returns:
            predictions: List of predicted techniques
        """
        if not self.recent_techniques:
            return []
        
        # Start with most recent technique
        current = self.recent_techniques[-1]
        predictions = [current]
        
        # Predict next steps
        for _ in range(num_steps):
            # Query knowledge graph for likely next techniques
            next_techniques = self.rules_engine.query_knowledge_graph(
                'related_techniques',
                {'technique': current}
            )
            
            if next_techniques:
                # Sort by confidence
                next_techniques.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Get top technique
                next_tech = next_techniques[0]['technique']
                predictions.append(next_tech)
                current = next_tech
            else:
                break
        
        # Remove the first element (current technique)
        return predictions[1:]
    
    def query_knowledge(self, query_type, params):
        """Pass-through method to query the knowledge graph"""
        return self.rules_engine.query_knowledge_graph(query_type, params)
    
    def visualize_knowledge(self, output_file="fencing_knowledge.png"):
        """Visualize the knowledge graph"""
        return self.rules_engine.visualize_knowledge_graph(output_file)
    
    def close(self):
        """Close connections"""
        self.rules_engine.close()

# Example usage
if __name__ == "__main__":
    # Initialize reasoning engine
    reasoning_engine = FencingReasoningEngine()
    
    # Visualize knowledge graph
    reasoning_engine.visualize_knowledge()
    
    # Sample pose data
    pose_data = {
        'right_arm': 165,  # Extended arm
        'right_leg': 110,  # Bent leg
        'torso': 20,       # Forward lean
        'movement': 10     # Forward movement
    }
    
    # Sample VideoMAE scores
    videomae_scores = {
        'attack': 0.3,
        'defense': 0.1,
        'parry': 0.05,
        'riposte': 0.05,
        'lunge': 0.4,
        'advance': 0.05,
        'retreat': 0.02,
        'feint': 0.03
    }
    
    # Analyze frame
    refined_scores, explanations = reasoning_engine.analyze_frame(pose_data, videomae_scores)
    
    print("Analysis Results:")
    for technique, score in sorted(refined_scores.items(), key=lambda x: x[1], reverse=True):
        if score > 0.1:
            print(f"  {technique}: {score:.3f}")
    
    print("\nExplanations:")
    for explanation in explanations:
        print(f"  {explanation}")
    
    # Predict future techniques
    predictions = reasoning_engine.get_sequence_predictions(3)
    if predictions:
        print("\nPredicted next techniques:")
        for i, technique in enumerate(predictions, 1):
            print(f"  {i}. {technique}")
    
    # Clean up
    reasoning_engine.close() 