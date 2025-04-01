import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import Counter, defaultdict

class FencingVisualizer:
    """
    Visualization utilities for fencing analysis
    """
    def __init__(self, output_dir='visualizations'):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set aesthetics
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("Set2")
        
        # Default colors for techniques
        self.technique_colors = {
            'attack': '#FF5733',  # orange-red
            'defense': '#33A8FF',  # blue
            'parry': '#33FF57',   # green
            'riposte': '#A833FF',  # purple
            'lunge': '#FF33A8',   # pink
            'advance': '#FFFF33',  # yellow
            'retreat': '#33FFF5',  # teal
            'feint': '#FF8333'    # light orange
        }
    
    def plot_technique_distribution(self, technique_counts, title="Fencing Techniques Distribution", 
                                   filename="technique_distribution.png"):
        """
        Create a pie chart of technique distributions
        
        Args:
            technique_counts: Dict of technique names and their counts
            title: Title of the chart
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Filter out techniques with zero counts
        filtered_counts = {k: v for k, v in technique_counts.items() if v > 0}
        
        # Get colors for the techniques
        colors = [self.technique_colors.get(tech, '#CCCCCC') for tech in filtered_counts.keys()]
        
        wedges, texts, autotexts = ax.pie(
            filtered_counts.values(), 
            labels=filtered_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            shadow=False
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')
        plt.title(title, fontsize=16, pad=20)
        
        # Make the percentage labels more readable
        for autotext in autotexts:
            autotext.set_fontsize(12)
            autotext.set_weight('bold')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def plot_technique_bar_chart(self, technique_counts, title="Fencing Techniques Frequency", 
                               filename="technique_bar_chart.png"):
        """
        Create a bar chart of technique frequencies
        
        Args:
            technique_counts: Dict of technique names and their counts
            title: Title of the chart
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort techniques by count (descending)
        sorted_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
        techniques, counts = zip(*sorted_techniques) if sorted_techniques else ([], [])
        
        # Get colors for the techniques
        colors = [self.technique_colors.get(tech, '#CCCCCC') for tech in techniques]
        
        bars = ax.bar(techniques, counts, color=colors)
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontsize=12)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Techniques', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        return os.path.join(self.output_dir, filename)
        
    def plot_technique_timeline(self, timeline_data, title="Technique Timeline", 
                              filename="technique_timeline.png"):
        """
        Create a timeline visualization of techniques
        
        Args:
            timeline_data: List of (frame_idx, technique) tuples
            title: Title of the chart
            filename: Output filename
        """
        if not timeline_data:
            return None
            
        # Extract frames and techniques
        frames, techniques = zip(*timeline_data)
        
        # Create a categorical color map
        unique_techniques = list(set(techniques))
        technique_to_id = {tech: i for i, tech in enumerate(unique_techniques)}
        
        # Get colors for the techniques
        colors = [self.technique_colors.get(tech, '#CCCCCC') for tech in unique_techniques]
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot points
        scatter = ax.scatter(frames, [1] * len(frames), 
                  c=[technique_to_id[tech] for tech in techniques],
                  cmap=plt.cm.get_cmap('Set2', len(unique_techniques)),
                  s=100, alpha=0.8)
        
        # Add legend
        legend1 = ax.legend(scatter.legend_elements()[0], 
                           unique_techniques,
                           title="Techniques", loc="upper right")
        ax.add_artist(legend1)
        
        # Customize the plot
        plt.title(title, fontsize=16)
        plt.xlabel('Frame', fontsize=14)
        plt.yticks([])  # Hide y-axis
        
        # Add vertical lines to indicate sequence boundaries
        for i in range(1, len(frames)):
            if frames[i] - frames[i-1] > 10:  # Threshold for new sequence
                plt.axvline(x=(frames[i] + frames[i-1])/2, color='gray', linestyle='--', alpha=0.5)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def plot_sequence_transitions(self, sequence_data, title="Technique Transitions", 
                                filename="technique_transitions.png"):
        """
        Create a transition matrix visualization showing how techniques flow into one another
        
        Args:
            sequence_data: List of technique sequences
            title: Title of the chart
            filename: Output filename
        """
        # Count transitions between techniques
        transitions = defaultdict(Counter)
        
        for sequence in sequence_data:
            if len(sequence) < 2:
                continue
                
            for i in range(len(sequence) - 1):
                from_technique, to_technique = sequence[i], sequence[i+1]
                transitions[from_technique][to_technique] += 1
        
        if not transitions:
            return None
            
        # Convert to matrix form
        all_techniques = sorted(set(k for k in transitions.keys()) | 
                             set(k for d in transitions.values() for k in d.keys()))
        
        matrix = np.zeros((len(all_techniques), len(all_techniques)))
        
        for i, from_tech in enumerate(all_techniques):
            for j, to_tech in enumerate(all_techniques):
                matrix[i, j] = transitions[from_tech][to_tech]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='viridis',
                   xticklabels=all_techniques, yticklabels=all_techniques)
        
        plt.title(title, fontsize=16)
        plt.xlabel('To Technique', fontsize=14)
        plt.ylabel('From Technique', fontsize=14)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def plot_hit_heatmap(self, hit_data, fencer_names=None, title="Hit Distribution", 
                       filename="hit_heatmap.png"):
        """
        Create a heatmap showing where fencers are hitting each other
        
        Args:
            hit_data: Dict of (fencer_id, target_zone): count
            fencer_names: Optional list of fencer names
            title: Title of the chart
            filename: Output filename
        """
        if not hit_data:
            return None
            
        # Extract unique fencers and target zones
        fencers = set()
        zones = set()
        
        for (fencer, zone), _ in hit_data.items():
            fencers.add(fencer)
            zones.add(zone)
        
        fencers = sorted(fencers)
        zones = sorted(zones)
        
        if fencer_names and len(fencer_names) >= len(fencers):
            fencer_labels = [fencer_names[i] for i in range(len(fencers))]
        else:
            fencer_labels = [f"Fencer {i+1}" for i in range(len(fencers))]
        
        # Create matrix
        matrix = np.zeros((len(fencers), len(zones)))
        
        for (fencer, zone), count in hit_data.items():
            fencer_idx = fencers.index(fencer)
            zone_idx = zones.index(zone)
            matrix[fencer_idx, zone_idx] = count
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd',
                   xticklabels=zones, yticklabels=fencer_labels)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Target Zone', fontsize=14)
        plt.ylabel('Fencer', fontsize=14)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def create_technique_summary_dashboard(self, technique_counts, timeline_data=None, 
                                         sequence_data=None, hit_data=None,
                                         title="Fencing Analysis Summary",
                                         filename="fencing_summary.png"):
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            technique_counts: Dict of technique counts
            timeline_data: Optional timeline data
            sequence_data: Optional sequence transition data
            hit_data: Optional hit distribution data
            title: Dashboard title
            filename: Output filename
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle(title, fontsize=20, y=0.98)
        
        # Plot technique distribution pie chart
        ax1 = plt.subplot(2, 2, 1)
        filtered_counts = {k: v for k, v in technique_counts.items() if v > 0}
        colors = [self.technique_colors.get(tech, '#CCCCCC') for tech in filtered_counts.keys()]
        wedges, texts, autotexts = ax1.pie(
            filtered_counts.values(), 
            labels=filtered_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            colors=colors
        )
        ax1.set_title("Technique Distribution", fontsize=16)
        ax1.axis('equal')
        
        # Plot technique bar chart
        ax2 = plt.subplot(2, 2, 2)
        sorted_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
        techniques, counts = zip(*sorted_techniques) if sorted_techniques else ([], [])
        colors = [self.technique_colors.get(tech, '#CCCCCC') for tech in techniques]
        bars = ax2.bar(techniques, counts, color=colors)
        ax2.set_title("Technique Frequency", fontsize=16)
        ax2.set_ylabel("Count", fontsize=12)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot timeline if available
        if timeline_data:
            ax3 = plt.subplot(2, 2, 3)
            frames, techniques = zip(*timeline_data)
            unique_techniques = list(set(techniques))
            technique_to_id = {tech: i for i, tech in enumerate(unique_techniques)}
            colors = [self.technique_colors.get(tech, '#CCCCCC') for tech in unique_techniques]
            scatter = ax3.scatter(frames, [1] * len(frames), 
                      c=[technique_to_id[tech] for tech in techniques],
                      cmap=plt.cm.get_cmap('Set2', len(unique_techniques)),
                      s=100, alpha=0.8)
            ax3.set_title("Technique Timeline", fontsize=16)
            ax3.set_xlabel("Frame", fontsize=12)
            ax3.set_yticks([])
            ax3.legend(scatter.legend_elements()[0], unique_techniques, 
                      title="Techniques", loc="upper right")
        
        # Plot sequence transitions if available
        if sequence_data and len(sequence_data) > 0:
            ax4 = plt.subplot(2, 2, 4)
            # Count transitions between techniques
            transitions = defaultdict(Counter)
            for sequence in sequence_data:
                if len(sequence) < 2:
                    continue
                for i in range(len(sequence) - 1):
                    from_technique, to_technique = sequence[i], sequence[i+1]
                    transitions[from_technique][to_technique] += 1
            
            if transitions:
                all_techniques = sorted(set(k for k in transitions.keys()) | 
                                     set(k for d in transitions.values() for k in d.keys()))
                matrix = np.zeros((len(all_techniques), len(all_techniques)))
                for i, from_tech in enumerate(all_techniques):
                    for j, to_tech in enumerate(all_techniques):
                        matrix[i, j] = transitions[from_tech][to_tech]
                
                sns.heatmap(matrix, annot=True, fmt='g', cmap='viridis',
                           xticklabels=all_techniques, yticklabels=all_techniques, ax=ax4)
                ax4.set_title("Technique Transitions", fontsize=16)
                ax4.set_xlabel("To Technique", fontsize=12)
                ax4.set_ylabel("From Technique", fontsize=12)
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                plt.setp(ax4.get_yticklabels(), rotation=0)
        
        # Save the dashboard
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the main title
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        return os.path.join(self.output_dir, filename)
    
    def visualize_hit_detection(self, frame, fencer_boxes, hit_points=None, 
                              fencer_names=None, frame_idx=None):
        """
        Visualize hit detection on a single frame
        
        Args:
            frame: Video frame (numpy array)
            fencer_boxes: List of bounding boxes for fencers [(x1,y1,x2,y2), ...]
            hit_points: Optional list of hit coordinates [(x,y), ...]
            fencer_names: Optional list of fencer names
            frame_idx: Optional frame index for labeling
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw bounding boxes for fencers
        for i, box in enumerate(fencer_boxes):
            x1, y1, x2, y2 = box
            name = fencer_names[i] if fencer_names and i < len(fencer_names) else f"Fencer {i+1}"
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Draw name
            cv2.putText(annotated_frame, name, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw hit points if available
        if hit_points:
            for hit_point in hit_points:
                x, y = hit_point
                cv2.circle(annotated_frame, (int(x), int(y)), 10, (0, 0, 255), -1)
                cv2.circle(annotated_frame, (int(x), int(y)), 12, (255, 255, 255), 2)
        
        # Add frame number if available
        if frame_idx is not None:
            cv2.putText(annotated_frame, f"Frame: {frame_idx}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        return annotated_frame

def create_fencing_visualization(video_path, analysis_results, output_dir='visualizations'):
    """
    Create comprehensive visualizations for fencing video analysis
    
    Args:
        video_path: Path to the video file
        analysis_results: Dict containing analysis results
        output_dir: Directory to save visualizations
    
    Returns:
        Dict of generated visualization paths
    """
    # Extract required data from analysis results
    technique_counts = analysis_results.get('technique_counts', {})
    timeline_data = analysis_results.get('timeline_data', [])
    sequence_data = analysis_results.get('sequence_data', [])
    hit_data = analysis_results.get('hit_data', {})
    
    # Create visualizer
    visualizer = FencingVisualizer(output_dir)
    
    # Generate visualizations
    visualizations = {}
    
    # Basic technique distributions
    if technique_counts:
        visualizations['pie_chart'] = visualizer.plot_technique_distribution(
            technique_counts,
            title=f"Technique Distribution - {os.path.basename(video_path)}",
            filename=f"{os.path.splitext(os.path.basename(video_path))[0]}_pie.png"
        )
        
        visualizations['bar_chart'] = visualizer.plot_technique_bar_chart(
            technique_counts,
            title=f"Technique Frequency - {os.path.basename(video_path)}",
            filename=f"{os.path.splitext(os.path.basename(video_path))[0]}_bar.png"
        )
    
    # Timeline visualization
    if timeline_data:
        visualizations['timeline'] = visualizer.plot_technique_timeline(
            timeline_data,
            title=f"Technique Timeline - {os.path.basename(video_path)}",
            filename=f"{os.path.splitext(os.path.basename(video_path))[0]}_timeline.png"
        )
    
    # Sequence transitions
    if sequence_data:
        visualizations['transitions'] = visualizer.plot_sequence_transitions(
            sequence_data,
            title=f"Technique Transitions - {os.path.basename(video_path)}",
            filename=f"{os.path.splitext(os.path.basename(video_path))[0]}_transitions.png"
        )
    
    # Hit heatmap
    if hit_data:
        visualizations['hit_heatmap'] = visualizer.plot_hit_heatmap(
            hit_data,
            title=f"Hit Distribution - {os.path.basename(video_path)}",
            filename=f"{os.path.splitext(os.path.basename(video_path))[0]}_hits.png"
        )
    
    # Create summary dashboard
    visualizations['dashboard'] = visualizer.create_technique_summary_dashboard(
        technique_counts,
        timeline_data,
        sequence_data, 
        hit_data,
        title=f"Fencing Analysis - {os.path.basename(video_path)}",
        filename=f"{os.path.splitext(os.path.basename(video_path))[0]}_summary.png"
    )
    
    return visualizations

# Example usage
if __name__ == "__main__":
    # Sample data
    technique_counts = {
        'attack': 15,
        'defense': 8,
        'parry': 12,
        'riposte': 5,
        'lunge': 10,
        'advance': 20,
        'retreat': 18,
        'feint': 3
    }
    
    timeline_data = [
        (10, 'advance'), (20, 'attack'), (30, 'parry'),
        (40, 'riposte'), (60, 'retreat'), (80, 'advance'),
        (90, 'lunge'), (100, 'retreat')
    ]
    
    sequence_data = [
        ['advance', 'attack', 'retreat'],
        ['parry', 'riposte', 'retreat'],
        ['advance', 'lunge', 'retreat'],
        ['advance', 'feint', 'attack']
    ]
    
    hit_data = {
        (0, 'head'): 3,
        (0, 'torso'): 5,
        (0, 'arm'): 2,
        (1, 'head'): 1,
        (1, 'torso'): 4,
        (1, 'arm'): 3,
    }
    
    analysis_results = {
        'technique_counts': technique_counts,
        'timeline_data': timeline_data,
        'sequence_data': sequence_data,
        'hit_data': hit_data
    }
    
    # Create visualizations
    visualizations = create_fencing_visualization(
        "sample_video.mp4",
        analysis_results
    )
    
    print("Generated visualizations:")
    for name, path in visualizations.items():
        print(f"{name}: {path}") 