import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
import datetime

# Import our modules
try:
    from pose_estimation_helper import process_video_with_pose, extract_pose_descriptions
    from simple_fencing_analyzer import FencingAnalyzer
    from sequence_analysis import FencingSequenceAnalyzer
    from knowledge_graph_rules import FencingReasoningEngine
    from fencer_segmentation import FencerSegmentation
    from data_visualization import create_fencing_visualization, FencingVisualizer
    
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing modules: {e}")
    MODULES_AVAILABLE = False

class AdvancedFencingAnalyzer:
    """
    Comprehensive fencing analysis system that integrates multiple analysis approaches:
    1. VideoMAE for technique classification
    2. Pose estimation for movement analysis
    3. Sequence analysis for temporal understanding
    4. Knowledge graph with logical rules
    5. Fencer segmentation and tracking
    6. Data visualization
    """
    def __init__(self, sam_checkpoint=None):
        """
        Initialize the advanced analyzer
        
        Args:
            sam_checkpoint: Optional path to SAM model checkpoint for segmentation
        """
        print("Initializing Advanced Fencing Analyzer...")
        
        # Initialize core analyzer (VideoMAE + basic pose)
        print("Loading core analyzer module...")
        self.core_analyzer = FencingAnalyzer()
        
        # Initialize sequence analyzer
        print("Loading sequence analyzer module...")
        self.sequence_analyzer = FencingSequenceAnalyzer()
        
        # Initialize reasoning engine
        print("Loading knowledge graph reasoning engine...")
        self.reasoning_engine = FencingReasoningEngine()
        
        # Initialize segmentation (with SAM if available)
        print("Loading segmentation module...")
        self.segmenter = FencerSegmentation(sam_checkpoint=sam_checkpoint)
        
        # Initialize visualizer
        print("Loading visualization module...")
        self.visualizer = FencingVisualizer(output_dir='visualizations')
        
        print("All modules loaded successfully")
    
    def analyze_video(self, video_path, output_dir="results", num_frames=None, 
                     save_visualization=True, save_segmentation=True):
        """
        Perform comprehensive analysis on a fencing video
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save results
            num_frames: Maximum number of frames to process (None for all)
            save_visualization: Whether to save visualization outputs
            save_segmentation: Whether to save segmentation outputs
            
        Returns:
            results: Dict containing comprehensive analysis results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Analyzing video: {video_path}")
        
        # Step 1: Basic VideoMAE + Pose analysis
        print("Step 1: Running core analysis (VideoMAE + Pose)...")
        videomae_scores, techniques = self.core_analyzer.analyze_video(video_path)
        
        # Convert scores to dict for easier handling
        videomae_dict = {
            techniques[i]: score for i, score in enumerate(videomae_scores)
        }
        
        print("\nVideoMAE Classification Results:")
        for technique, score in sorted(videomae_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {technique}: {score:.4f}")
        
        # Step 2: Fencer segmentation and tracking
        print("\nStep 2: Segmenting and tracking fencers...")
        segmentation_path = os.path.join(output_dir, f"{video_name}_segmentation.mp4")
        if save_segmentation:
            fencer_data = self.segmenter.process_video(
                video_path=video_path,
                output_path=segmentation_path,
                max_frames=num_frames
            )
        else:
            fencer_data = self.segmenter.process_video(
                video_path=video_path,
                output_path=None,
                max_frames=num_frames
            )
        
        print(f"Tracked {len(fencer_data['trajectories'])} fencers")
        
        # Generate heatmap of fencer movement
        if save_visualization:
            heatmap_path = os.path.join(output_dir, f"{video_name}_heatmap.png")
            self.segmenter.generate_heatmap(
                fencer_data=fencer_data,
                output_path=heatmap_path
            )
            print(f"Saved fencer movement heatmap to {heatmap_path}")
        
        # Detect potential hits
        hits_path = os.path.join(output_dir, f"{video_name}_hits.png") if save_visualization else None
        hits = self.segmenter.detect_hits(
            fencer_data=fencer_data,
            output_path=hits_path
        )
        
        print(f"Detected {len(hits)} potential hits")
        
        # Step 3: Extract movement sequences
        print("\nStep 3: Analyzing technique sequences...")
        
        # Process video with pose estimation to get frame-by-frame analysis
        pose_analysis_path = os.path.join(output_dir, f"{video_name}_pose_analysis.mp4")
        frame_analyses, _ = process_video_with_pose(
            video_path, 
            pose_analysis_path if save_visualization else None,
            save_frames=save_visualization
        )
        
        # Extract technique timeline
        timeline_data = []
        detected_techniques = set()
        
        for analysis in frame_analyses:
            frame_idx = analysis['frame_idx']
            
            # Check if any technique was detected in this frame
            technique_detected = False
            for technique in techniques:
                if technique.lower() in analysis['sentence'].lower():
                    timeline_data.append((frame_idx, technique))
                    detected_techniques.add(technique)
                    technique_detected = True
                    break
            
            # Add generic "movement" if no specific technique detected
            if not technique_detected:
                timeline_data.append((frame_idx, "movement"))
        
        # Extract sequences from timeline
        sequences = self.sequence_analyzer.extract_sequences_from_timeline(timeline_data)
        print(f"Extracted {len(sequences)} technique sequences")
        
        # Feed sequences to sequence analyzer
        for sequence in sequences:
            if len(sequence) >= 2:
                self.sequence_analyzer.add_observed_sequence(sequence)
        
        # Train sequence model on observed data
        print("Training sequence model on observed data...")
        self.sequence_analyzer.train_model(sequences, epochs=30)
        
        # Save model if directory exists
        model_dir = os.path.join(output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{video_name}_sequence_model.pth")
        self.sequence_analyzer.save_model(model_path)
        print(f"Saved sequence model to {model_path}")
        
        # Step 4: Apply knowledge graph reasoning
        print("\nStep 4: Applying knowledge graph reasoning...")
        
        # Get transition matrix from sequence analysis
        transition_matrix, labels = self.sequence_analyzer.get_transition_matrix()
        
        # Add each significant transition as a rule in the knowledge graph
        for i, from_tech in enumerate(labels):
            for j, to_tech in enumerate(labels):
                if transition_matrix[i, j] > 0.3:  # Only add significant transitions
                    self.reasoning_engine.rules_engine.add_rule(
                        from_tech, to_tech, transition_matrix[i, j]
                    )
        
        # Visualize knowledge graph
        if save_visualization:
            kg_path = os.path.join(output_dir, f"{video_name}_knowledge_graph.png")
            self.reasoning_engine.visualize_knowledge(kg_path)
            print(f"Saved knowledge graph visualization to {kg_path}")
        
        # Create signature sequences for each fencer
        fencer_sequences = {}
        for fencer_id in fencer_data['trajectories'].keys():
            # Extract frames where this fencer appears
            fencer_frames = set(fencer_data['frames'][fencer_id])
            
            # Extract techniques in these frames
            fencer_timeline = [
                (frame_idx, technique) for frame_idx, technique in timeline_data
                if frame_idx in fencer_frames
            ]
            
            # Convert to sequences
            fencer_seqs = self.sequence_analyzer.extract_sequences_from_timeline(fencer_timeline)
            if fencer_seqs:
                fencer_sequences[fencer_id] = fencer_seqs
                
                # Identify signature sequences
                signatures = self.sequence_analyzer.identify_signature_sequences(fencer_seqs)
                print(f"\nFencer {fencer_id} signature sequences:")
                for sequence, count in signatures:
                    print(f"  {' → '.join(sequence)}: {count} occurrences")
        
        # Step 5: Generate visualizations
        print("\nStep 5: Generating visualizations...")
        
        # Count technique occurrences
        technique_counts = {}
        for _, technique in timeline_data:
            if technique not in technique_counts:
                technique_counts[technique] = 0
            technique_counts[technique] += 1
        
        # Create hit data
        hit_data = {}
        for hit in hits:
            fencer_pair = hit['fencers']
            if fencer_pair not in hit_data:
                hit_data[fencer_pair] = 0
            hit_data[fencer_pair] += 1
        
        # Prepare visualization data
        analysis_results = {
            'technique_counts': technique_counts,
            'timeline_data': timeline_data,
            'sequence_data': sequences,
            'hit_data': hit_data
        }
        
        # Create visualizations
        if save_visualization:
            vis_paths = create_fencing_visualization(
                video_path, 
                analysis_results, 
                output_dir=output_dir
            )
            
            print("Generated visualizations:")
            for name, path in vis_paths.items():
                print(f"  {name}: {path}")
        
        # Step 6: Package all results
        print("\nStep 6: Packaging all results...")
        
        # Compile all results
        results = {
            'video_path': video_path,
            'videomae_scores': videomae_dict,
            'detected_techniques': list(detected_techniques),
            'technique_counts': technique_counts,
            'sequences': sequences,
            'timeline_data': timeline_data,
            'fencer_data': {
                'count': len(fencer_data['trajectories']),
                'trajectories': {str(k): v for k, v in fencer_data['trajectories'].items()},
                'frames': {str(k): v for k, v in fencer_data['frames'].items()}
            },
            'hits': hits,
            'fencer_sequences': {str(k): v for k, v in fencer_sequences.items()}
        }
        
        # Save results to JSON
        results_path = os.path.join(output_dir, f"{video_name}_analysis.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = json.loads(
                json.dumps(results, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else None)
            )
            json.dump(serializable_results, f, indent=2)
        
        print(f"Analysis results saved to {results_path}")
        
        return results
    
    def generate_report(self, results, output_path):
        """
        Generate a comprehensive HTML report from analysis results
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save the HTML report
        """
        # Get video name
        video_name = os.path.splitext(os.path.basename(results['video_path']))[0]
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fencing Analysis: {video_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .section {{
                    margin-bottom: 30px;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 20px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .visualization {{
                    margin: 20px 0;
                    max-width: 100%;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fencing Video Analysis Report</h1>
                <p>Video: {results['video_path']}</p>
                <p>Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>This report presents a comprehensive analysis of fencing techniques, movements, and interactions.</p>
                <p>Number of detected fencers: {results['fencer_data']['count']}</p>
                <p>Number of detected techniques: {len(results['detected_techniques'])}</p>
                <p>Number of identified sequences: {len(results['sequences'])}</p>
                <p>Number of potential hits: {len(results['hits'])}</p>
            </div>
            
            <div class="section">
                <h2>Technique Classification</h2>
                <table>
                    <tr>
                        <th>Technique</th>
                        <th>Score</th>
                        <th>Count</th>
                    </tr>
        """
        
        # Add technique scores
        for technique, score in sorted(results['videomae_scores'].items(), key=lambda x: x[1], reverse=True):
            count = results['technique_counts'].get(technique, 0)
            html_content += f"""
                    <tr>
                        <td>{technique}</td>
                        <td>{score:.4f}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
        """
        
        # Add visualizations
        visualizations = [
            f"{video_name}_summary.png",
            f"{video_name}_pie.png",
            f"{video_name}_timeline.png",
            f"{video_name}_transitions.png",
            f"{video_name}_heatmap.png",
            f"{video_name}_knowledge_graph.png"
        ]
        
        for vis in visualizations:
            vis_path = os.path.join(os.path.dirname(output_path), vis)
            if os.path.exists(vis_path):
                html_content += f"""
                <div class="visualization">
                    <h3>{vis.replace(f"{video_name}_", "").replace(".png", "").title()}</h3>
                    <img src="{vis}" alt="{vis}" />
                </div>
                """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Sequences</h2>
                <h3>Most Common Sequences</h3>
                <ul>
        """
        
        # Add sequences
        sequence_counts = {}
        for sequence in results['sequences']:
            if len(sequence) >= 3:  # Only sequences of length 3+
                seq_str = " → ".join(sequence)
                if seq_str not in sequence_counts:
                    sequence_counts[seq_str] = 0
                sequence_counts[seq_str] += 1
        
        for seq, count in sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            html_content += f"""
                <li>{seq} ({count} occurrences)</li>
            """
        
        html_content += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Fencer Analysis</h2>
        """
        
        # Add fencer information
        for fencer_id, trajectories in results['fencer_data']['trajectories'].items():
            # Check if we have sequences for this fencer
            fencer_seqs = results['fencer_sequences'].get(fencer_id, [])
            
            html_content += f"""
                <h3>Fencer {fencer_id}</h3>
                <p>Frames present: {len(results['fencer_data']['frames'][fencer_id])}</p>
            """
            
            if fencer_seqs:
                html_content += """
                <h4>Signature Sequences</h4>
                <ul>
                """
                
                # Count sequence occurrences
                seq_counts = {}
                for seq in fencer_seqs:
                    if len(seq) >= 3:
                        seq_str = " → ".join(seq)
                        if seq_str not in seq_counts:
                            seq_counts[seq_str] = 0
                        seq_counts[seq_str] += 1
                
                for seq, count in sorted(seq_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                    html_content += f"""
                    <li>{seq} ({count} occurrences)</li>
                    """
                
                html_content += """
                </ul>
                """
        
        html_content += """
            </div>
            
            <div class="section">
                <h2>Hits Analysis</h2>
                <table>
                    <tr>
                        <th>Frame</th>
                        <th>Fencers</th>
                        <th>Distance</th>
                    </tr>
        """
        
        # Add hits
        for hit in sorted(results['hits'], key=lambda x: x['frame']):
            html_content += f"""
                    <tr>
                        <td>{hit['frame']}</td>
                        <td>Fencer {hit['fencers'][0]} vs Fencer {hit['fencers'][1]}</td>
                        <td>{hit['distance']:.2f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated at {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced Fencing Video Analysis")
    parser.add_argument("video_path", help="Path to the fencing video")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--sam_checkpoint", default=None, help="Path to SAM model checkpoint (optional)")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization outputs")
    parser.add_argument("--no_seg", action="store_true", help="Disable segmentation video output")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} not found!")
        return
    
    # Check if modules are available
    if not MODULES_AVAILABLE:
        print("Error: Required modules not available!")
        return
    
    # Initialize analyzer
    analyzer = AdvancedFencingAnalyzer(sam_checkpoint=args.sam_checkpoint)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze video
    results = analyzer.analyze_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.max_frames,
        save_visualization=not args.no_viz,
        save_segmentation=not args.no_seg
    )
    
    # Generate report
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    report_path = os.path.join(args.output_dir, f"{video_name}_report.html")
    analyzer.generate_report(results, report_path)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()