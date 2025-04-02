from fencer_detector import FencerDetector

def main():
    # Initialize detector with pretrained model
    print("Initializing fencer detector...")
    detector = FencerDetector()
    
    # Process the video
    video_path = "evenevenmorecropped (1).mp4"
    output_path = "output_detection.mp4"
    
    print(f"\nProcessing video: {video_path}")
    print("This will detect and track fencers in the video...")
    
    detector.process_video(
        video_path=video_path,
        output_path=output_path
    )
    
    print(f"\nProcessing complete!")
    print(f"Output saved as: {output_path}")

if __name__ == "__main__":
    main() 