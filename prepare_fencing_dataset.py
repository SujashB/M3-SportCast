import os
import argparse
import shutil
from glob import glob
import cv2
from tqdm import tqdm

def create_dataset_directory(base_dir, categories):
    """Create the dataset directory structure"""
    os.makedirs(base_dir, exist_ok=True)
    
    for category in categories:
        category_dir = os.path.join(base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        print(f"Created directory: {category_dir}")

def check_video_integrity(video_path):
    """Check if a video file is valid and can be opened"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        if not ret or frame is None:
            return False
        cap.release()
        return True
    except Exception as e:
        print(f"Error checking video {video_path}: {e}")
        return False

def copy_videos_to_dataset(source_videos, base_dir, category, min_length=1.0, max_length=30.0):
    """Copy valid videos to the dataset directory with the specified category"""
    target_dir = os.path.join(base_dir, category)
    os.makedirs(target_dir, exist_ok=True)
    
    copied_count = 0
    skipped_count = 0
    
    for video_path in tqdm(source_videos, desc=f"Copying videos to {category}"):
        # Check if it's a valid video
        if not check_video_integrity(video_path):
            print(f"Skipping invalid video: {video_path}")
            skipped_count += 1
            continue
        
        # Check video length
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration < min_length or duration > max_length:
            print(f"Skipping video with duration {duration:.2f}s: {video_path}")
            skipped_count += 1
            continue
        
        # Get filename and copy to target directory
        filename = os.path.basename(video_path)
        target_path = os.path.join(target_dir, filename)
        
        # Copy the file
        shutil.copy2(video_path, target_path)
        copied_count += 1
    
    print(f"Copied {copied_count} videos to {category}, skipped {skipped_count} videos")
    return copied_count

def main():
    parser = argparse.ArgumentParser(description="Prepare a fencing dataset for fine-tuning VideoMAE")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to create the dataset in")
    parser.add_argument("--categories", type=str, nargs="+", required=True,
                        help="List of fencing categories to create (e.g., 'attack' 'defense' 'footwork')")
    parser.add_argument("--source_dirs", type=str, nargs="+", required=True,
                        help="Source directories containing raw videos")
    parser.add_argument("--category_mapping", type=str, nargs="+", required=True,
                        help="Category for each source directory in the same order")
    parser.add_argument("--min_length", type=float, default=1.0,
                        help="Minimum video length in seconds")
    parser.add_argument("--max_length", type=float, default=30.0,
                        help="Maximum video length in seconds")
    
    args = parser.parse_args()
    
    # Validate number of source_dirs matches category_mapping
    if len(args.source_dirs) != len(args.category_mapping):
        print("Error: Number of source directories must match number of category mappings")
        return
    
    # Validate that all category mappings are in the list of categories
    invalid_categories = [cat for cat in args.category_mapping if cat not in args.categories]
    if invalid_categories:
        print(f"Error: Invalid category mappings: {invalid_categories}")
        print(f"Valid categories are: {args.categories}")
        return
    
    # Create the dataset directory structure
    create_dataset_directory(args.output_dir, args.categories)
    
    # Process each source directory
    total_copied = 0
    for source_dir, category in zip(args.source_dirs, args.category_mapping):
        video_extensions = ["mp4", "avi", "mov", "mkv"]
        source_videos = []
        
        for ext in video_extensions:
            pattern = os.path.join(source_dir, f"*.{ext}")
            source_videos.extend(glob(pattern))
        
        print(f"Found {len(source_videos)} videos in {source_dir}")
        
        # Copy videos to dataset directory
        copied = copy_videos_to_dataset(source_videos, args.output_dir, category, 
                                        args.min_length, args.max_length)
        total_copied += copied
    
    print(f"Dataset preparation complete. Total videos in dataset: {total_copied}")
    
    # Print directory structure statistics
    print("\nDataset Statistics:")
    for category in args.categories:
        category_dir = os.path.join(args.output_dir, category)
        video_count = len(glob(os.path.join(category_dir, "*.mp4")))
        video_count += len(glob(os.path.join(category_dir, "*.avi")))
        video_count += len(glob(os.path.join(category_dir, "*.mov")))
        video_count += len(glob(os.path.join(category_dir, "*.mkv")))
        print(f"  - {category}: {video_count} videos")

if __name__ == "__main__":
    main() 