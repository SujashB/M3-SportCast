# VideoMAE for Video Classification

This is an implementation of VideoMAE (Video Masked Autoencoder) for video classification, based on the [HuggingFace example notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/video_classification.ipynb).

## Installation

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

You can run the basic example script with:

```bash
python main.py
```

This will:
1. Download a sample video
2. Load the VideoMAE model pre-trained on Kinetics-400
3. Predict the class of the action in the video
4. Display the top 5 predicted classes with their probabilities

### Demo with Visualization

For a more visual demonstration that shows video frames and prediction charts:

```bash
python demo.py
```

This demo script will:
1. Download and load a sample video
2. Display sample frames from the video (saving them as "video_frames.png")
3. Predict action classes using VideoMAE
4. Visualize the top predictions in a bar chart (saving it as "prediction_scores.png")

### Classify Your Own Videos

To classify your own videos, use the classify_custom_video.py script:

```bash
python classify_custom_video.py --video_path path/to/your/video.mp4 --top_k 10
```

Command line arguments:
- `--video_path`: Path to your video file (required)
- `--model`: HuggingFace model ID (default: "MCG-NJU/videomae-base-finetuned-kinetics")
- `--top_k`: Number of top predictions to display (default: 10)

## Model Details

This implementation uses the `MCG-NJU/videomae-base-finetuned-kinetics` model from HuggingFace, which is a VideoMAE model fine-tuned on the Kinetics-400 dataset. The model can classify videos into 400 different action categories.

The Video Masked Autoencoder (VideoMAE) architecture is designed specifically for video understanding tasks and is based on the Vision Transformer (ViT) architecture, extended to handle the temporal dimension of videos.

## References

- [VideoMAE Paper](https://arxiv.org/abs/2203.12602)
- [HuggingFace VideoMAE Documentation](https://huggingface.co/docs/transformers/model_doc/videomae)
- [HuggingFace VideoMAE Example Notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/video_classification.ipynb)
