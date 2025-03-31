# Fine-tuning VideoMAE for Fencing Video Classification

This README explains how to fine-tune the VideoMAE model on a custom dataset of fencing videos to create a specialized fencing action classifier.

## Overview

The process involves three main steps:
1. **Preparing the dataset**: Organize fencing videos into categories
2. **Fine-tuning the model**: Train the pre-trained VideoMAE model on your fencing dataset
3. **Evaluating and using the model**: Test the fine-tuned model on new fencing videos

## Requirements

Install the required dependencies by running:

```bash
pip install -r requirements_finetuning.txt
```

## 1. Preparing the Dataset

### Dataset Structure

For fine-tuning, you need to organize your fencing videos into categories. Each category should be a subdirectory containing video files. For example:

```
fencing_dataset/
├── attack/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── defense/
│   ├── video3.mp4
│   ├── video4.mp4
│   └── ...
├── footwork/
│   ├── video5.mp4
│   ├── video6.mp4
│   └── ...
└── ...
```

### Using the Dataset Preparation Script

We provide a script to help organize your videos into the correct structure:

```bash
python prepare_fencing_dataset.py \
  --output_dir fencing_dataset \
  --categories attack defense footwork parry riposte \
  --source_dirs /path/to/attack/videos /path/to/defense/videos /path/to/footwork/videos \
  --category_mapping attack defense footwork \
  --min_length 1.0 \
  --max_length 30.0
```

This script will:
1. Create the necessary directories
2. Copy and organize videos from your source directories to the dataset structure
3. Check video integrity and duration
4. Report statistics about the resulting dataset

### Dataset Recommendations

For best results:
- Aim for at least 50-100 videos per category
- Keep videos relatively short (5-30 seconds) and focused on a single action
- Include diverse examples of each action (different angles, lighting, fencers)
- Ensure balanced categories (similar number of videos in each category)
- Consider including a "background" or "other" category for non-specific actions

## 2. Fine-tuning the Model

Once your dataset is prepared, you can fine-tune the VideoMAE model:

```bash
python finetune_videomae.py \
  --data_dir fencing_dataset \
  --output_dir finetuned_videomae_fencing \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --num_epochs 10
```

### Fine-tuning Parameters

- `--data_dir`: Directory containing your fencing dataset
- `--output_dir`: Directory to save the fine-tuned model
- `--pretrained_model`: Pre-trained model to start from (default: "MCG-NJU/videomae-base-finetuned-kinetics")
- `--num_frames`: Number of frames to extract from each video (default: 16)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate for training (default: 5e-5)
- `--num_epochs`: Number of epochs for training (default: 10)

### Resources Required

- GPU with at least 8GB of VRAM is strongly recommended
- Training time depends on dataset size and GPU capability (typically a few hours)
- Disk space for dataset and model weights (several GB)

## 3. Evaluating the Model

After fine-tuning, evaluate your model on test videos:

```bash
python evaluate_finetuned_model.py \
  --model_dir finetuned_videomae_fencing \
  --test_dir fencing_test_dataset \
  --output_dir evaluation_results \
  --use_gpu
```

### Running Inference on a Single Video

To classify a single fencing video:

```bash
python evaluate_finetuned_model.py \
  --model_dir finetuned_videomae_fencing \
  --output_dir inference_results \
  --single_video path/to/your/fencing_video.mp4 \
  --use_gpu
```

## Tips for Improving Fine-tuning Results

1. **Data Augmentation**: If your dataset is small, consider data augmentation techniques like temporal cropping, spatial cropping, or color jittering.

2. **Learning Rate**: If training is unstable, try reducing the learning rate. If it's too slow, try increasing it.

3. **Model Size**: If fine-tuning is too slow or memory-intensive, consider using a smaller variant of VideoMAE.

4. **Transfer Learning Stages**: Start by only training the classification head (freeze the backbone), then gradually unfreeze more layers.

5. **Validation**: Always keep a separate validation set to monitor performance and prevent overfitting.

## Reusing the Fine-tuned Model

After fine-tuning, you can use your model in the same way as the original VideoMAE model, but with fencing-specific categories:

```python
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import torch

# Load your fine-tuned model
model = VideoMAEForVideoClassification.from_pretrained("path/to/finetuned_videomae_fencing")
processor = VideoMAEImageProcessor.from_pretrained("path/to/finetuned_videomae_fencing")

# Extract frames from video (see extract_frames function in evaluate_finetuned_model.py)
frames = extract_frames(video_path, model.config.num_frames, model.config.image_size)

# Run inference
inputs = processor(frames, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]
print(f"Predicted fencing action: {predicted_label}")
```

## Common Issues and Solutions

- **Out of Memory Errors**: Reduce batch size or number of frames
- **Overfitting**: Use more data, add regularization, or reduce model complexity
- **Poor Performance**: Ensure dataset quality, try different learning rates, train longer
- **Slow Training**: Use a GPU, reduce batch size or model size

## References

- [VideoMAE Paper](https://arxiv.org/abs/2203.12602)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/videomae)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) 