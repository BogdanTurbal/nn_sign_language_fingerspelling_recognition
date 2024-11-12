# ASL Fingerspelling Recognition
American Sign Language Fingerspelling Recognition project based on AI, particularly 2d Deepwise CNN + Transformers + CTC Loss

A deep learning model for recognizing American Sign Language (ASL) fingerspelling from video landmarks, achieving efficient real-time translation of fingerspelled words to text.

## Architecture Overview

The model uses a hybrid CNN-Transformer architecture:

- **Input**: 300 frames x 273 landmarks (face, hands, pose coordinates)
- **Preprocessing**: 
  - Normalization using pre-computed mean/std
  - Frame padding/resizing for uniform sequence length
  - Gaussian noise augmentation for robustness

- **Main Architecture**:
  1. Dense embedding layer + positional encoding
  2. 6 blocks of:
     - Multi-scale temporal convolutions (kernel sizes: 11,5,3)
     - Transformer encoder (6 attention heads)
     - Max pooling at blocks 2 & 4
  3. CTC loss for sequence prediction

- **Output**: Character-level predictions with consecutive duplicate removal

## Training

- Dataset: 3M+ fingerspelled characters from 100+ Deaf signers
- Mixed-precision training with Adam optimizer
- CTC loss for sequence alignment
- Data augmentation: Gaussian noise, hand flipping

## Usage

The model takes landmark coordinates from video frames and outputs the corresponding text:

```python
interpreter = tf.lite.Interpreter("model.tflite")
runner = interpreter.get_signature_runner("serving_default")
prediction = runner(inputs=landmarks)["outputs"]
text = "".join([char_map[i] for i in np.argmax(prediction, axis=1)])
```

## Performance

- Inference time: <5ms per frame on CPU
- Memory footprint: ~40MB
- Normalized total levenshtein distance on private kaggle leaderboard: 0.766
