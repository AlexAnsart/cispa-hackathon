# European Championship in Trustworthy AI - CISPA Hackathon

Team: Loki  
Members: Alexandre Ansart, Bastian Paoli, Florian Dougnon-Greder, Melissa Abider  
Rank: 2nd Place out of 16 Teams  
Prize: 400 euros and qualification for the European Finals in Germany (July 2025)  
Location: La Grande Arche, La Defense, Paris  
Date: November 2024

## Overview

This repository contains the solutions developed by our team during the 24-hour AI and Cybersecurity Hackathon organized by CISPA (Helmholtz Center for Information Security), the world-leading cybersecurity research institution based in Germany. The event gathered PhD students and top engineering students from prestigious European institutions, including Ecole Polytechnique and various Swiss and Polish universities, to solve cutting-edge problems in Trustworthy AI.

The hackathon took place in La Defense, Paris' business district, and featured a live leaderboard that updated throughout the 24-hour competition period. Participants competed on three distinct tasks related to machine learning security and forensics, with submissions evaluated in real-time against hidden test sets.

Our team secured the 2nd place on the live leaderboard, demonstrating robust solutions against black-box models and advanced forensic capabilities. This performance qualified us for the European Grand Finals, which will take place in Germany in July 2025, with transportation and accommodation expenses covered.

## Task 1: Adversarial Examples (Black-Box Attack)

### Problem Statement

The objective of this task was to generate adversarial examples for 100 natural images. An adversarial example is an image that has been slightly modified to fool an artificial intelligence classifier, while remaining visually unchanged to a human observer. The challenge was to trick a black-box model (a model whose architecture and training data were unknown to us) while minimizing the L2 norm of the perturbation.

The L2 norm represents the mathematical distance between the original image and the modified image. It is computed as the square root of the sum of squared pixel differences across all channels. A lower L2 norm means the modification is less visible and the score is better. The evaluation metric was the average normalized L2 distance, where correctly classified images received a score of 1.0 (the worst possible score).

We had access to a black-box API that could return logits (pre-softmax predictions) for any input image, but we could only query it once every 15 minutes. Submissions could be made every 5 minutes. This constraint made iterative query-based attacks impractical, forcing us to rely on transfer attacks from local surrogate models.

### Attempts 1 and 2: The Learning Phase

Our first attempts involved naive attacks using standard algorithms like Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) on standard surrogate models like ResNet-18 pretrained on ImageNet. These attempts resulted in poor transferability, meaning the attacks worked on our local models but failed on the evaluation server.

In attempt 2, we tried ensemble methods with larger models (ResNet50, DenseNet121, VGG16) which improved the success rate but resulted in excessive perturbation distances because we applied a fixed noise budget (epsilon) to every image regardless of its difficulty. Some images required very little perturbation to be misclassified, while others needed much more. Using a global epsilon meant we were wasting perturbation budget on easy images and failing on hard ones.

### Attempt 3: The Lock and Ram Strategy

We developed a strategy we named Lock and Ram to solve the surrogate mismatch problem. We realized that our local models (ImageNet pretrained models upscaled from 28x28 to 224x224) did not correlate perfectly with the target model (likely a native CIFAR-10 model trained on 32x32 images). This created a transfer gap: attacks optimized locally failed remotely.

#### The Crisis: Surrogate Mismatch

The fundamental problem was that we were attacking models trained on ImageNet (high-resolution natural images) while the target was likely trained on CIFAR-10 (low-resolution images). When we upscaled our 28x28 images to 224x224 to feed them to ImageNet models, we introduced artifacts that the target model did not see. The gradients computed on these upscaled images were therefore misaligned with the true decision boundaries of the black-box model.

#### Why Not Carlini-Wagner?

Standard Carlini-Wagner (C&W) optimization acts like a scalpel: it finds the exact decision boundary of the local model with minimal perturbation. In a black-box setting with surrogate mismatch, this precision is a weakness. The attack barely crosses the local boundary and fails to cross the remote one. We needed a hammer: Binary Search PGD with Confidence Margin. We force a gap (Margin kappa) to ensure the attack survives the transfer.

#### The Strategy: Lock and Ram

Instead of a one-size-fits-all approach, we implemented a dynamic pipeline with three phases:

**LOCK (The Scalpel):** For approximately 32% of the images, we identified that they required very little noise to be misclassified (L2 norm around 0.06). We locked these successful perturbations and did not modify them further, preserving their minimal L2 distance.

**RAM (The Hammer):** For the remaining 68% of images that were resistant to simple attacks, we escalated parameters. We increased epsilon (the maximum allowed perturbation) and kappa (the confidence margin requirement) to force the decision boundary crossing. The kappa parameter is crucial: it represents the difference between the logit of the wrong class and the logit of the true class. By enforcing a high kappa (e.g., 50.0), we forced the algorithm to push the image deep into the wrong class territory, ensuring that the adversarial nature of the image would survive the transfer to the unknown target.

**REFINE (The Polish):** Once an image successfully transferred to the black-box model, we ran a reverse binary search locally to dial back the noise until we found the minimal necessary perturbation. This allowed us to optimize the L2 distance while maintaining successful misclassification.

#### Technical Implementation

**Binary Search PGD (BS-PGD):** Unlike standard PGD which maximizes loss within a fixed epsilon ball, we minimized the L2 norm subject to the constraint that the image is misclassified. We implemented a binary search over epsilon values, testing multiple candidates and selecting the smallest epsilon that still achieves misclassification. For each epsilon candidate, we ran multiple random restarts (typically 15) of PGD to escape local minima in the non-convex loss landscape.

**Hybrid Multi-Scale Ensemble:** We combined two groups of models to maximize transferability. Group A consisted of ImageNet pretrained models (ResNet50, DenseNet121, VGG16, EfficientNet-B0) upscaled to 224x224. These models capture high-level semantic features. Group B consisted of models trained on CIFAR-10 (ResNet18) at native 32x32 resolution. These models capture low-level pixel patterns. By attacking both groups simultaneously, we generated perturbations that transfer to both types of target architectures.

**Input Diversity:** We applied random transformations (scaling, cropping, brightness jitter) during gradient computation to prevent overfitting to specific pixel positions. This forces the attack to find robust perturbations that work across transformations.

**Adaptive Step Size:** The step size alpha was set adaptively as alpha = 2.5 * epsilon / num_steps. This ensures that the optimization is fine-grained enough to converge within the epsilon ball without oscillating.

**Momentum:** We used Momentum Iterative Fast Gradient Sign Method (MI-FGSM) with momentum 0.9 to stabilize gradient directions and escape local minima.

**Feedback Loop:** We automated the submission and analysis cycle. Every 15 minutes (respecting the API rate limit), our system queried the evaluation server to get the real logits, identified which images were still failing, and automatically adjusted the epsilon and kappa parameters for those specific images for the next iteration. This created a closed-loop optimization system that learned from the black-box responses.

The epsilon parameter represents the maximum allowed L2 perturbation distance. It defines the radius of the ball within which we search for adversarial examples. The kappa parameter represents the confidence margin: the difference between the maximum wrong-class logit and the true-class logit. A high kappa means we require the model to be very confident in the wrong prediction, which increases the likelihood of transfer.

## Task 2: Image Attribution (Open-Set Recognition)

### Task Description

This task required us to identify the source of a given image. The images could come from four specific generative models (VAR-16d, RAR-XL, Taming Transformers, Stable Diffusion) or be an outlier (a natural image or one from an unknown model). The evaluation metric was Score = TPR / (1 + 5 * FPR), where TPR is the True Positive Rate and FPR is the False Positive Rate, macro-averaged across the four known classes. This metric penalized false positives very heavily, meaning identifying an outlier as a known model was extremely costly for our score.

The challenge was that we only had 250 labeled training images per class, which is insufficient for training deep neural networks. Additionally, the outlier class is mathematically infinite and cannot be fully learned. Standard classification methods fail because they try to learn a decision boundary around an unbounded set.

### Our Solution: Hybrid Forensic Cascade

We developed a two-stage cascade architecture that functions as a rejection pipeline rather than a simple classifier.

#### Stage A: Structural Verification for Quantized Models

We first handled VAR-16d and Taming Transformers, which are vector quantized models. These models use a discrete codebook: they encode images into a finite set of discrete tokens, then decode them back. The key insight is that if an image was truly generated by VAR, the VAR decoder should be able to reconstruct it almost perfectly, resulting in a near-zero reconstruction error. If the image came from Stable Diffusion (which uses continuous latent spaces) or was a real photo, the reconstruction error would be high.

We isolated the pre-trained VAE (Variational Autoencoder) decoders for VAR and Taming. For each test image, we computed the reconstruction loss: L_rec = ||x - Dec(Enc(x))||_2^2, where x is the test image, Enc is the encoder, and Dec is the decoder. We calibrated a threshold tau_rec on the validation set. If the reconstruction loss was below tau_rec, we classified the image as VAR or Taming with near-certainty. This acted as a hard filter with approximately 0% false positives.

VAR (Visual Autoregressive) generates images progressively from low to high resolution. It starts with a 1x1 sketch, then adds details at 2x2, 4x4, 8x8, and so on until the final resolution. It uses a VQ-VAE (Vector Quantized Variational Autoencoder) to encode images into discrete tokens from a codebook. The autoregressive model then predicts the next token given previous tokens.

RAR (Randomized Autoregressive) generates images block by block, starting from the top-left corner. The key innovation is that during training, it randomly swaps blocks to learn better spatial relationships. This allows it to generate coherent images even when generating blocks in a non-sequential order.

Taming Transformers uses a two-stage approach: first, a CNN compresses the image into a latent space, then a transformer generates tokens in this latent space. Like VAR, it uses vector quantization with a discrete codebook.

Stable Diffusion uses a continuous diffusion process. It starts with random noise and iteratively removes noise guided by a text prompt. Unlike the quantized models, it does not use a discrete codebook.

#### Stage B: Textural Analysis for Continuous Models

For the remaining models (RAR and Stable Diffusion), which work in a continuous pixel space, we used a ConvNeXt-Base neural network. ConvNeXt is a modern CNN architecture inspired by Vision Transformers, with large 7x7 kernels that are adept at detecting global coherence artifacts.

We noticed that the provided training dataset was small and biased. For example, the VAR model might have mostly generated dogs in the training set. To prevent our model from learning these semantic biases instead of the actual generative artifacts, we used the provided generator scripts (VAR.py and RAR.py) to create over 5,000 new images, cycling through random ImageNet classes. This forced the model to learn the textural and structural fingerprints of the generators rather than the content of the images.

We also calibrated strict entropy thresholds. If the model was not at least 92% confident in its prediction (max softmax probability < 0.92), we automatically classified the image as an outlier. This safety net was essential to keep our false positive rate close to zero, which is critical given the heavy penalty (coefficient 5) in the evaluation metric.

During training, we optimized the thresholds tau_rec and tau_conf on the validation set to maximize the hackathon metric Score = TPR / (1 + 5 * FPR). This metric encourages high detection rates while enforcing very low false positives.

## Task 3: Watermark Detection

### Task Description

The goal was to detect invisible watermarks embedded in images. The specific watermarks were Tree-Rings, BitMark, and Stable Signature. The evaluation metric was TPR at 1% FPR (True Positive Rate at 1% False Positive Rate). This means we needed to achieve a high detection rate while keeping false alarms under 1%, which is extremely challenging.

Each submission had to output a continuous confidence score between 0 and 1 for every test image, indicating how likely it is that the image is watermarked. Higher scores indicate higher likelihood of being watermarked.

### Our Solution: Dual-Stream Forensic Network

We designed a two-branch neural network that analyzes images in two different domains simultaneously to capture both spatial and frequency anomalies.

#### Stream A: Frequency Domain (The Spectral Hunter)

The first stream focused on the frequency domain to detect Tree-Rings watermarks. Tree-Rings embeds information by altering the frequency spectrum of the image in a circular pattern. The watermark is invisible in the spatial domain but creates concentric circles in the Fourier spectrum.

We applied a Fast Fourier Transform (FFT) to the images and converted the spectrum to polar coordinates. This transformation turned the concentric circular artifacts into vertical lines, which are much easier for a convolutional neural network to detect. We used EfficientNet-V2-S as the backbone for this stream.

To train this branch effectively, we created a synthetic data augmentation routine that injected fake spectral rings (pseudo-Tree-Rings) into clean images. We computed the FFT, added energy spikes at specific frequency radii, then applied the inverse FFT. This taught the model to be hypersensitive to these specific frequency anomalies, even though we did not have access to the exact Tree-Rings algorithm.

#### Stream B: Spatial Domain (The Noise Hunter)

The second stream focused on the spatial domain to detect BitMark and Stable Signature watermarks. These watermarks hide information in the high-frequency noise of the image pixels. The challenge is that the visible image content (the object in the photo) dominates the signal, making the watermark noise difficult to detect.

We used Spatial Rich Models (SRM) filters, which are specialized, non-learnable high-pass filters used in steganalysis. These filters suppress the image content and expose the underlying noise residual. We applied three SRM filters: KB (3rd order), KV (edge detection), and Spam (square 5x5). The filtered residuals were then fed to an EfficientNet-V2-S backbone.

This allowed the network to focus purely on the invisible noise patterns where the watermark was hidden, without being distracted by the visible image content.

#### Fusion and Inference

The two streams were fused using a fully connected layer that combined the features from both branches. The final output was a single score between 0 and 1.

During inference, we used test-time augmentation (TTA). For every test image, we averaged the prediction scores from three views: the original center crop, a horizontally flipped version, and a vertically flipped version. This stabilized the predictions and further reduced the risk of false positives.

The model was trained with early stopping based on the validation TPR at 1% FPR metric. We used gradient accumulation to simulate a larger batch size (effective batch size = 16 * 2 = 32) since the images were large (512x512) and the dual-stream architecture was memory-intensive.

## Repository Structure

```text
.
├── task_1_adversarial_examples/
│   ├── attempt1/              # Initial experiments with FGSM/PGD
│   ├── attempt2/               # Ensemble PGD with fixed epsilon
│   └── attempt3/               # Final solution: Lock and Ram strategy
│       ├── run_lock_and_ram_v4.sh    # Master script for SLURM cluster
│       ├── lock_and_ram_v4.py        # Main adaptive solver logic
│       ├── attack.py                  # BS-PGD implementation
│       ├── models.py                  # Hybrid Ensemble definitions
│       ├── main_solver.py             # Phase 1 local solver
│       ├── analyze.py                 # Analysis utilities
│       └── submit.py                  # Submission utilities
│
├── task_2_image_attribution/
│   ├── main_attribution.py     # Training script (Cascade architecture)
│   ├── submit.py               # Submission logic
│   ├── VAR.py                  # VAR image generation script
│   └── RAR.py                  # RAR image generation script
│
└── task_3_watermark_detection/
    ├── solution.py             # Dual-Stream training and inference
    └── ...
```

## Key Technical Concepts

**Epsilon (ε):** The maximum allowed L2 perturbation distance. It defines the radius of the ball within which we search for adversarial examples. In our implementation, epsilon is adaptive per image based on previous API feedback.

**Kappa (κ):** The confidence margin requirement. It represents the difference between the logit of the wrong class and the logit of the true class: κ = logit_max_wrong - logit_true. A high kappa means we require the model to be very confident in the wrong prediction, which increases the likelihood of transfer to the black-box model.

**L2 Distance:** The Euclidean distance between the original and adversarial images, computed as the square root of the sum of squared pixel differences. The normalized L2 distance is divided by sqrt(C*H*W) where C, H, W are the number of channels, height, and width.

**PGD (Projected Gradient Descent):** An iterative adversarial attack method that projects perturbations onto an L2 ball of radius epsilon. At each step, it computes the gradient of the loss with respect to the input, takes a step in the direction that increases the loss, then projects back onto the epsilon ball.

**BS-PGD (Binary Search PGD):** A variant of PGD that performs binary search over epsilon values to find the minimal perturbation that achieves misclassification. Unlike standard PGD which maximizes loss within a fixed epsilon, BS-PGD minimizes epsilon subject to misclassification.

**Transfer Attack:** An attack generated on a local surrogate model that transfers to an unknown target model. This is necessary in black-box settings where we cannot compute gradients directly on the target.

**Vector Quantization:** A technique used by VAR and Taming where images are encoded into discrete tokens from a finite codebook. This allows perfect reconstruction of images generated by the same model.

**Reconstruction Loss:** The L2 distance between an original image and its reconstruction by a VAE encoder-decoder pair. Low reconstruction loss indicates the image was likely generated by that specific VAE.

**SRM Filters (Spatial Rich Models):** Fixed, non-learnable high-pass filters used in steganalysis to detect high-frequency noise residuals. They suppress image content and expose underlying noise patterns.

**Polar FFT:** A transformation that converts circular patterns in the Fourier spectrum into vertical lines by converting to polar coordinates. This makes circular watermarks easier to detect with CNNs.

**Test-Time Augmentation (TTA):** Averaging predictions over multiple augmented versions of a test image (e.g., flips, crops) to improve robustness and reduce false positives.

Developed by Team Loki for the CISPA Trustworthy AI Hackathon 2024.
