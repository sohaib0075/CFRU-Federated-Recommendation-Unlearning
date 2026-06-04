# CFRU: Certified Unlearning for Federated Recommendation

This project implements a CFRU-inspired federated recommendation unlearning pipeline for efficient and verifiable user data deletion from federated recommender systems.

## Overview

Federated recommender systems preserve user privacy by keeping user data on local devices. However, when a user requests deletion, their influence may still remain embedded in global model through historical gradients.

This project explores Certified Federated Recommendation Unlearning (CFRU), which aims to remove user influence without full retraining.

## Key Features

- Federated recommendation using Neural Collaborative Filtering
- MovieLens-1M dataset
- User-wise federated client partitioning
- Gradient rollback-based unlearning
- Skew correction mechanism
- Utility evaluation using HR@10 and NDCG@10
- Privacy evaluation using Membership Inference Attack
- Comparison with polluted baseline and retrained gold standard

## Dataset

The project uses the MovieLens-1M dataset.

Dataset statistics:
- 1M+ ratings
- 6,040 users
- 3,706 movies

Download from:
https://grouplens.org/datasets/movielens/1m/

Place the dataset files inside the `dataset/` folder.

## Tools & Libraries

- Python
- PyTorch
- Flower
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Implementation Pipeline

1. Load and preprocess MovieLens-1M.
2. Partition data user-wise for federated learning.
3. Train Neural Collaborative Filtering model using federated setup.
4. Select 5% users for unlearning.
5. Apply rollback and skew correction.
6. Inject calibrated noise.
7. Evaluate recommendation utility and privacy leakage.

## Results

### Federated Data Distribution
Shows non-IID user interaction distribution and long-tail item popularity.

### Training Loss
Federated NCF training loss decreases across 20 global rounds, showing convergence.

### Utility Preservation
CFRU preserves recommendation quality using HR@10 and NDCG@10.

### Privacy Evaluation
Membership Inference Attack is used to test whether deleted users are still detectable.

Current results show residual privacy leakage, indicating the need for stronger noise calibration.

## Results Summary

| Metric | Observation |
|---|---|
| HR@10 | CFRU preserves recommendation utility |
| NDCG@10 | Ranking quality remains competitive |
| MIA Accuracy | Still above ideal 0.50 threshold |
| Training Loss | Converges over 20 rounds |

## Limitations

- Certified privacy guarantee is not fully achieved.
- MIA accuracy remains above random guessing.
- Noise calibration needs improvement.
- Implementation is a simplified CFRU-inspired prototype.

## Future Work

- Improve certified noise calibration.
- Strengthen skew estimation.
- Test on larger federated settings.
- Add stronger privacy defense mechanisms.

## Authors

- Fasih Ur Rehman
- Sohaib Shahzad

## Reference

Paper: Certified Unlearning for Federated Recommendation  
Published in ACM Transactions on Information Systems, 2025.
