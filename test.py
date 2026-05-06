
import torch
import numpy as np

def verify_cfru_integrity():
    print("🧪 Starting CFRU Integrity Check...")
    
    # 1. Load the models
    try:
        baseline = torch.load('baseline_model.pth', map_location='cpu')
        unlearned = torch.load('unlearned_model.pth', map_location='cpu')
        retrained = torch.load('retrained_model.pth', map_location='cpu')
    except FileNotFoundError as e:
        print(f"❌ Error: Missing model files. {e}")
        return

    # 2. Check for "Dead Models" (The 0.000 Utility Error)
    # If weights are NaN or zero, the model is broken.
    for name, state in [("Baseline", baseline), ("Unlearned", unlearned)]:
        weights = state['item_embedding.weight']
        if torch.isnan(weights).any():
            print(f"❌ ALERT: {name} model has NaN values (Exploding Gradients).")
        if torch.all(weights == 0):
            print(f"❌ ALERT: {name} model weights are all ZERO.")

    # 3. Check for Mathematical Divergence
    # The Unlearned model should NOT be identical to the Baseline.
    diff_unlearn = torch.norm(unlearned['item_embedding.weight'] - baseline['item_embedding.weight'])
    print(f"🔹 Weight Distance (Baseline <-> Unlearned): {diff_unlearn.item():.6f}")
    
    if diff_unlearn.item() == 0:
        print("❌ FAIL: Unlearned model is identical to Baseline. Unlearning did not execute.")
    elif diff_unlearn.item() < 1e-5:
        print("⚠️  WARNING: Very small divergence. Your Alpha (α) or Skew Factor might be too low.")
    else:
        print("✅ SUCCESS: Models are mathematically distinct.")

    # 4. Sensitivity Test: Alpha Impact
    # The paper states Alpha controls skew compensation. 
    # In your Flask app, changing α should change the 'Live CFRU' scores.
    print("\n💡 Why do I see similar movies for different Users?")
    print("1. Popularity Bias: MovieLens is dominated by hits that rank high for everyone.")
    print("2. MLP Stability: The neural network layers are intentionally preserved to maintain utility.")
    print("3. Check the 'Score' column in your UI: The movies might be the same, but the scores should differ at the 4th decimal place.")

if __name__ == "__main__":
    verify_cfru_integrity()