"""
================================================================================
 use_saved_model.py  --  Load the trained model and play with it
================================================================================

This shows how to USE a model after it has been trained and saved.

Key idea for a beginner:
  * The PICKLE FILE is just *data* -- all the learned numbers, packed into a
    plain Python dictionary. It contains no code.
  * The FUNCTIONS that run the model (`forward`, `generate`) live in
    `simple_transformer.py`. We import them here, the same way you would
    import any helper module.

Run it with:   python use_saved_model.py
================================================================================
"""

import os
from simple_transformer import load_model, generate, forward, softmax

# ------------------------------------------------------------------------------
# 1. Load the model that simple_transformer.py trained and saved.
# ------------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "saved_models", "transformer.pkl")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("No saved model found. Run 'python simple_transformer.py' first.")

model = load_model(MODEL_PATH)
print("Loaded model from:", MODEL_PATH)
print("It knows these characters:", model["itos"])
print()

# ------------------------------------------------------------------------------
# 2. Ask the model to continue some text.
# ------------------------------------------------------------------------------
print("--- Text generation -------------------------------------------")
for prompt in ["hello", "hello world h", "world h"]:
    result = generate(model, prompt, 30)
    print(f"  start {prompt!r:18s} -> {result!r}")
print()

# ------------------------------------------------------------------------------
# 3. Peek inside: what does the model think comes next, with probabilities?
# ------------------------------------------------------------------------------
print("--- Looking inside the model's 'mind' -------------------------")
context_text = "hello"
context = [model["stoi"][ch] for ch in context_text]
logits, _ = forward(model, context)

# logits[-1] is the model's opinion about the character AFTER the context.
probs = softmax(logits[-1])
ranked = sorted(range(len(probs)), key=lambda c: probs[c], reverse=True)

print(f"  After {context_text!r}, the model predicts the next character:")
for c in ranked:
    ch = model["itos"][c]
    bar = "#" * int(probs[c] * 40)
    print(f"    {ch!r:5s} {probs[c]*100:5.1f}%  {bar}")
print()
print("Notice it is very confident about the right answer -- that confidence")
print("is what 'training' built up, one tiny nudge at a time.")
