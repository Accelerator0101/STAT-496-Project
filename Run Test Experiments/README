# STAT 496 Capstone: Prompt Engineering Experiment

Testing the effect of examples in prompts on sentiment classification accuracy.

## Experiment Design

**Task**: Classify Twitter sentiment as positive, neutral, or negative

**Conditions**:
1. Zero-shot — No examples
2. Zero-shot + Definitions — Explain what pos/neg/neu means
3. Few-shot (3) — 1 positive, 1 negative, 1 neutral example
4. Many-shot (10) — 3 positive, 3 negative, 4 neutral examples

**Model**: GPT-3.5-Turbo

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

## Run

```bash
python3 main.py
```

## Configuration

Edit `main.py` to change:
- `test_size` — Number of tweets to classify
- `few_shot` / `many_shot` — Number of examples per category

## Output

Results saved to `data/results/`:
- `accuracy_results.csv` — Accuracy summary per condition
- `detailed_predictions.csv` — Per-tweet predictions
