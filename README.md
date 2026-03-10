# STAT 496 Capstone: Prompt Engineering Experiment

Testing the effect of prompt structure (in-context examples) and temperature on LLM sentiment classification accuracy and output consistency.

## Experiment Design

**Task**: Classify Twitter sentiment as positive, neutral, or negative (100 tweets per run)

**Models**:
- GPT-3.5-Turbo
- GPT-4.1-mini
- GPT-4.1

**Prompt Conditions**:
1. Zero-shot — No examples
2. Zero-shot + Definitions — Explain what positive/negative/neutral means
3. Few-shot (3) — 1 positive, 1 negative, 1 neutral example
4. Many-shot (10) — 3 positive, 3 negative, 4 neutral examples

**Temperature Settings**: 0, 0.5, 1.0, 2.0

**Runs per Configuration**: 3 (to measure consistency)

**Total API Calls**: 3 models x 4 temperatures x 4 conditions x 3 runs = 144

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
- `models` — List of OpenAI models to test
- `temperatures` — List of temperature values
- `num_runs` — Number of runs per configuration (for consistency measurement)
- `test_size` — Number of tweets to classify (default: 100)
- `few_shot` / `many_shot` — Number of examples per category

## Output

Results saved to `Results/`:
- `accuracy_results.csv` — Accuracy summary per condition with consistency scores
- `detailed_predictions.csv` — Per-tweet predictions with per-tweet consistency

### Output Columns

| Column | Description |
|--------|-------------|
| Model | The LLM used |
| Temperature | Sampling temperature (0 = deterministic, 2 = max randomness) |
| Condition | Prompt type (zero_shot, definitions, few_shot, many_shot) |
| Correct | Number of correct predictions (first run) |
| Total | Total tweets classified |
| Accuracy | Accuracy of the first run (%) |
| Avg_Accuracy | Average accuracy across all runs (%) |
| Consistency | Average majority agreement rate across tweets (1.0 = perfectly stable) |
| Num_Runs | Number of successful runs |

## Consistency Metric

For each tweet, consistency is calculated as:

**Consistency = (count of most frequent label across K runs) / K**

Then averaged across all tweets. A score of 1.0 means every run produced the same label for every tweet. A score of 0.33 would mean completely random outputs.

## What Prompts Were Used

Each prompt asked the model to classify tweets as **Positive**, **Negative**, or **Neutral**. Four prompt conditions were tested, differing only in the amount of context provided:

- **Zero-shot:** Only the task instruction and input tweets.
- **Definitions:** Task instruction plus explicit definitions of each sentiment category.
- **Few-shot:** Three labeled examples (one per class) before the input tweets.
- **Many-shot:** Ten labeled examples (3 positive, 3 negative, 4 neutral) before the input tweets.

All prompts shared the same base instruction and output format.

## Key Findings

- **Model capability is the dominant factor**: GPT-4.1 (92-95%) >> GPT-4.1-mini (68-84%) >> GPT-3.5-turbo (55-76%)
- **Higher temperature reduces both accuracy and consistency** across all models
- **Few-shot/many-shot prompts** provide slight accuracy gains and better consistency at higher temperatures
- **Definitions** help more than raw zero-shot, especially for weaker models
- **GPT-4.1** maintains high consistency (~0.9+) even at temperature=2.0
- **GPT-3.5-turbo at temperature=2.0** shows the most erratic behavior (consistency as low as 0.667)
