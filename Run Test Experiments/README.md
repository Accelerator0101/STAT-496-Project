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


## What prompts were used

We used prompts designed for a **sentiment classification task** on real-world Twitter text data.  
Each prompt asked the model to classify a tweet as **Positive**, **Negative**, or **Neutral**.

Three prompt conditions were tested, differing **only in the number of in-context examples provided**:

- **0-shot:** No examples were included; only the task instruction and the input tweet were provided.
- **Few-shot:** Three labeled examples (one per sentiment class) were included before the input tweet.
- **Many-shot:** Multiple labeled examples per class (3 positive, 3 negative, and 4 neutral examples) were included before the input tweet.

All prompts shared the same task description and output format, with the number of in-context examples as the only varying factor.


## What kind of responses were received

The model responses consisted of **single sentiment labels**: `Positive`, `Negative`, or `Neutral`.

Across repeated runs:
- Outputs were short and consistently formatted.
- The model typically returned only a sentiment label without additional explanation.
- **0-shot prompts** showed the highest variability across runs.
- **Few-shot prompts** produced more consistent outputs.
- **Many-shot prompts** yielded the most stable and reproducible predictions.

## How might improve experiment?
<br/><br/>We will use a larger and more balanced dataset so that the results are more reliable. I would also test more than one language model to see whether my findings apply more generally. We will repeat some trials when using nonzero temperature to measure variability. Finally,We will standardize output formats and analyze common mistakes to better understand where the model fails.<br/><br/>What variables do you intend to vary?<br/><br/>We plan to vary several factors, like the model being used, and the temperature setting. We will also test small changes to the input text, such as paraphrasing or adding typos, to examine how robust the model is.<br/><br/>How will you expand on your starting experiment?<br/><br/>We will begin with a simple baseline experiment and then gradually add more conditions, such as few-shot prompts and input variations. After that, We can test multiple models to see if the results remain consistent.<br/><br/>How might you automate largescale data collection?<br/><br/>We will build an automated pipeline using APIs to generate prompts, send requests to the model, and store the results. This system will also handle data cleaning, evaluation, and cost tracking. By automating these steps, it efficiently scale the experiment to larger datasets while keeping expenses under control.
No file chosen
