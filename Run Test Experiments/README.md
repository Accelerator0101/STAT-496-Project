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










## How might you improve your experiment?
<br/><br/>We will use a larger and more balanced dataset so that the results are more reliable. I would also test more than one language model to see whether my findings apply more generally. We will repeat some trials when using nonzero temperature to measure variability. Finally,We will standardize output formats and analyze common mistakes to better understand where the model fails.<br/><br/>What variables do you intend to vary?<br/><br/>We plan to vary several factors, like the model being used, and the temperature setting. We will also test small changes to the input text, such as paraphrasing or adding typos, to examine how robust the model is.<br/><br/>How will you expand on your starting experiment?<br/><br/>We will begin with a simple baseline experiment and then gradually add more conditions, such as few-shot prompts and input variations. After that, We can test multiple models to see if the results remain consistent.<br/><br/>How might you automate largescale data collection?<br/><br/>We will build an automated pipeline using APIs to generate prompts, send requests to the model, and store the results. This system will also handle data cleaning, evaluation, and cost tracking. By automating these steps, it efficiently scale the experiment to larger datasets while keeping expenses under control.
No file chosen
