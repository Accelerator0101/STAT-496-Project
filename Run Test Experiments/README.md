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

In this experiment, we designed prompts for a sentiment classification task using real-world Twitter text data.
The task was to classify each tweet into one of three categories: positive, negative, or neutral.

We evaluated three prompt conditions that differ only in the number of in-context examples provided, while keeping the task description, model, and generation parameters fixed.

0-shot Prompt

The model was given only the task instruction and the input text, with no examples:

Classify the sentiment of the following tweet as Positive, Negative, or Neutral.

Tweet: <tweet text>

Sentiment:

Few-shot Prompt

The prompt included three examples (one per sentiment class) before the test tweet:

Classify the sentiment of the following tweet as Positive, Negative, or Neutral.

Example 1:
Tweet: <positive example>
Sentiment: Positive

Example 2:
Tweet: <negative example>
Sentiment: Negative

Example 3:
Tweet: <neutral example>
Sentiment: Neutral

Tweet: <tweet text>

Sentiment:

Many-shot Prompt

The prompt included multiple examples per class (3 positive, 3 negative, and 4 neutral examples) before the test tweet, increasing the total number of in-context examples:

Classify the sentiment of the following tweet as Positive, Negative, or Neutral.

(Multiple labeled examples provided here)

Tweet: <tweet text>

Sentiment:

All prompts were run using the same model (gpt-3.5-turbo) and the same generation settings, with the only experimental variable being the number of in-context examples 

main

.

## What kind of responses were received

The model responses consisted of single sentiment labels (Positive, Negative, or Neutral) for each input tweet.

General observations

The outputs were short and structured, typically returning only the predicted sentiment label.

No additional explanations or reasoning text was produced, which simplified comparison across runs.

Across repeated runs, outputs sometimes varied, particularly in borderline or ambiguous tweets.

Differences across prompt conditions

0-shot prompts showed the greatest variability across runs, with the model occasionally assigning different sentiment labels to the same tweet.

Few-shot prompts produced more consistent outputs, with fewer changes in predicted labels across repeated runs.

Many-shot prompts showed the highest stability, with predictions remaining largely consistent across runs and a clearer separation between sentiment classes.

Overall, increasing the number of in-context examples appeared to reduce output variability and increase consistency, suggesting that in-prompt examples help constrain the model’s behavior even when model parameters are fixed.

## How might improve experiment?
<br/><br/>We will use a larger and more balanced dataset so that the results are more reliable. I would also test more than one language model to see whether my findings apply more generally. We will repeat some trials when using nonzero temperature to measure variability. Finally,We will standardize output formats and analyze common mistakes to better understand where the model fails.<br/><br/>What variables do you intend to vary?<br/><br/>We plan to vary several factors, like the model being used, and the temperature setting. We will also test small changes to the input text, such as paraphrasing or adding typos, to examine how robust the model is.<br/><br/>How will you expand on your starting experiment?<br/><br/>We will begin with a simple baseline experiment and then gradually add more conditions, such as few-shot prompts and input variations. After that, We can test multiple models to see if the results remain consistent.<br/><br/>How might you automate largescale data collection?<br/><br/>We will build an automated pipeline using APIs to generate prompts, send requests to the model, and store the results. This system will also handle data cleaning, evaluation, and cost tracking. By automating these steps, it efficiently scale the experiment to larger datasets while keeping expenses under control.
No file chosen
