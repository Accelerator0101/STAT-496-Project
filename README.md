# STAT 496 Capstone: Prompt Engineering Experiment

This project examines how prompt structure and temperature affect large language model performance on sentiment classification. The task is to classify tweets as positive, neutral, or negative while comparing both accuracy and output consistency across repeated runs.

## Overview

Each run uses 100 tweets. The experiment tests three models: GPT-3.5-Turbo, GPT-4.1-mini, and GPT-4.1. Every model is evaluated under four prompt conditions and four temperature settings. Each configuration is repeated three times to measure consistency.

The four prompt conditions are zero-shot, definitions, few-shot, and many-shot. The temperature settings are 0, 0.5, 1.0, and 2.0.

The full experiment includes 144 API calls.

## Experiment Design

The task is Twitter sentiment classification with three labels: positive, neutral, and negative.

Zero-shot uses only the task instruction and the input tweets.

Definitions uses the task instruction together with explicit definitions of positive, negative, and neutral sentiment.

Few-shot uses three labeled examples before the input tweets, with one example from each class.

Many-shot uses ten labeled examples before the input tweets, including three positive examples, three negative examples, and four neutral examples.

Each configuration is repeated three times to evaluate how stable the model outputs remain across runs.

## Setup

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

Create a `.env` file and add your OpenAI API key:

```env
OPENAI_API_KEY=your_key_here
```

## Running the Experiment

Run the project with:

```bash
python3 main.py
```

## Configuration

You can edit `main.py` to change the experimental settings.

`models` controls which OpenAI models are tested.

`temperatures` controls the sampling temperatures.

`num_runs` sets how many times each configuration is repeated.

`test_size` sets the number of tweets classified in each run. The default is 100.

`few_shot` and `many_shot` control how many labeled examples are included in those prompt conditions.

## Output Files

Results are saved in the `Results/` directory.

`accuracy_results.csv` contains summary statistics for each configuration, including accuracy and consistency.

`detailed_predictions.csv` contains tweet-level predictions and tweet-level consistency values.

## Output Columns

| Column | Description |
|--------|-------------|
| Model | The model used for classification |
| Temperature | The sampling temperature |
| Condition | The prompt condition |
| Correct | Number of correct predictions from run 1 |
| Total | Number of tweets classified |
| Accuracy | Accuracy from run 1 as a percentage |
| Avg_Accuracy | Mean accuracy across all runs |
| Consistency | Mean majority agreement across tweets |
| Num_Runs | Number of successful runs |

## Consistency Metric

Consistency is computed at the tweet level across repeated runs.

For each tweet, consistency is defined as the number of times the most common label appears divided by the total number of runs.

If a tweet receives the same label in every run, its consistency score is 1.0. If predictions vary more across runs, the score is lower.

The final consistency value reported for a configuration is the average of these tweet-level scores across the full dataset.

## Prompt Structure

All prompts ask the model to classify tweets as positive, negative, or neutral. The base instruction and output format remain the same across all conditions. The only difference is how much context is provided before the test tweets.

Zero-shot includes no labeled examples.

Definitions adds explicit descriptions of each sentiment category.

Few-shot includes three labeled examples.

Many-shot includes ten labeled examples.

## Key Findings

Model capability is the strongest factor in performance. GPT-4.1 achieves the highest accuracy, followed by GPT-4.1-mini, then GPT-3.5-Turbo.

Higher temperature generally lowers both accuracy and consistency.

Few-shot and many-shot prompting provide small accuracy improvements and tend to improve consistency when temperature is high.

Definitions are usually more helpful than plain zero-shot prompting, especially for weaker models.

GPT-4.1 remains highly consistent even at temperature 2.0.

GPT-3.5-Turbo at temperature 2.0 produces the least stable outputs and shows the lowest consistency scores.
