import pandas as pd
import re
import os
import time
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv


class SentimentExperiment:
    """Sentiment classification experiment with different prompt conditions."""

    def __init__(self, config):
        load_dotenv()
        self.config = config
        self.models = config.get("models", ["gpt-3.5-turbo"])
        self.temperatures = config.get("temperatures", [0])
        self.num_runs = config.get("num_runs", 1)
        self.test_size = config.get("test_size", 50)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.results = {}
    
    def load_data(self, filepath):
        """Load and split data into test set and example pool."""
        df = pd.read_csv(filepath)
        self.test_df = df.head(self.test_size)
        self.example_pool = df.iloc[self.test_size:]
        self.ground_truth = [self._cat_to_label(c) for c in self.test_df['category']]
        return self
    
    def select_examples(self):
        """Select balanced examples for few-shot and many-shot."""
        few = self.config.get("few_shot", {"pos": 1, "neg": 1, "neu": 1})
        many = self.config.get("many_shot", {"pos": 3, "neg": 3, "neu": 4})
        
        self.few_examples = self._get_examples(few["pos"], few["neg"], few["neu"])
        self.many_examples = self._get_examples(many["pos"], many["neg"], many["neu"])
        return self
    
    def run(self):
        """Run all combinations of model x temperature x prompt condition."""
        conditions = {
            "zero_shot": self._prompt_zero_shot(),
            "definitions": self._prompt_definitions(),
            "few_shot": self._prompt_with_examples(self.few_examples),
            "many_shot": self._prompt_with_examples(self.many_examples)
        }

        for model in self.models:
            for temp in self.temperatures:
                for name, prompt in conditions.items():
                    key = (model, temp, name)
                    all_run_predictions = []

                    for run_i in range(self.num_runs):
                        print(f"Running: model={model}, temp={temp}, condition={name}, run={run_i+1}/{self.num_runs}")

                        response = self._call_api(prompt, model, temp)
                        if response is None:
                            print(f"  Skipped (API error)")
                            continue
                        time.sleep(1)

                        predictions = self._parse_response(response)
                        if len(predictions) > 0:
                            all_run_predictions.append(predictions)

                    if not all_run_predictions:
                        continue

                    # Use first run as the "primary" predictions for accuracy
                    primary = all_run_predictions[0]
                    n = len(primary)
                    correct = sum(p == t for p, t in zip(primary, self.ground_truth))

                    # Compute per-tweet consistency across runs
                    consistency_scores = []
                    k = len(all_run_predictions)
                    for i in range(n):
                        labels = [run[i] for run in all_run_predictions if i < len(run)]
                        if labels:
                            most_common_count = Counter(labels).most_common(1)[0][1]
                            consistency_scores.append(most_common_count / len(labels))

                    avg_consistency = round(sum(consistency_scores) / len(consistency_scores), 3) if consistency_scores else 0.0

                    # Average accuracy across all runs
                    run_accuracies = []
                    for run_preds in all_run_predictions:
                        rn = len(run_preds)
                        rc = sum(p == t for p, t in zip(run_preds, self.ground_truth))
                        run_accuracies.append(round(rc / rn * 100, 1))

                    self.results[key] = {
                        "predictions": primary,
                        "all_run_predictions": all_run_predictions,
                        "correct": correct,
                        "total": n,
                        "accuracy": round(correct / n * 100, 1),
                        "avg_accuracy": round(sum(run_accuracies) / len(run_accuracies), 1),
                        "consistency": avg_consistency,
                        "num_runs": k,
                        "per_tweet_consistency": consistency_scores
                    }
        return self
    
    def save_results(self, output_dir):
        """Save accuracy summary and detailed predictions to CSV."""
        os.makedirs(output_dir, exist_ok=True)

        # Accuracy summary
        summary = [
            {"Model": model, "Temperature": temp, "Condition": cond,
             "Correct": data["correct"], "Total": data["total"],
             "Accuracy": data["accuracy"], "Avg_Accuracy": data["avg_accuracy"],
             "Consistency": data["consistency"], "Num_Runs": data["num_runs"]}
            for (model, temp, cond), data in self.results.items()
        ]
        pd.DataFrame(summary).to_csv(f"{output_dir}/accuracy_results.csv", index=False)

        # Detailed predictions
        detailed = [
            {"Model": model, "Temperature": temp, "Condition": cond,
             "Tweet": i+1, "True": self.ground_truth[i],
             "Predicted": data["predictions"][i],
             "Correct": data["predictions"][i] == self.ground_truth[i],
             "Consistency": data["per_tweet_consistency"][i] if i < len(data["per_tweet_consistency"]) else None}
            for (model, temp, cond), data in self.results.items()
            for i in range(len(data["predictions"]))
        ]
        pd.DataFrame(detailed).to_csv(f"{output_dir}/detailed_predictions.csv", index=False)
        return self
    
    # ---- Private helpers ----
    
    def _get_examples(self, n_pos, n_neg, n_neu):
        pool = self.example_pool
        return pd.concat([
            pool[pool['category'] == 1].head(n_pos),
            pool[pool['category'] == -1].head(n_neg),
            pool[pool['category'] == 0].head(n_neu)
        ])
    
    def _tweet_list(self):
        return "\n".join(f"{i+1}. {row['clean_text']}" 
                         for i, (_, row) in enumerate(self.test_df.iterrows()))
    
    def _prompt_zero_shot(self):
        return f"""Classify each tweet as positive, neutral, or negative. One label per line.

Tweets:
{self._tweet_list()}

Output:"""
    
    def _prompt_definitions(self):
        defs = self.config.get("definitions", {
            "positive": "support, praise, favorable views",
            "negative": "criticism, disapproval, unfavorable views",
            "neutral": "factual, no clear sentiment"
        })
        return f"""Classify each tweet as positive, neutral, or negative. One label per line.

Definitions:
- Positive: {defs['positive']}
- Negative: {defs['negative']}
- Neutral: {defs['neutral']}

Tweets:
{self._tweet_list()}

Output:"""
    
    def _prompt_with_examples(self, examples):
        ex_text = "\n".join(f"Tweet: {row['clean_text']}\nSentiment: {self._cat_to_label(row['category'])}\n"
                           for _, row in examples.iterrows())
        return f"""Classify each tweet as positive, neutral, or negative. One label per line.

Examples:
{ex_text}
Tweets:
{self._tweet_list()}

Output:"""
    
    def _call_api(self, prompt, model, temperature):
        try:
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
            }
            if model in ("gpt-5", "o1", "o3", "o4-mini"):
                params["max_completion_tokens"] = 250
            else:
                params["max_tokens"] = 250
            resp = self.client.chat.completions.create(**params)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"  API error: {e}")
            return None
    
    def _parse_response(self, response):
        labels = []
        for line in response.split('\n'):
            if not line.strip():
                continue
            text = re.sub(r'^\d+[\.\:\)]\s*', '', line).strip().lower()
            if 'positive' in text:
                labels.append('positive')
            elif 'negative' in text:
                labels.append('negative')
            elif 'neutral' in text:
                labels.append('neutral')
        return labels[:self.test_size]
    
    @staticmethod
    def _cat_to_label(cat):
        return {1: "positive", -1: "negative"}.get(cat, "neutral")
