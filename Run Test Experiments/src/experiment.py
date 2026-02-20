import pandas as pd
import re
import os
from openai import OpenAI
from dotenv import load_dotenv


class SentimentExperiment:
    """Sentiment classification experiment with different prompt conditions."""

    def __init__(self, config):
        load_dotenv()
        self.config = config
        self.models = config.get("models", ["gpt-3.5-turbo"])
        self.temperatures = config.get("temperatures", [0])
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
                    print(f"Running: model={model}, temp={temp}, condition={name}")

                    response = self._call_api(prompt, model, temp)
                    if response is None:
                        print(f"  Skipped (API error)")
                        continue

                    predictions = self._parse_response(response)
                    n = len(predictions)
                    if n == 0:
                        continue
                    correct = sum(p == t for p, t in zip(predictions, self.ground_truth))

                    self.results[key] = {
                        "predictions": predictions,
                        "correct": correct,
                        "total": n,
                        "accuracy": round(correct / n * 100, 1)
                    }
        return self
    
    def save_results(self, output_dir):
        """Save accuracy summary and detailed predictions to CSV."""
        os.makedirs(output_dir, exist_ok=True)

        # Accuracy summary
        summary = [
            {"Model": model, "Temperature": temp, "Condition": cond,
             "Correct": data["correct"], "Total": data["total"], "Accuracy": data["accuracy"]}
            for (model, temp, cond), data in self.results.items()
        ]
        pd.DataFrame(summary).to_csv(f"{output_dir}/accuracy_results.csv", index=False)

        # Detailed predictions
        detailed = [
            {"Model": model, "Temperature": temp, "Condition": cond,
             "Tweet": i+1, "True": self.ground_truth[i],
             "Predicted": data["predictions"][i],
             "Correct": data["predictions"][i] == self.ground_truth[i]}
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
        # First split by newlines, then handle merged lines (e.g. "3. Neutral 4. Positive")
        lines = [line for line in response.split('\n') if line.strip()]
        labels = []
        for line in lines:
            # Split line on numbered patterns to catch merged entries
            parts = re.split(r'(?<=\s)\d+[\.\:\)]\s*', line)
            for part in parts:
                cleaned = re.sub(r'^\d+[\.\:\)]\s*', '', part).strip().lower()
                if cleaned:
                    labels.append(cleaned)
        return labels[:self.test_size]
    
    @staticmethod
    def _cat_to_label(cat):
        return {1: "positive", -1: "negative"}.get(cat, "neutral")
