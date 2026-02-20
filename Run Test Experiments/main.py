from src.experiment import SentimentExperiment

config = {
    "models": ["gpt-3.5-turbo", "gpt-4.1-mini", "gpt-4.1"],
    "temperatures": [0, 0.5, 1.0],
    "test_size": 50,
    "few_shot": {"pos": 1, "neg": 1, "neu": 1},
    "many_shot": {"pos": 3, "neg": 3, "neu": 4}
}

if __name__ == "__main__":
    exp = SentimentExperiment(config)
    exp.load_data("Input/Twitter_Data.csv")
    exp.select_examples()
    exp.run()
    exp.save_results("Results/")

    # Print summary
    print("\n=== Results ===")
    for (model, temp, cond), data in exp.results.items():
        print(f"[{model} | temp={temp}] {cond}: {data['correct']}/{data['total']} = {data['accuracy']}%")
