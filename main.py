from src.experiment import SentimentExperiment

config = {
    "model": "gpt-3.5-turbo",
    "test_size": 30,
    "few_shot": {"pos": 1, "neg": 1, "neu": 1},
    "many_shot": {"pos": 3, "neg": 3, "neu": 4}
}

if __name__ == "__main__":
    exp = SentimentExperiment(config)
    exp.load_data("data/input/Twitter_Data.csv")
    exp.select_examples()
    exp.run()
    exp.save_results("data/results/")
    
    # Print summary
    print("\n=== Results ===")
    for cond, data in exp.results.items():
        print(f"{cond}: {data['correct']}/{data['total']} = {data['accuracy']}%")
