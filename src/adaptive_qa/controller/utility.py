"""Configurable costs. Reporting includes sunk costs; choices compare future costs."""
class CostModel:
    def __init__(self, config, rewards):
        self.config, self.rewards = config, rewards

    def dollars(self, usage):
        ip, op = self.config["input_usd_per_million"], self.config["output_usd_per_million"]
        if ip is None or op is None:
            return None
        return (usage.input_tokens * ip + usage.output_tokens * op) / 1_000_000

    def cost(self, usage):
        result = sum(getattr(usage, field) * self.config[field + "_weight"] for field in
                     ("input_tokens", "output_tokens", "llm_calls", "retrieval_calls", "latency_seconds"))
        if self.config["dollars_weight"]:
            dollars = self.dollars(usage)
            if dollars is None:
                raise ValueError("Missing prices")
            result += dollars * self.config["dollars_weight"]
        return float(result)

    def reward(self, correct=None, answered=True):
        return self.rewards["abstain"] if not answered else self.rewards["correct" if correct else "incorrect"]

    def utility(self, correct, answered, usage, lambda_):
        return self.reward(correct, answered) - lambda_ * self.cost(usage)
