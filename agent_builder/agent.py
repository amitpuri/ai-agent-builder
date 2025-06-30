class Agent:
    def __init__(self, memory, planner, executor):
        self.memory = memory
        self.planner = planner
        self.executor = executor

    def act(self, input_data):
        plan = self.planner.plan(input_data, self.memory)
        result = self.executor.execute(plan)
        self.memory.update(input_data, result)
        return result 