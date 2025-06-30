class Memory:
    def __init__(self):
        self.history = []

    def update(self, input_data, result):
        self.history.append({'input': input_data, 'result': result})

    def get_history(self):
        return self.history 