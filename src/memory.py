class ConversationMemory:
    def __init__(self,max_turns=5):
        self.history = []
        self.max_turns = max_turns

    def add_turn(self,question,answer):
        self.history.append(
            {
                "question": question,
                "answer": answer
            }
        )

        self.history = self.history[-self.max_turns:]

    def get_history(self):
        return self.history

    def clear(self):
        self.history=[]