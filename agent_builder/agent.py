import logging
from agent_builder.tools.format_response_tool import FormatResponseTool

class Agent:
    """
    Agent class that coordinates memory, planning, and execution using unified interfaces and error handling.
    Returns both raw and formatted output for display flexibility.
    """
    def __init__(self, memory, planner, executor):
        self.memory = memory
        self.planner = planner
        self.executor = executor
        self.formatter = FormatResponseTool()

    def act(self, input_data, format_response=True, show_full_details=False):
        try:
            plan = self.planner.plan(input_data, self.memory)
            result = self.executor.execute(plan)
            self.memory.add_interaction(input_data=input_data, result=result)
            # If it's an LLM response and formatting is requested, format for display
            if format_response and plan.get('type') == 'llm_response':
                if show_full_details and plan.get('raw') is not None:
                    return self.formatter.run(plan['raw'])
                return self.formatter.run(result)
            return result
        except Exception as e:
            logging.exception(f"Agent act error for input: {input_data}")
            self.memory.add_interaction(input_data=input_data, result=f"Agent error: {str(e)}")
            return f"Agent error: {str(e)}" 