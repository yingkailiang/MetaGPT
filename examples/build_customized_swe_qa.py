"""
Filename: MetaGPT/examples/build_customized_swe_qa.py
build an example for simple SWE and QA feedback loop
Created Date: 02/25/2024 
Author: kai 
"""
import re

import fire
import subprocess

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team


################## SWE ########################### 

def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text


class SimpleWriteCode(Action):
    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction}.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """
    name: str = "SimpleWriteCode"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        code_text = parse_code(rsp)
        # Open the file in write mode (overwrites existing content)
        with open("test/test.py", "w") as file:
            # Write content to the file
            file.write(code_text)

        return code_text


class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement, SimpleQA])
        self.set_actions([SimpleWriteCode])

################## Unit test Engineer ########################### 

class SimpleWriteTest(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Write {k} unit tests using pytest for the given function, assuming you have imported it.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    name: str = "SimpleWriteTest"

    async def run(self, context: str, k: int = 3):
        prompt = self.PROMPT_TEMPLATE.format(context=context, k=k)

        rsp = await self._aask(prompt)

        code_text = parse_code(rsp)
        # Open the file in write mode (overwrites existing content)
        with open("test/test.py", "a") as file:
            # Write content to the file
            file.write(code_text)

        return code_text


class SimpleTester(Role):
    name: str = "Bob"
    profile: str = "SimpleTester"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteTest])
        # self._watch([SimpleWriteCode])
        self._watch([SimpleWriteCode, SimpleWriteReview])  # feel free to try this too

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        context = self.get_memories(k=1)[0].content # use the most recent memory as context
        # context = self.get_memories()  # use all memories as context
        logger.info(f"!!!!!!!!!!!!!!!!SimpleWriteTest context: {context}")

        code_text = await todo.run(context, k=5)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg

################## QA Engineer ########################### 

class SimpleQA(Action):
    PROMPT_TEMPLATE: str = """
    Role: You are a senior development and qa engineer, your role is summarize the pytest running result.
    If the pytest running result does not include failure or error, you should explicitly approve the result.
    On the other hand, if the test result indicates failure case, you should point out which part, the development code or the test code, produces the error,
    and give specific instructions on fixing the failure or error. Here is the unit test result:
    {context}
    Now you should begin your analysis
    ---
    ## instruction:
    Please summarize the cause of the errors and give correction instruction
    ## Status:
    Determine if all of the code works fine, if so write PASS, else FAIL,
    WRITE ONLY ONE WORD, PASS OR FAIL, IN THIS SECTION
    ## Send To:
    Please write NoOne if there are no errors, Engineer if the errors are due to problematic development codes, else QaEngineer,
    WRITE ONLY ONE WORD, NoOne OR Engineer OR QaEngineer, IN THIS SECTION.
    ---
    You should fill in necessary instruction, status, send to, and finally return all content between the --- segment line.
    """

    name: str = "SimpleWriteTest"

    async def run(self, context: str):
        logger.info(f"SimpleQA feed LLM: {context}")
        prompt = self.PROMPT_TEMPLATE.format(context=context)
        rsp = await self._aask(prompt)
        return rsp 

class SimpleRunCode(Action):
    name: str = "SimpleRunCode"

    async def run(self, code_text: str):
        result = subprocess.run(["pytest", "test/test.py"], capture_output=True, text=True)
        logger.info(f"pytest result: {result.stdout}")
        return result.stdout 

class SimpleQAEngineer(Role):
    name: str = "Ryan"
    profile: str = "SimpleQAEngineer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleRunCode, SimpleQA])
        self._set_react_mode(react_mode="by_order")
        self._watch([SimpleWriteReview])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        context = self.get_memories(k=1)[0].content # use the most recent memory as context
        # context = self.get_memories()  # use all memories as context

        code_text = await todo.run(context)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))
        self.rc.memory.add(msg)
        return msg

################## Unit test reviewer ########################### 

class SimpleWriteReview(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Review the test cases and provide one critical comments:
    """

    name: str = "SimpleWriteReview"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        rsp = await self._aask(prompt)

        return rsp


class SimpleReviewer(Role):
    name: str = "Charlie"
    profile: str = "SimpleReviewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteReview])
        self._watch([SimpleWriteTest])


async def main(
    idea: str = "write a function that calculates the product of a list. The code should be used in the main function.",
    investment: float = 3.0,
    n_round: int = 5,
    add_human: bool = False,
):
    logger.info(idea)

    team = Team()
    team.hire(
        [
            SimpleCoder(),
            SimpleQAEngineer(),
            SimpleTester(),
            SimpleReviewer(),
        ]
    )

    team.invest(investment=investment)
    team.run_project(idea)
    await team.run(n_round=n_round)


if __name__ == "__main__":
    fire.Fire(main)