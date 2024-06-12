import re
import os
# import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.bamboogle import * 

choose_evaluate = '''Evaluate whether the provided answer matches any of the options listed in the question  (Likely/Impossible).

Question: You are presented with the question "What do veins carry?" and the following answer choices:  - copper  - glucose  - Energy  - Length  - oxygen  - warmth  - Cells  - voltage  Now knowing that veins generally carry deoxygenated blood and blood carries blood cells, choose the best answer.
Answer: Blood
Evaluation Process: The listed choices include copper, glucose, energy, length, oxygen, warmth, cells, and voltage, but blood is not among these options.
Evaluation Result:
Impossible

Question: Please answer the following question: You are presented with the question "What converts chemical energy into sound?" and the following answer choices:  - a firework  - sensory neurons  - Gunshots  - a bottle  - a battery  - animals  - a flashlight  - engines  Now knowing that afirecracker converts chemical energy into sound and fireworks are illegal, including firecrackers, choose the best answer.
Answer: a firework
Evaluation Process: Among the provided options, which include sensory neurons, gunshots, a bottle, a battery, animals, a flashlight, and engines, a firework is indeed one of the choices.
Evaluation Results:
Likely

Question: You are presented with the question "A good way for older adults to strengthen bones is to " and the following answer choices:  - Through play  - drive more slowly  - exercise  - use a hearing aid  - quadriceps  - donate bone marrow  - sweating  - movement  Now knowing that exercise increases a body 's strength and strength training is also effective for increasing bone strength in older adults, choose the best answer.
Answer: exercise
Evaluation Process: Among the options listed, such as through play, driving more slowly, using a hearing aid, quadriceps, donating bone marrow, sweating, and movement, exercise is indeed included.
Evaluation Results:
Likely

Question: You are presented with the question "what is a negative impact on an organism" and the following answer choices:  - malnutrition  - plants  - hyperthyroidism  - sweat  - laughter  - smallpox  - leukemia  Now knowing that disease has a negative impact on an organism and infectious diseases and diseases of malnutrition are prevalent, choose the best answer.
Answer: cancer
Evaluation Process: The options provided include malnutrition, plants, hyperthyroidism, sweat, laughter, smallpox, and leukemia, but cancer is not listed among these choices.
Evaluation Results:
Impossible

Question: '''

wiki_evaluate = '''Evaluate whether the language model can effectively decompose the question into relevant sub-questions, and assess whether this decomposition helps in partially or directly answering the original question. The outcome will determine if this process of decomposition is "Likely" or "Impossible" to aid in finding the answer.

Evaluation Steps: Check if the language model can identify and decompose key sub-questions that are directly related to the original question.
Evaluation Process: 1. Analyze whether each sub-question identified by the model is directly relevant to the answer to the original question. 2. Determine if the decomposition of these sub-questions forms a reasonable response to the original question.
Evaluation Result: 1. Likely: If the language model successfully decomposes the original question into relevant sub-questions that help construct the final answer. 2. Impossible: If the language model fails to effectively decompose the question, or if the decomposed sub-questions are not directly relevant to finding the answer.

Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins?
Thought Process: Step 1, when did Theodor Haecker die? Theodor Haecker was 65 years old when he died. Step 2, when did  Harry Vaughan Watkins die? Harry Vaughan Watkins was 69 years old when he died.
Evaluation Process:
Relevance of Sub-Questions: The sub-question regarding Theodor Haecker's age at death is directly relevant to the main question, as it provides necessary information to determine his lifespan. Similarly, the sub-question about Harry Vaughan Watkins' age at death is also directly relevant for the same reason.
Effectiveness of Decomposition: The decomposition into two key sub-questions (ages at death of both individuals) is an effective strategy. It breaks down the main question (comparison of lifespans) into specific, answerable elements. Each sub-question contributes a crucial piece of information required to compare the lifespans of the two individuals.
Evaluation Result:
Likely

Question: When did the last king from Britain's House of Hanover die?
Thought: Step 1, when did the last king from Britain's House of Hanover born? 
Evaluation Process:
The thought process focuses on the birth date of the last king from Britain's House of Hanover. However, knowing the birth date does not directly help in determining the date of death, which is the actual question. The lifespan of an individual can vary widely and cannot be accurately inferred from their birth date alone. Therefore, this thought process is unlikely to lead to the correct answer without additional information.
So the evaluation result is: this thought is impossible to help pariticially or directly answer the question.
Evaluation Results:
Impossible

Question: What is the highest mountain in the world?
Thought Process: Step 1, identify the tallest mountains known globally. Mount Everest is commonly known as the highest mountain peak in the world.
Evaluation Process:
The thought process begins with identifying the tallest mountains known globally, which is a logical first step. Since Mount Everest is commonly known and recognized as the highest mountain peak in the world, this thought directly leads to the answer to the question. Therefore, this approach is very likely to help in answering the question correctly.
So, the evaluation result is: this thought is likely to help partially or directly answer the question.
Evaluation Results:
Likely

Question: How many planets are in our solar system?
Thought Process: Step 1, consider the composition of the Sun and its impact on the solar system.
Evaluation Process:
The thought process of considering the composition of the Sun and its impact on the solar system does not directly lead to an answer for the number of planets in our solar system. The Sun's composition and its effects are more relevant to solar physics and do not provide specific information about the count or existence of planets. The question requires knowledge about the classification and count of planets in the solar system, which is unrelated to the Sun's composition.
So, the evaluation result is: this thought is impossible to help partially or directly answer the question.
Evaluation Results:
Impossible

Question: '''
Final_evaluate = '''Evaluate if the given sentence is possible to answer the question (Likely/Impossible).

Question: Who was the President of the United States in the year that Citibank was founded?
So the final answer is: james madison.
Evaluation Process:
Yes, james madison is a person and is likely to answer a question start with 'who'.
Evaluation Results:
Likely

Question: What rocket was the first spacecraft that ever approached Uranus launched on?
So the final answer is: Voyager 2.
Evaluation Process:
Voyager 2 is not a rocket, so it can not answer a question start with 'what rokect'.
Evaluation Results:
Impossible

Question: '''
class FactualQA(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='Bamboogle Prerelease - Sheet1.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'bamboogle', file)
        self.data = list(pd.read_csv(path)['Question'])
        self.ground_truth = list(pd.read_csv(path)['Answer'])
        self.value_cache = {}
        self.steps = 3
        self.stops = ['.', '.','Question'] 

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, ground_truth: str, output: str, out):
        if 'answer is' not in output:
            print('====output====')
            print(output)
            return {'r':0}, out
        expression = output.strip().lower().split('so the final answer is')[1].lower().split('\n')[0]
        expression = expression.replace(': ', '')
        ground_truth = str(ground_truth)
        print('====GR===='+str(ground_truth) +'====Pre===='+str(expression))
        # if re.search(ground_truth, expression, re.IGNORECASE):
        if ground_truth in expression:
            return {'r': 1}, out
        else:
            expression_ = re.sub(r'\W+', '', expression, flags=re.IGNORECASE)
            ground_truth = re.sub(r'\W+', '', ground_truth, flags=re.IGNORECASE)
            if re.search(ground_truth, expression_, re.IGNORECASE):
                return {'r': 1}, out
            else:
                ground_truth = ground_truth.split(' ')
                tmp = 1
                i = 0
                flag = 0
                while tmp:
                    tmp = re.search(ground_truth[i], expression, re.IGNORECASE)
                    i += 1
                    if i == len(ground_truth):
                        if tmp:
                            flag = 1
                            break
                if flag == 1:
                    return {'r': 1}, out
                else:
                    return {'r': 0}, out

        
        
            
    # @staticmethod
    # def standard_prompt_wrap(x: str, y:str='') -> str:
    #     return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    # @staticmethod
    # def value_prompt_wrap(x: str, y: str) -> str:
    #     return wiki_evaluate + x + '\nThought Process: ' + y + '\nEvaluation Process:'  
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        # return Final_evaluate + x + '\n' + y + '\nEvaluation Process: \n'
        if 'the final answer is' not in y.lower():
            return wiki_evaluate + x +'\nThought Process: ' + y + '\nEvaluation Process:\n'  
        else:
            if 'choose the best answer' in x.lower():
                return choose_evaluate + x +'\nAnswer: ' + y.lower().split('the final answer is')[1].replace(': ','') + '\nEvaluation Process:\n'  
            else:
                return Final_evaluate + x + '\n' + y + '\nEvaluation Process: \n'
