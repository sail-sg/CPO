import re
import os
# import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.fever import * 

wiki_evaluate = '''Evaluate whether the language model can effectively decompose the claim into relevant sub-questions, and assess whether this decomposition helps in partially or directly verifying the original claim. The outcome will determine if this process of decomposition is "Likely" or "Impossible" to aid in verifing the claim.

Evaluation Steps: Check if the language model can identify and decompose key sub-questions that are directly related to the original question.
Evaluation Process: 1. Analyze whether each sub-question identified by the model is directly relevant to the verify the original claim. 2. Determine if the decomposition of these sub-questions forms a reasonable verification process to the original claim.
Evaluation Result: 1. Likely: If the language model successfully decomposes the original claim into relevant sub-questions that help construct the final answer. 2. Impossible: If the language model fails to effectively decompose the claim, or if the decomposed sub-questions are not directly relevant to make the verification.

Claim: Reg Watson is a current television producer.
Thought Process: Step 1, who is Reg Watson? Reginald James Watson AM was an Australian television producer and screenwriter.
Evaluation Process:
Relevance of Sub-Questions: The sub-question of identifying who Reg Watson is, is directly relevant as it establishes the necessary context to further explore the original claim about his current status as a television producer.
Effectiveness of Decomposition: By first identifying Reg Watson and then investigating his current professional activities, this approach forms a reasonable verification process. 
Evaluation Result:
Likely

Claim: The Gadsden flag was named by Christopher Gadsden.
Thought: Step 1, why did Christopher Gadsden die? Gadsden died from head injuries suffered in a fall near his home.
Evaluation Process:
Relevance of Sub-Questions: The sub-question about the cause of Gadsden's death does not directly contribute to verifying the claim about the naming of the Gadsden flag. Instead, questions should focus on his involvement with the flag and any direct actions or contributions he made towards its naming.
Verification Process: A more effective verification process would involve gathering evidence of Christopher Gadsden's direct involvement with the flag, including any documented instances where he is credited with its naming, as well as understanding the historical context of the flag's creation and use.
Evaluation Results:
Impossible

Claim: Black Mirror is about society.
Thought Process: Step 1, what is the son of Black Mirror? Black Mirror is a British anthology television series. Step 2, what issues does this series discuss? The series uses technology to comment on contemporary social issues.
Evaluation Process:
Relevance of Sub-Questions: Each sub-question is directly relevant and helps verify the original claim. The first establishes the series' nature and scope, and the second addresses the thematic content, specifically its societal focus.
Verification Process: This process is reasonable for verifying the original claim. First, it establishes what "Black Mirror" is, laying the groundwork for further inquiry. Then, it dives into the series' thematic concerns, confirming its focus on societal issues through the lens of technology. This approach not only verifies the claim but also provides insight into how the series approaches its critique of society.
Evaluation Results:
Likely

Claim: '''

Final_evaluate = '''Evaluate whether the conclusion can be drawn based on reasoning logic in the thought process." (Likely/Impossible).

Claim: The Gadsden flag was named by Christopher Gadsden.
Thought Process: Step 1, why did Christopher Gadsden die? Gadsden died from head injuries suffered in a fall near his home. Step 2, what is the origin of the name of the Gadsden flag? The Gadsden flag is named after politician Christopher Gadsden. 
Conclusion: so the final answer is: REFUTES.
Evaluation Process:
The thought process includes a correct statement about the origin of the Gadsden flag's name that aligns with the claim, but then concludes incorrectly that this information refutes the claim.
Evaluation Results:
Impossible

Claim: Black Mirror is about society.
Thought Process: Step 1, what is the son of Black Mirror? Black Mirror is a British anthology television series. Step 2, what issues does this series discuss? The series uses technology to comment on contemporary social issues.
Conclusion: so the final answer is: SUPPORTS.
Evaluation Process:
The conclusion logically follows from the information provided, supporting the claim that "Black Mirror" is about society.
Evaluation Results:
Likely

Claim: '''

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
        expression = output.strip().split('answer is')[1].lower().split('\n')[0]
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

        
        
            


    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y


    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        # return Final_evaluate + x + '\n' + y + '\nEvaluation Process: \n'
        if 'the final answer is' not in y.lower():
            return wiki_evaluate + x +'\nThought Process: ' + y + '\nEvaluation Process:\n'  
        else:
            try:
                return Final_evaluate + x + '\nThought Process:' + y.split('step 3, ')[0] + '\nConclusion: ' + y.split('step 3, ')[1] + '\nEvaluation Process: \n'
            except:
                print(y)
                exit()
