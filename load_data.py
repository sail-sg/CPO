from datasets import load_dataset
import re 
import random
import json

def remove_substrings_with_double_angle_brackets(input_string):
    # Define the pattern to match substrings within double angled brackets
    pattern = r"<<[^>]+>>"
    # Use the sub() function from the re module to replace matching substrings with an empty string
    result = re.sub(pattern, "", input_string)
    return result

def load_gsm8k_test(path: str = "gsm8k", subset: str = "main", split="test"):
    samples = []
    # dataset = dataset[:200]
    i = 0
    for raw in load_dataset(path, subset, split=split):
        i +=1
        explanation, answer = raw["answer"].split("####")
        explanation = remove_substrings_with_double_angle_brackets(explanation)
        samples.append(
            {
                'question':raw["question"].strip(),
                'explanation':explanation.strip(),
                'answer':answer.strip(),
            }
        )
        if i == 200:
            break
    return samples


def load_svamp_test(path: str = 'tot/data/SVAMP/train.json'):
    samples = []
    with open(path,'r') as f:
        ins = json.load(f)
    for d in ins:
        samples.append(
            {
                'question':d['Body'].strip()+d["Question"].strip(),
                'answer':str(d['Answer']).strip(),
            })
        if 'train' in path:
            if len(samples)>=300:
                break
    return samples

def create_demo_text(cot_flag=True):
    x, z, y = [], [], []  
    # example sentences ...    
    if 1:
        
        x.append("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?")
        z.append("Step 1, There are 15 trees originally, Then there were 21 trees after some more were planted. Step 2, So there must have been 21 - 15 = 6.")
        y.append("6")

        x.append("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?")
        z.append("Step 1, There are originally 3 cars, and  2 more cars arrive. Step 2, 3 + 2 = 5.")
        y.append("5")        

        x.append("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?")
        z.append("Step 1, Originally, Leah had 32 chocolates, and her sister had 42. Step 2, So in total they had 32 + 42 = 74. Step 3, After eating 35, they had 74 - 35 = 39.")
        y.append("39")        

        x.append("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?")
        z.append("Step 1, Jason started with 20 lollipops. Then he had 12 after giving some to Denny. Step 2, So he gave Denny 20 - 12 = 8.")
        y.append("8")        

        x.append("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?")
        z.append("Step 1, Shawn started with 5 toys. Step 2, If he got 2 toys each from his mom and dad, then that is 4 more toys. Step 3, So he has 5 + 4 = 9 toys")
        y.append("9")        

        x.append("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?")
        z.append("Step 1, There were originally 9 computers. Step 2, For each of 4 days, 5 more computers were added, so 5 * 4 = 20 computers were added. Step 3, 9 + 20 is 29.")
        y.append("29")        

        x.append("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?")
        z.append("Step 1, Michael started with 58 golf balls. Step 2, After losing 23 on tuesday, he had 58 - 23 = 35. Step 3, After losing 2 more, he had 35 - 2 = 33 golf balls.")
        y.append("33")        

        x.append("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?")
        z.append("Step 1, Olivia had 23 dollars. Step 2, 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. Step 3, So she has 23 - 15 = 8 dollars left.")
        y.append("8")
    
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # randomize order of the examples ...
    index_list = list(range(len(x)))
    random.shuffle(index_list)

    # Concatenate demonstration examples ...
    direct_answer_trigger_for_fewshot = "The answer (arabic numerals) is "
    demo_text = ""
    for i in index_list:
        if cot_flag:
            if 'Step 3' in z[i]:
                demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         'Step 4, ' + direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
            else:
                demo_text += "Q: " + x[i] + "\nA: " + z[i] + " " + \
                         'Step 3, ' + direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            demo_text += "Q: " + x[i] + "\nA: " + \
                         direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    
    return demo_text
# math_evaluate = '''Evaluate whether the thought helps in partially or directly answering the original question (likely/impossible). 

# Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
# Thought: step 1, 16 eggs per day means 16 * 24 hours = 384 eggs per day. 
# Evaluation Process: It multiplies 16 eggs per day by 24 hours, resulting in 384 eggs per day. The actual statement clearly says Janet's ducks lay 16 eggs per day in total, not per hour. 
# Impossible

# Question: Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days
# Thought: step 1, 80 miles was covered in one day, and 150 miles was covered in another day.
# Evaluation Process: the thought is correct and aligns with the question's details. Although it cannot directly answer the question, this is very helpful in promoting the next step towards the correct reasoning.
# Likely

# Question: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?
# Thought: step 1, 2x as many means that toulouse has 2x as many sheep as charleston. step 2, 4x as many means that charleston has 4x as many sheep as seattle. step 3, 2(2x) = 4x means that toulouse has 4x as many sheep as charleston.
# Evaluation Process: The thought process makes a critical error in step 3 by stating "2(2x) = 4x means that Toulouse has 4x as many sheep as Charleston." This is incorrect. The initial information that Toulouse has twice as many sheep as Charleston is accurate and should remain the basis for calculation. The error seems to be in misunderstanding the multiplication of relationships. 
# Impossible

# Question: Mary is baking a cake. The recipe calls for 6 cups of flour 8 cups of sugar and 7 cups of salt. She already put in 5 cups of flour.How many more cups of sugar than cups of salt does she need to add now?
# Thought: step 1, 6 cups of flour + 8 cups of sugar + 7 cups of salt = 21 cups. 
# Evaluation Process: The given thought for evaluating the question about Mary baking a cake is not helpful for answering the question because it focuses on summing up the total amount of ingredients needed for the cake, which does not directly address the specific question asked. The question is specifically about the difference in the number of cups of sugar versus cups of salt Mary needs to add now, given that she already added 5 cups of flour. 
# Impossible

# Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
# Thought: Step 1, 2 bolts of blue fiber and half that much white fiber means 2/2 + 1/2 = 3. 
# Evaluation Process: It correctly identifies that if the robe requires 2 bolts of blue fiber and half that amount of white fiber, then the total amount of white fiber needed is half of 2 bolts, which is 1 bolt. By adding the 2 bolts of blue fiber to the 1 bolt of white fiber, it correctly calculates that a total of 3 bolts of fiber are needed to make the robe. This thought process directly addresses the question by accurately calculating the total number of bolts required for the robe.
# Likely

# '''
math_evaluate = '''Evaluate whether the thought helps in partially or directly answering the original question (likely/impossible). 

Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Thought: step 1, 16 eggs per day means 16 * 24 hours = 384 eggs per day. 
Evaluation Process: To answer the question ‘How much in dollars does she make every day at the farmers' market?’, we need know how much of eggs are laid per day and the price of each egg. The given thought multiplies 16 eggs per day by 24 hours, resulting in 384 eggs per day. The actual statement clearly says Janet's ducks lay 16 eggs per day in total, not per hour. Thus, the thought will lead to a wrong answer. So the final evaluation is impossible.
Impossible

Question: Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days
Thought: step 1, 80 miles was covered in one day, and 150 miles was covered in another day.
Evaluation Process: To answer the question ‘What's the distance covered by each train in the two days?’,  the thought provide helpful information without any unnecessary details about the distance covered in both days respectively. Although it cannot directly answer the question, this is very helpful in promoting the next step towards the correct reasoning. So the final evaluation is likely.
Likely

Question: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?
Thought: step 1, 2x as many means that toulouse has 2x as many sheep as charleston. step 2, 4x as many means that charleston has 4x as many sheep as seattle. step 3, 2(2x) = 4x means that toulouse has 4x as many sheep as charleston.
Evaluation Process: To answer the question ‘How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?’, the thought process makes a critical error in step 3 by stating "2(2x) = 4x means that Toulouse has 4x as many sheep as Charleston." This is incorrect. The initial information that Toulouse has twice as many sheep as Charleston is accurate and should remain the basis for calculation. The error seems to be in misunderstanding the multiplication of relationships. Thus, the thought will lead to a wrong answer. So the final evaluation is impossible.
Impossible

Question: There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groups. How big is each group of bananas?
Thought: step 1, 87 oranges and 290 bananas are given. 
Evaluation Process: To answer the question ‘How big is each group of bananas?’, we nee to know the number of bananas and groups respectively. The thought provide one of the key information, which is the number of bananas. Although it cannot directly answer the question, this is very helpful in promoting the next step towards the correct reasoning. So the final evaluation is likely.
Likely

Question: Mary is baking a cake. The recipe calls for 6 cups of flour 8 cups of sugar and 7 cups of salt. She already put in 5 cups of flour. How many more cups of sugar than cups of salt does she need to add now?
Thought: step 1, 6 cups of flour + 8 cups of sugar + 7 cups of salt = 21 cups. 
Evaluation Process: To answer the question ‘How many more cups of sugar than cups of salt does she need to add now?’, we need to know the number of cups of sugar and salts respectively, and then calculate the difference.  The given thought focuses on summing up the total amount of ingredients needed for the cake, which does not directly address the specific question asked. Thus, the thought will lead to a wrong answer. So the final evaluation is impossible.
Impossible

Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
Thought: Step 1, 2 bolts of blue fiber and half that much white fiber means 2/2 + 1/2 = 3. 
Evaluation Process: To answer the question ‘How many bolts in total does it take?’, it correctly identifies that if the robe requires 2 bolts of blue fiber and half that amount of white fiber, then the total amount of white fiber needed is half of 2 bolts, which is 1 bolt. By adding the 2 bolts of blue fiber to the 1 bolt of white fiber, it correctly calculates that a total of 3 bolts of fiber are needed to make the robe. This thought process directly addresses the question by accurately calculating the total number of bolts required for the robe. So the final evaluation is likely.
Likely

'''


math_final_evaluate = '''Evaluate whether the thought answers the question (likely/impossible).

Question: Paul got a box of 110 crayons for his birthday. During the school year he gave 90 crayons to his friends while he lost 412 crayons.How many more crayons did he lose than those he gave to his friends?
Thought: step 1, 110 crayons was given by paul's parents as a gift. step 2, 90 crayons were given away by him. step 3, 90-110= -20. step 4, the answer (arabic numerals) is -20.
Evaluation Process: The question asks for the difference in the number of crayons he lost compared to those he gave to his friends. However, the thought process incorrectly focuses on the crayons received by Paul from his parents and those he gave away, which is a different comparison.
Imposible

Question: There are 87 oranges and 290 bananas in Philip's collection. If the bananas are organized into 2 groups and oranges are organized into 93 groupsHow big is each group of bananas?
Thought: step 1, 87 oranges and 290 bananas are given. step 2, 290 divided by 2 is 145. step 3, 145 bananas are in one group. step 4, the answer (arabic numerals) is 145. 
Evaluation Process: The question specifically asked how many bananas would be in each of two equally divided groups, then the thought process is correct.
Likely

Question: Randy has 95 blocks. He uses 20 blocks to build a house and 50 blocks to build a tower.How many more blocks did he use to build the tower than he did to build the house?
Thought: step 1, 95 blocks. step 2, 20 blocks to build a house and 50 blocks to build a tower. step 3, 20 + 50 = 70 blocks used to build the tower. step 4, the answer (arabic numerals) is 70.
Evaluation Process: The question specifically asks for the difference in the number of blocks used to build the tower compared to the house. However, the thought process erroneously adds the number of blocks used for both the house and the tower, leading to a total number of blocks used for both constructions, not the difference between the two as asked.
Impossible

Question: You have 104 dollars.How many packs of dvds can you buy if each pack costs 26 dollars?
Thought: step 1, 104 dollars will buy 104 / 26 = 4 packs of dvd's. step 2, 4 packs of dvd's cost 4 x 26 = 104 dollars. step 3, 104 dollars buys 4 packs of dvd's. step 4, the answer (arabic numerals) is 4. 
Evaluation Process: The question asks how many packs of DVDs can be purchased with 104 dollars if each pack costs 26 dollars. The calculation performed in the thought (104 / 26 = 4) correctly determines the number of DVD packs that can be bought, which is 4 packs. 
Likely

'''

def math_test_output(d, y, out):
    # print('======testing========')
    expression = y.strip().replace(',','')
    if 'arabic numerals' in expression:
        expression = expression.split('(arabic numerals) is ')[-1]
    numbers = re.findall(r'\d+', expression)
    try:
        problem_numbers = re.findall(r'\d+', d['answer'][0])
    except:
        print(d)
        return {'r': 0}, out
    # print('====GR===='+str(problem_numbers) +'====Pre===='+str(numbers))
    if len(numbers)>0:
        numbers = numbers[-1]
    if len(problem_numbers)>0:
        problem_numbers = problem_numbers[0]
    print('====GR===='+str(problem_numbers) +'====Pre===='+str(numbers))
    if numbers != problem_numbers:
        return {'r': 0}, out
    else:
        return {'r': 1}, out

def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
    value_map = {'impossible': 0.001, 'unlikely': 0.001, 'likely': 1, 'sure': 1}  # TODO: ad hoc
    value = 0.001
    for item in value_outputs:
        if item.lower() in value_map:
            value = value_map[item.lower()]
            return value
    for v in ['impossible','unlikely','likely', 'sure']:
        for item in value_outputs:
            if v in item.lower():
                    value = value_map[v]
                    return value
    # value = sum(value * value_names.count(name) for name, value in value_map.items())
    return value
