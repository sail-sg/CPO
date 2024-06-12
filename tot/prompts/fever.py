cot_prompt = '''Task: Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFO.
Claim: Reg Watson is a current television producer. Answer: Step 1, who is Reg Watson? Reginald James Watson AM was an Australian television producer and screenwriter. Step 2, when did  Reginald James Watson AM die? Reginald James Watson AM died on 8 October 2019. Step 3, so the final answer is: REFUTES.
Claim: The Gadsden flag was named by Christopher Gadsden. Answer: Step 1, what is the origin of the name of the Gadsden flag? The Gadsden flag is named after politician Christopher Gadsden. Step 2,who named the Gadsden flag? there is no information on who named the Gadsden flag. Step 3, so the final answer is: NOT ENOUGH INFO.
Claim: Black Mirror is about society. Answer: Step 1, what is the son of Black Mirror? Black Mirror is a British anthology television series. Step 2, what issues does this series discuss? The series uses technology to comment on contemporary social issues. Step 3, so the final answer is: SUPPORTS.
Claim: {input}
'''
