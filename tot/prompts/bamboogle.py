cot_prompt = '''
Task: Answer the given question step-by-step, and conclude with the phrase 'so the final answer is: '.
Question: Who lived longer, Theodor Haecker or Harry Vaughan Watkins? Answer: Step 1, when did Theodor Haecker die? Theodor Haecker was 65 years old when he died. Step 2, when did  Harry Vaughan Watkins die? Harry Vaughan Watkins was 69 years old when he died. Step 3, so the final answer is: Harry Vaughan Watkins.
Question: Why did the founder of Versus die? Answer: Step 1, who is the funder of Versus? The founder of Versus was Gianni Versace. Step 2, why did Gianni Versace die? Gianni Versace was shot and killed on the steps of his Miami Beach mansion on July 15, 1997. Step 3, so the final answer is: Shot.
Question: Who is the grandchild of Dambar Shah? Answer: Step 1, who is the son of  Dambar Shah? Dambar Shah (? - 1645) was the father of Krishna Shah. Step 2, who is the son of Krishna Shah? Krishna Shah (? - 1661) was the father of Rudra Shah. Step 3, so the final answer is: Rudra Shah.
Question: {input}
'''
