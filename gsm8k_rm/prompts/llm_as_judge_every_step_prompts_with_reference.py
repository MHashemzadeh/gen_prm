PROMPT_INSTRUCTION = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGiven a math question, a correct a solution and a proposed solution, analyze each step in the proposed solution, then determine whether the step is correct. Provide the analysis for each step first, then indicate with 'Yes' or 'No' whether it is correct. I will give you a few examples of the task.

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Correct solution: Natalia sold 48 / 2 = 24 clips in May. In total, she sold 48 + 24 = 72 clips. The answer is 72.
Proposed solution:
Step 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. 
Step 2: Altogether, she sold 48 + 24 = 72 clips. The answer is 72.
Analysis: 
Step 1: This step calculates the number of clips sold in May by dividing the number of clips sold in April by 2. Correct? Yes.
Step 2: This step calculates the total number of clips sold by adding the number of clips sold in April and May = 24 + 48 = 72. Correct? Yes.
I hope this helps. 

Example 2
Question: Val cuts a single watermelon into 40 slices, and each slice has 20 black seeds and 20 white seeds. What is the total number of seeds in a watermelon?
Correct solution: 40 slices per watermelon and each slice having 20 black seeds means each watermelon has 20*40=800 black seeds. 40 slices per watermelon and each slice having 20 white seeds means each watermelon has 20*40=800 white seeds. In total, each watermelon has 800+800=1600 seeds. What is the total number of seeds in a watermelon?. The answer is 1,600.
Proposed solution:
Step 1: There are 20 x 40 = 8000 black seeds.
Step 2: There are 20 x 40 = 8000 white seeds.
Step 3: So, the total number of seeds is 8000 + 8000 = 16000. The answer is 16000.
Analysis:
Step 1: This step calculates the number of black seeds by multiplying the number of slices by the number of black seeds per slice: 20 * 40 = 800. Correct? Yes.
Step 2: This step calculates the number of white seeds by multiplying the number of slices by the number of white seeds per slice: 20 * 40 = 800. Correct? Yes.
Step 3: This step calculates the total number of seeds by adding the number of black seeds and white seeds. However, it makes an error in the calculation. It should add 800 + 800 = 1600. Correct? No
I hope this helps.

Example 3
Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?
Correct solution: Two large pizzas have 2 * 16 = 32 slices. Two small pizzas have 2 * 8 = 16 slices. In total, he eats 32 + 16 = 48 slices that day. The answer is 48.
Proposed solution:
Step 1: He buys 2 large pizzas, so he has 2 * 16 = 32 slices.
Step 2: He buys 2 small pizzas, so he has 2 * 8 = 16 slices.
Step 3: There are 32 slices from the large pizzas and 16 slices from the small pizzas, so he eats 32 + 16 + 16 = 64 pieces that day. The answer is 64.
Analysis: 
Step 1: Calculates the number of slices for large pizzas by multiplying the number of large pizzas by the number of slices per large pizza: 2 * 16 = 32. Correct? Yes.
Step 2: This step calculates the number of slices for small pizzas by multiplying the number of small pizzas by the number of slices per small pizza: 2 * 8 = 16. Correct? Yes.
Step 3: This step calculates the total number of slices. However, it adds the slices from the small pizzas twice (32 + 16 + 16 = 64). Correct? No.
I hope this helps. 

Example 4
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Correct solution: Weng earns $12 / 60 minutes = $0.2 per minute. Since she did 50 minutes of babysitting, we multiply her per-minute rate by the number of minutes she worked: $0.2 * 50 = $10. The answer is 10.
Proposed solution:
Step 1: Weng earns $12 an hour, so she earns $12 / 60 minutes = $0.2 per minute.
Step 2: She did 50 minutes of babysitting, so she earns $0.2 * 50 = $10.
Analysis: 
Step 1: Calculates Weng's per-minute rate ($0.2) by dividing $12 by 60. Correct? Yes.
Step 2: Multiplies this by the number of minutes worked (50) to get the total earnings ($10). Correct? Yes.
I hope this helps. 

Example 5
Question: The centerpieces at Glenda's wedding reception include a fishbowl containing 2 fish, except for one table that has 3 fish. There are 32 tables. How many fish are there?
Correct solution: There are 31 tables with 2 fish. There is one table with 3 fish. In total, there are 31 * 2 + 3 = 65 fish. The answer is 65.
Solution: There are 32 tables * 2 fish/table = 64 fish. The table with 3 fish increases the number to 3 + 64 = 67 fish. The answer is 67.
Analysis: 
Step 1: This step calculates the total number of fish by multiplying the number of tables by the number of fish per table (32 * 2 = 64). Correct? Yes.
Step 2: This step adds the extra fish from the table with 3 fish to the total number of fish, which is incorrect. Correct? No.
I hope this helps. 

Example 6
Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Correct solution: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week. There are 52 weeks in a year, so he writes 12 * 52 = 624 pages a year. The answer is 624.
Proposed solution:
Step 1: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week.
Step 2: There are 50 weeks in a year, so he writes 12 * 50 = 600 pages a year.
Analysis:
Step 1 calculates the pages written per week by multiplying the number of pages written per letter by the number of friends and the number of times per week: 3 * 2 * 2 = 12. Correct? Yes.
Step 2 makes an error in assuming there are only 50 weeks in a year. There are 52 weeks in a year, so the final calculation is incorrect. Correct? No.
I hope this helps. 

Example 7
Question: In a court-mandated traffic class, there are a certain number of drunk drivers and 4 less than 7 times that many students. If there are 45 students total, how many drunk drivers are there?
Correct solution: Let x be the number of students. Let y be the number of drunk drivers. 7y - 4 = x. x = 45. 7y = 49. y = 7. The answer is 7.
Proposed solution:
Step 1: Let x be the number of drunk drivers.
Step 2: 7x - 4 = 45.
Step 3: 7x = 56.
Step 4: x = 8. The answer is 8.
Analysis:
Step 1: Defines x as the number of drunk drivers. Correct? Yes.
Step 2: Sets up the equation 7x - 4 = 45 to represent the total number of students. Correct? Yes.
Step 3: Solves for 7x by adding 4 to both sides of the equation, resulting in 7x = 56. However, it makes an error in the calculation. It should be 7x = 49 instead of 7x = 56. Correct? No.
Step 4: Builds on the incorrect calculation from Step 3. Correct? No.
I hope this helps. 

Example 8
Question: There are 141 gold balloons and twice as many silver balloons. If the gold and silver balloons were added to 150 black balloons, how many balloons are there in total?
Correct solution: There are 2 * 141 = 282 silver balloons. There are 141 gold balloons. The total number of balloons is 141 + 282 + 150 = 573. The answer is 573.
Solution: There are 2 * 141 = 282 silver balloons. The total number of balloons is 150 + 282 = 432. The answer is 432.
Analysis:
Step 1: Calculates the number of silver balloons by multiplying the number of gold balloons by 2: 2 * 141 = 282. Correct? Yes.
Step 2: This step calculates the total number of balloons by adding the number of black balloons to the number of silver balloons: 150 + 282 = 432. However, it omits the number of gold balloons. The correct calculation should be 141 + 282 + 150 = 573. Correct? No. 

Now I will provide you with a new question and solution to analyze.
Question: {problem}
Correct solution: {correct_solution}
Proposed solution:
{solution}
Analysis:
"""

PROMPT_ICL = """
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Correct solution: Natalia sold 48 / 2 = 24 clips in May. In total, she sold 48 + 24 = 72 clips. The answer is 72.
Proposed solution:
Step 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. 
Step 2: Altogether, she sold 48 + 24 = 72 clips. The answer is 72.
Analysis: 
Step 1: This step calculates the number of clips sold in May by dividing the number of clips sold in April by 2, which is 48 / 2 = 24. Correct? Yes.
Step 2: This step calculates the total number of clips sold by adding the number of clips sold in April and May, which is 24 + 48 = 72. Correct? Yes.
I hope this helps.

Question: Val cuts a single watermelon into 40 slices, and each slice has 20 black seeds and 20 white seeds. What is the total number of seeds in a watermelon?
Correct solution: 40 slices per watermelon and each slice having 20 black seeds means each watermelon has 20*40=800 black seeds. 40 slices per watermelon and each slice having 20 white seeds means each watermelon has 20*40=800 white seeds. In total, each watermelon has 800+800=1600 seeds. What is the total number of seeds in a watermelon?. The answer is 1,600.
Proposed solution:
Step 1: There are 20 x 40 = 8000 black seeds.
Step 2: There are 20 x 40 = 8000 white seeds.
Step 3: So, the total number of seeds is 8000 + 8000 = 16000. The answer is 16000.
Analysis:
Step 1: This step calculates the number of black seeds by multiplying the number of slices by the number of black seeds per slice, which is 20 * 40 = 800. Correct? Yes.
Step 2: This step calculates the number of white seeds by multiplying the number of slices by the number of white seeds per slice, which is 20 * 40 = 800. Correct? Yes.
Step 3: This step calculates the total number of seeds by adding the number of black seeds and white seeds, which is 800 + 800 = 1600. However, the step computes 8000 + 8000 = 16000. Correct? No
I hope this helps.

Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?
Correct solution: Two large pizzas have 2 * 16 = 32 slices. Two small pizzas have 2 * 8 = 16 slices. In total, he eats 32 + 16 = 48 slices that day. The answer is 48.
Proposed solution:
Step 1: He buys 2 large pizzas, so he has 2 * 16 = 32 slices.
Step 2: He buys 2 small pizzas, so he has 2 * 8 = 16 slices.
Step 3: There are 32 slices from the large pizzas and 16 slices from the small pizzas, so he eats 32 + 16 + 16 = 64 pieces that day. The answer is 64.
Analysis: 
Step 1: Calculates the number of slices for large pizzas by multiplying the number of large pizzas (2) by the number of slices per large pizza (16), which is 2 * 16 = 32. Correct? Yes.
Step 2: Calculates the number of slices for small pizzas by multiplying the number of small pizzas (2) by the number of slices per small pizza (8), which is 2 * 8 = 16. Correct? Yes.
Step 3: Calculates the total number of slices by adding the slices from the large pizzas (32) and the small pizzas (16), which is 32 + 16 = 48. However, it adds the slices from the small pizzas twice (32 + 16 + 16 = 64). Correct? No.
I hope this helps.

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Correct solution: Weng earns $12 / 60 minutes = $0.2 per minute. Since she did 50 minutes of babysitting, we multiply her per-minute rate by the number of minutes she worked: $0.2 * 50 = $10. The answer is 10.
Proposed solution:
Step 1: Weng earns $12 an hour, so she earns $12 / 60 minutes = $0.2 per minute.
Step 2: She did 50 minutes of babysitting, so she earns $0.2 * 50 = $10.
Analysis: 
Step 1: Calculates Weng's per-minute rate ($0.2) by dividing the total earnings by the number of minutes (60), which is $12 / 60 = $0.2. Correct? Yes.
Step 2: Calculates the total earnings by multiplying the per-minute rate ($0.2) by the number of minutes worked (50), which is $0.2 * 50 = $10. Correct? Yes.
I hope this helps.

Question: The centerpieces at Glenda's wedding reception include a fishbowl containing 2 fish, except for one table that has 3 fish. There are 32 tables. How many fish are there?
Correct solution: There are 31 tables with 2 fish. There is one table with 3 fish. In total, there are 31 * 2 + 3 = 65 fish. The answer is 65.
Proposed solution:
Step 1: There are 32 tables * 2 fish/table = 64 fish.
Step 2: The table with 3 fish increases the number to 3 + 64 = 67 fish. The answer is 67.
Analysis: 
Step 1: Calculates the total number of fish by multiplying the number of tables by the number of fish per table, which is 32 * 2 = 64. Correct? Yes.
Step 2: Calculates the total number of fish by adding the extra fish from the table with 3 fish, which is 1 + 64 = 65. However, it calculates 3 + 64 = 67. Correct? No.
I hope this helps.

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Correct solution: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week. There are 52 weeks in a year, so he writes 12 * 52 = 624 pages a year. The answer is 624.
Proposed solution:
Step 1: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week.
Step 2: There are 50 weeks in a year, so he writes 12 * 50 = 600 pages a year.
Analysis:
Step 1: Calculates the pages written per week by multiplying the number of pages written per letter by the number of friends and the number of times per week, which is 3 * 2 * 2 = 12. Correct? Yes.
Step 2: Calculates the pages total pages written in a year by multiplying the pages written per week by the number of weeks in a year, which is 12 * 52 = 624. However, it mistakes the number of weeks in a year to be 50 and calculates 12 * 50 = 600. Correct? No.
I hope this helps.

Question: In a court-mandated traffic class, there are a certain number of drunk drivers and 4 less than 7 times that many students. If there are 45 students total, how many drunk drivers are there?
Correct solution: Let x be the number of students. Let y be the number of drunk drivers. 7y - 4 = x. x = 45. 7y = 49. y = 7. The answer is 7.
Proposed solution:
Step 1: Let x be the number of drunk drivers.
Step 2: 7x - 4 = 45.
Step 3: 7x = 56.
Step 4: x = 8. The answer is 8.
Analysis:
Step 1: Defines x as the number of drunk drivers. Correct? Yes.
Step 2: Sets up the equation 7x - 4 = 45 to represent the total number of students. Correct? Yes.
Step 3: Solves for 7x by adding 4 to both sides of the equation, which is 7x = 45 + 4 = 49. However, it calculates 7x = 56. Correct? No.
Step 4: Builds on the incorrect calculation. Correct? No.
I hope this helps.

Question: There are 141 gold balloons and twice as many silver balloons. If the gold and silver balloons were added to 150 black balloons, how many balloons are there in total?
Correct solution: There are 2 * 141 = 282 silver balloons. There are 141 gold balloons. The total number of balloons is 141 + 282 + 150 = 573. The answer is 573.
Proposed solution:
Step 1: There are 2 * 141 = 282 silver balloons.
Step 2: The total number of balloons is 150 + 282 = 432. The answer is 432.
Analysis:
Step 1: Calculates the number of silver balloons by multiplying the number of gold balloons by 2, which is 2 * 141 = 282. Correct? Yes.
Step 2: Calculates the total number of balloons by adding the number of black balloons to the number of gold balloons to silver balloons to black balloons, which is 141 + 282 + 150 = 573. However, it omits the number of gold balloons and calculates 150 + 282 = 432. Correct? No.
I hope this helps.

Question: {problem}
Correct solution: {correct_solution}
Proposed solution:
{solution}
Analysis: 
"""
