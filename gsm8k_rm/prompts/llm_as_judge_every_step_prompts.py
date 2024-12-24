PROMPT_INSTRUCTION = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGiven a math question and partial solution steps, analyze each step in the solution, then determine whether it is correct. Provide the analysis for each step first, then indicate with 'Yes' or 'No' whether it is correct. I will give you a few examples of the task.

Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Solution:
Step 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. 
Step 2: Altogether, she sold 48 + 24 = 72 clips. The answer is 72.
Analysis: 
Step 1: This step calculates the number of clips sold in May by dividing the number of clips sold in April by 2. Correct? Yes.
Step 2: This step calculates the total number of clips sold by adding the number of clips sold in April and May = 24 + 48 = 72. Correct? Yes.
I hope this helps. 

Example 2
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Solution:
Step 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May.
Step 2: Altogether, she sold 24 + 24 = 48 clips. The answer is 48.
Analysis: 
Step 1: Divides the number of clips sold in April by 2 to get the number of clips sold in May. Correct? Yes.
Step 2: Calculates the total number of clips by adding the number of clips sold in April and May. However, it makes an error in the calculation. It should add the clips (48 + 24 = 72) instead of (24 + 24 = 48). Correct? No.
I hope this helps. 

Example 3
Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?
Solution:
Step 1: He buys 2 large pizzas, so he has 2 * 16 = 32 slices.
Step 2: He buys 2 small pizzas, so he has 2 * 8 = 16 slices.
Step 3: There are 32 slices from the large pizzas and 16 slices from the small pizzas, so he eats 32 - 16 = 16 pieces that day. The answer is 16.
Analysis: 
Step 1: Calculates the number of slices for large pizzas by multiplying the number of large pizzas by the number of slices per large pizza: 2 * 16 = 32. Correct? Yes.
Step 2: This step calculates the number of slices for small pizzas by multiplying the number of small pizzas by the number of slices per small pizza: 2 * 8 = 16. Correct? Yes.
Step 3: This step calculates the total number of slices by subtracting the number of slices from the small pizzas from the number of slices from the large pizzas. However, it makes an error in the calculation. It should add the slices (32 + 16 = 48) instead of subtracting them (32 - 16 = 16). Correct? No.
I hope this helps. 

Example 4
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Solution:
Step 1: Weng earns $12 an hour, so she earns $12 / 60 minutes = $0.2 per minute.
Step 2: She did 50 minutes of babysitting, so she earns $0.2 * 50 = $10.
Analysis: 
Step 1: Calculates Weng's per-minute rate ($0.2) by dividing $12 by 60. Correct? Yes.
Step 2: Multiplies this by the number of minutes worked (50) to get the total earnings ($10). Correct? Yes.
I hope this helps. 

Example 5
Question: The centerpieces at Glenda's wedding reception include a fishbowl containing 2 fish, except for one table that has 3 fish. There are 32 tables. How many fish are there?
Solution:
Step 1: There are 32 tables * 2 fish/table = 64 fish.
Step 2: The table with 3 fish increases the number to 3 + 64 = 67 fish. The answer is 67.
Analysis: 
Step 1: This step calculates the total number of fish by multiplying the number of tables by the number of fish per table: 32 * 2 = 64. Correct? Yes.
Step 2: This step adds the extra fish from the table with 3 fish to the total number of fish. However, it makes an error in the calculation. It should add the extra fish to the total (64 + 1 = 65) instead of (3 + 64 = 67). Correct? No.
I hope this helps. 

Example 6
Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Solution:
Step 1: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week.
Step 2: There are 50 weeks in a year, so he writes 12 * 50 = 600 pages a year.
Analysis:
Step 1 calculates the pages written per week by multiplying the number of pages written per letter by the number of friends and the number of times per week: 3 * 2 * 2 = 12. Correct? Yes.
Step 2 makes an error in assuming there are only 50 weeks in a year. There are 52 weeks in a year, so the final calculation is incorrect. Correct? No.
I hope this helps. 

Example 7
Question: In a court-mandated traffic class, there are a certain number of drunk drivers and 4 less than 7 times that many students. If there are 45 students total, how many drunk drivers are there?
Solution:
Step 1: Let x be the number of drunk drivers.
Step 2: 7x - 4 = 45.
Step 3: 7x = 52.
Step 4: x = 7 drivers. The answer is 7.
Analysis:
Step 1: Defines x as the number of drunk drivers. Correct? Yes.
Step 2: Sets up the equation 7x - 4 = 45 to represent the total number of students. Correct? Yes.
Step 3: Solves for 7x by adding 4 to both sides of the equation, resulting in 7x = 52. However, it makes an error in the calculation. It should be 7x = 49 instead of 7x = 52. Correct? No.
Step 4: Solves for x by dividing both sides of the equation by 7, resulting in x = 7. However, it is incorrect because it is based on the incorrect equation. Correct? No.
I hope this helps. 

Example 8
Question: There are 141 gold balloons and twice as many silver balloons. If the gold and silver balloons were added to 150 black balloons, how many balloons are there in total?
Solution:
Step 1: There are 2 * 141 = 282 silver balloons.
Step 2: The total number of balloons is 150 + 282 = 432.
Step 3: There are 432 balloons in total. The answer is 432.
Analysis:
Step 1: Calculates the number of silver balloons by multiplying the number of gold balloons by 2: 2 * 141 = 282. Correct? Yes.
Step 2: This step calculates the total number of balloons by adding the number of black balloons to the number of silver balloons: 150 + 282 = 432. However, it omits the number of gold balloons. The correct calculation should be 141 + 282 + 150 = 573. Correct? No.
Step 3: This step builds on the incorrect calculation from Step 2 and incorrectly states the total number of balloons. Correct? No.
I hope this helps. 

Now I will provide you with a new question and solution to analyze.
Question: {problem}
Solution:
{solution}
<|start_header_id|>assistant<|end_header_id|>\n\n
Analysis:
"""

PROMPT_ICL = """
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Solution:
Step 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. 
Step 2: Altogether, she sold 48 + 24 = 72 clips. The answer is 72.
Analysis: 
Step 1: This step calculates the number of clips sold in May by dividing the number of clips sold in April by 2. Correct? Yes.
Step 2: This step calculates the total number of clips sold by adding the number of clips sold in April and May = 24 + 48 = 72. Correct? Yes.
I hope this helps. 

Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Solution:
Step 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May.
Step 2: Altogether, she sold 24 + 24 = 48 clips. The answer is 48.
Analysis: 
Step 1: Divides the number of clips sold in April by 2 to get the number of clips sold in May. Correct? Yes.
Step 2: Calculates the total number of clips by adding the number of clips sold in April and May. However, it makes an error in the calculation. It should add the clips (48 + 24 = 72) instead of (24 + 24 = 48). Correct? No.
I hope this helps. 

Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?
Solution:
Step 1: He buys 2 large pizzas, so he has 2 * 16 = 32 slices.
Step 2: He buys 2 small pizzas, so he has 2 * 8 = 16 slices.
Step 3: There are 32 slices from the large pizzas and 16 slices from the small pizzas, so he eats 32 - 16 = 16 pieces that day. The answer is 16.
Analysis: 
Step 1: Calculates the number of slices for large pizzas by multiplying the number of large pizzas by the number of slices per large pizza: 2 * 16 = 32. Correct? Yes.
Step 2: This step calculates the number of slices for small pizzas by multiplying the number of small pizzas by the number of slices per small pizza: 2 * 8 = 16. Correct? Yes.
Step 3: This step calculates the total number of slices by subtracting the number of slices from the small pizzas from the number of slices from the large pizzas. However, it makes an error in the calculation. It should add the slices (32 + 16 = 48) instead of subtracting them (32 - 16 = 16). Correct? No.
I hope this helps. 

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Solution:
Step 1: Weng earns $12 an hour, so she earns $12 / 60 minutes = $0.2 per minute.
Step 2: She did 50 minutes of babysitting, so she earns $0.2 * 50 = $10.
Analysis: 
Step 1: Calculates Weng's per-minute rate ($0.2) by dividing $12 by 60. Correct? Yes.
Step 2: Multiplies this by the number of minutes worked (50) to get the total earnings ($10). Correct? Yes.
I hope this helps. 

Question: The centerpieces at Glenda's wedding reception include a fishbowl containing 2 fish, except for one table that has 3 fish. There are 32 tables. How many fish are there?
Solution:
Step 1: There are 32 tables * 2 fish/table = 64 fish.
Step 2: The table with 3 fish increases the number to 3 + 64 = 67 fish. The answer is 67.
Analysis: 
Step 1: This step calculates the total number of fish by multiplying the number of tables by the number of fish per table: 32 * 2 = 64. Correct? Yes.
Step 2: This step adds the extra fish from the table with 3 fish to the total number of fish. However, it makes an error in the calculation. It should add the extra fish to the total (64 + 1 = 65) instead of (3 + 64 = 67). Correct? No.
I hope this helps. 

Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?
Solution:
Step 1: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week.
Step 2: There are 50 weeks in a year, so he writes 12 * 50 = 600 pages a year.
Analysis:
Step 1 calculates the pages written per week by multiplying the number of pages written per letter by the number of friends and the number of times per week: 3 * 2 * 2 = 12. Correct? Yes.
Step 2 makes an error in assuming there are only 50 weeks in a year. There are 52 weeks in a year, so the final calculation is incorrect. Correct? No.
I hope this helps. 

Question: In a court-mandated traffic class, there are a certain number of drunk drivers and 4 less than 7 times that many students. If there are 45 students total, how many drunk drivers are there?
Solution:
Step 1: Let x be the number of drunk drivers.
Step 2: 7x - 4 = 45.
Step 3: 7x = 52.
Step 4: x = 7 drivers. The answer is 7.
Analysis:
Step 1: Defines x as the number of drunk drivers. Correct? Yes.
Step 2: Sets up the equation 7x - 4 = 45 to represent the total number of students. Correct? Yes.
Step 3: Solves for 7x by adding 4 to both sides of the equation, resulting in 7x = 52. However, it makes an error in the calculation. It should be 7x = 49 instead of 7x = 52. Correct? No.
Step 4: Solves for x by dividing both sides of the equation by 7, resulting in x = 7. However, it is incorrect because it is based on the incorrect equation. Correct? No.
I hope this helps. 

Question: There are 141 gold balloons and twice as many silver balloons. If the gold and silver balloons were added to 150 black balloons, how many balloons are there in total?
Solution:
Step 1: There are 2 * 141 = 282 silver balloons.
Step 2: The total number of balloons is 150 + 282 = 432.
Step 3: There are 432 balloons in total. The answer is 432.
Analysis:
Step 1: Calculates the number of silver balloons by multiplying the number of gold balloons by 2: 2 * 141 = 282. Correct? Yes.
Step 2: This step calculates the total number of balloons by adding the number of black balloons to the number of silver balloons: 150 + 282 = 432. However, it omits the number of gold balloons. The correct calculation should be 141 + 282 + 150 = 573. Correct? No.
Step 3: This step builds on the incorrect calculation from Step 2 and incorrectly states the total number of balloons. Correct? No.
I hope this helps. 

Question: {problem}
Solution:
{solution}
Analysis: 
"""