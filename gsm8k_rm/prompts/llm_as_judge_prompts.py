PROMPT_INSTRUCTION = """Given a math question and partial solution steps. You must determine whether the calculations so far are part of a complete correct solution. The final answer may not be reached yet. What matters is whether the steps do not include erros and will eventually lead to the correct final answer to the question. Provide an analysis first, then indicate with 'Yes' or 'No' whether it is correct. I will give you a few examples of the task.\n\nExample 1\nQuestion: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nNatalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips.\nAnalysis: The solution so far calculates the number of clips sold in May (24) and sums the clips sold in April (48) and May (24) to get the total (72).\nCorrect so far? Yes\n\nExample 2\nQuestion: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nStep 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May.\nStep 2: Altogether, she sold 24 + 24 = 48 clips.\nAnalysis: Step 1 calculates the number of clips sold in May (24). Step 2 makes an error in the final sum. It should be 48 (April) + 24 (May) = 72 clips, not 24 + 24 = 48.\nCorrect so far? No\n\nExample 3\nQuestion: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?\nStep 1: He buys 2 large pizzas, so he has 2 * 16 = 32 slices.\nStep 2: He buys 2 small pizzas, so he has 2 * 8 = 16 slices.\nStep 3: There are 32 slices from the large pizzas and 16 slices from the small pizzas, so he eats 32 - 16 = 16 pieces that day.\nAnalysis: Steps 1 and 2 calculate the number of slices for large and small pizzas. Step 3 makes an error in the final calculation. It should add the slices (32 + 16 = 48) instead of subtracting them (32 - 16 = 16).\nCorrect so far? No\n\nExample 4\nQuestion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nStep 1: Weng earns $12 an hour, so she earns $12 / 60 minutes = $0.2 per minute.\nStep 2: She did 50 minutes of babysitting, so she earns $0.2 * 50 = $10.\nAnalysis: The solution so far calculates Weng's per-minute rate ($0.2) and then multiplies this by the number of minutes worked (50) to get the total earnings ($10).\nCorrect so far? Yes\n\nExample 5\nQuestion: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?\nStep 1: Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today.\nStep 2: Since yesterday, Julie read 12 + 24 = 36 pages.\nStep 3: So, there are 120 - 36 = 84 pages left to be read.\nAnalysis: The solution so far calculates the pages read today (24), the total pages read so far (36), and the remaining pages (84). All calculations are correct.\nCorrect so far? Yes\n\nExample 6\nQuestion: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\nStep 1: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week.\nStep 2: There are 50 weeks in a year, so he writes 12 * 50 = 600 pages a year.\nAnalysis: Step 1 calculates the pages written per week (12). Step 2 makes an error in assuming there are only 50 weeks in a year. There are 52 weeks in a year, so the final calculation is incorrect.\nCorrect so far? No\n\nNow, given the following:\nQuestion: {problem}\n{prefix}\nAnalyze the solution so far and provide an analysis. After the analysis, determine whether the steps in solution will lead to the correct final answer. The final answer may not be reached yet. What matters is whether the steps provided are a part of a complete correct solution. Output Yes or No, nothing else. If you are unsure provide your best guess."""

PROMPT_NO_INSTRUCTION = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nNatalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May. Altogether, she sold 48 + 24 = 72 clips.\nAnalysis: The partial solution calculates the number of clips sold in May (24) and sums the clips sold in April (48) and May (24) to get the total (72).\nWill lead to correct final answer? Yes\n\nQuestion: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nStep 1: Natalia sold 48 clips in April and half as many clips in May, so she sold 48 / 2 = 24 clips in May.\nStep 2: Altogether, she sold 24 + 24 = 48 clips.\nAnalysis: Step 1 correctly calculates the number of clips sold in May (24). Step 2 makes an error in the final sum. It should be 48 (April) + 24 (May) = 72 clips, not 24 + 24 = 48.\nWill lead to correct final answer? No\n\nQuestion: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?\nStep 1: He buys 2 large pizzas, so he has 2 * 16 = 32 slices.\nStep 2: He buys 2 small pizzas, so he has 2 * 8 = 16 slices.\nStep 3: There are 32 slices from the large pizzas and 16 slices from the small pizzas, so he eats 32 - 16 = 16 pieces that day.\nAnalysis: While Steps 1 and 2 calculate the number of slices for large and small pizzas, Step 3 makes an error in the final calculation. It should add the slices (32 + 16 = 48) instead of subtracting them (32 - 16 = 16).\nWill lead to correct final answer? No\n\nQuestion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nStep 1: Weng earns $12 an hour, so she earns $12 / 60 minutes = $0.2 per minute.\nStep 2: She did 50 minutes of babysitting, so she earns $0.2 * 50 = $10.\nAnalysis: The solution calculates Weng's per-minute rate ($0.2) and then multiplies this by the number of minutes worked (50) to get the total earnings ($10).\nWill lead to correct final answer? Yes\n\nQuestion: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?\nStep 1: Julie read twice as many pages as yesterday, so she read 12 * 2 = 24 pages today.\nStep 2: Since yesterday, Julie read 12 + 24 = 36 pages.\nStep 3: So, there are 120 - 36 = 84 pages left to be read.\nAnalysis: The solution calculates the pages read today (24), the total pages read so far (36), and the remaining pages (84). All calculations are correct.\nWill lead to correct final answer? Yes\n\nQuestion: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?\nStep 1: James writes a 3-page letter to 2 different friends twice a week, so he writes 3 * 2 * 2 = 12 pages every week.\nStep 2: There are 50 weeks in a year, so he writes 12 * 50 = 600 pages a year.\nAnalysis: Step 1 calculates the pages written per week (12). Step 2 makes an error in assuming there are only 50 weeks in a year. There are 52 weeks in a year, so the final calculation is incorrect.\nWill lead to correct final answer? No\n\nQuestion: The exchange rate refers to the rate at which one country's currency is exchanged for another country's currency. Soojeong came back from a trip to the U.S. and exchanged the remaining 140 dollars for 280000 won in Korean money at the bank today. What is the exchange rate of the Korean Won to the US Dollar today?\n Step 1: To find the exchange rate of the Korean Won to the US Dollar, we need to divide the amount of Dollars Soojeong exchanged by received by the Korean Won he had. Step 2: So, the exchange rate is: 140 USD / 280000 KRW  = 0.0005. Step 3: Therefore, the exchange rate is 0.0005 Korean Won to 1 US Dollar.\nAnalysis: The solution contains an error in the calculation and interpretation of the exchange rate. The question asks for the exchange rate of Korean Won to US Dollar, but the calculation and final statement are reversed. The correct calculation should be 280000 KRW / 140 USD = 2000 KRW/USD. The exchange rate is 2000 Korean Won to 1 US Dollar, not 0.0005 Korean Won to 1 US Dollar.\nWill lead to correct final answer? No\n\nQuestion: {problem}\n{prefix}\nAnalysis:"
"""