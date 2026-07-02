def build_prompt(prompt_or_builder, input_text):
    if callable(prompt_or_builder):
        return prompt_or_builder(input_text)
    return prompt_or_builder.replace("{input}", input_text)


def llm_judge(input_text, output_text, llm):
    judge_prompt = f"""
Evaluate the response to a customer support request based on:
- Correctness: addresses the request with a plausible action or guidance
- Clarity: concise and easy to follow
- Instruction following: matches any requested format and tone

Score from 1 to 10. Decimals are allowed. Return only a number. No reasoning.

Customer message:
{input_text}

Response:
{output_text}
"""
    score = llm.invoke(judge_prompt).content
    return score


def test_harness(prompt_or_builder, inputs, llm):
    results = []

    for inp in inputs:
        prompt = build_prompt(prompt_or_builder, inp)
        output = llm.invoke(prompt).content

        score = llm_judge(inp, output, llm)
        results.append({
            "input": inp,
            "output": output,
            "score": score
        })

    return results


def average_score(results):
    return sum(float(r["score"]) for r in results) / len(results)


def self_checking_prompt(task, llm):
    prompt = f"""
You are an expert assistant.

Step 1: Solve the task.
Step 2: Critically review your answer for correctness, clarity, and completeness.
Step 3: Provide a refined final answer.

Task:
{task}

Return in this format:
Initial Answer:
Critique:
Final Answer:
"""
    response = llm.invoke(prompt)
    return response.content


def iterative_refinement(task, llm, iterations=3):
    answer = ""

    for _ in range(iterations):
        prompt = f"""
Improve the following answer:

Task: {task}

Current Answer:
{answer}

Provide a better version.
"""
        response = llm.invoke(prompt)
        answer = response.content

    return answer


def run_prompt_on_dataset(prompt_or_builder, inputs, llm):
    results = []

    for inp in inputs:
        prompt = build_prompt(prompt_or_builder, inp)
        response = llm.invoke(prompt).content

        results.append({
            "input": inp,
            "output": response
        })

    return results


def get_failures(results, threshold=8.5):
    return [r for r in results if float(r["score"]) <= threshold]
