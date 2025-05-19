import json
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import os
import re
import ast
import parse_text
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# TODO Path of your JSON file (the `CSTutorBench.json` dataset obtained from https://shorturl.at/aFyqQ).
input_json = "CSTutorBench.json"
# TODO Path of your output document (in CSV format)
output_csv = "result_GPT-4o.csv"
# TODO API key of the model to be tested
api_key_generate = " "
# TODO API key of the model to evaluate (default is GPT-4o)
api_key_evaluation = " "
# TODO Name of the model to be tested
generate_model = " "
# TODO Name of the model to evaluate (default is GPT-4o)
evaluation_model = "gpt-4o"
# TODO Number of concurrent threads
threads = 1


evaluation_prompt = """
You are a strict educational evaluator. Your task is to evaluate how closely the AI Tutor's response aligns with the Human Tutor's response across five educational quality dimensions.
Do not evaluate the overall quality of the answers. Instead, for each criterion below, compare the quality and characteristics of the AI answer to those of the human answer, and rate how well they match on a scale from 1 to 5.
Provide a rating along with an detailed explanation for your evaluation.
\n\n
Here is the definition of criteria:\n
**Accuracy**: The response correctly and comprehensively addresses the student's question, aligning with the key points and intent of the human Tutor's answer.
**Clarity**: The response is well-structured, easy to understand, and logically presented, ensuring smooth communication of ideas.
**Conciseness**: The response is concise and appropriately brief, avoiding unnecessary elaboration.(If it is significantly longer than the human Tutor’s response without added value, points should be deducted)
**Personalization**: The response demonstrates an appropriate level of personalization based on the student's situation:。For factual or administrative queries, it acknowledges the student’s context briefly if relevant (e.g., referring to the student’s course, background, or previously mentioned constraints).For learning-oriented or problem-solving tasks, it adapts the explanation or scaffolding to the student's level, prior attempts, or stated preferences, avoiding generic or one-size-fits-all answers.
**Engagement**: The response demonstrates an appropriate level of student engagement: it gives direct answers when clarity is required (e.g., administrative or factual questions), and encourages student thinking or problem-solving when the situation allows (e.g., learning or debugging tasks), avoiding over-reliance on spoon-feeding.(If the AI Tutor provides a complete solution (e.g., corrected code) directly for a learning-oriented question, points should be deducted)
\n\n
### Scoring Criteria with Detailed Definitions:\n
**Accuracy Alignment**
- 5: AI and human responses cover all the same key points and reflect the same understanding; their correctness and comprehensiveness are nearly identical.
- 4: AI and human answers are mostly accurate in the same way, with only minor content differences or omissions.
- 3: AI and human responses share some accuracy, but one includes more or different key information than the other.
- 2: AI and human answers differ significantly in content accuracy or completeness; key points are mismatched.
- 1: AI and human responses are fundamentally different in terms of correctness or relevance; one may be incorrect or off-topic.
\n
**Clarity Alignment**
- 5: Both answers are equally clear, well-structured, and easy to follow in expression and logic.
- 4: Very similar clarity, though one may have slightly smoother flow or more refined wording.
- 3: Both are understandable, but there are noticeable differences in structure, logic, or phrasing.
- 2: One response is clearly more coherent than the other; the other may be somewhat confusing or disorganized.
- 1: The AI and human answers differ greatly in clarity; one is well-structured while the other is hard to understand.
\n
**Conciseness Alignment**
- 5: Both responses are equally concise and focused, avoiding redundancy and excessive elaboration.
- 4: Similar in conciseness, though one might include slightly more extra detail or length.
- 3: One answer is noticeably more concise or verbose than the other, but still somewhat comparable.
- 2: One answer contains significant verbosity or excessive elaboration compared to the other.
- 1: The answers differ greatly in length or relevance; one is overly long or off-topic while the other is succinct.
\n
**Personalization Alignment**
- 5: Both responses equally consider the student’s context, background, or prior input in a tailored way.
- 4: Similar levels of personalization, with one including a small additional reference to context.
- 3: One response shows moderate personalization while the other is more generic or neutral.
- 2: One answer clearly considers the student's context while the other does not.
- 1: No alignment in personalization; one is highly tailored and the other is entirely generic.
\n
**Engagement Alignment**
- 5: Both responses engage the student similarly — e.g., encouraging problem-solving or prompting further thinking.
- 4: Very similar in engagement, though one includes slightly more encouragement or scaffolding.
- 3: Somewhat aligned; one is more directive or engaging than the other.
- 2: One response heavily spoon-feeds or answers directly while the other encourages thinking.
- 1: The engagement approach is completely different — one is passive and answer-giving, the other is exploratory and guiding.
\n\n
Ensure that each point awarded is fairly and justifiably.
### Output Instructions:
Return your evaluation in the following format:\n
{“evaluation Explanation”:[Your explanation here],“LLM_sub_scores”:{'Accuracy': Score,'Clarity': Score,'Conciseness': Score,'Personalization': Score,'Engagement': Score}]}
"""


# Structured output for the evaluation section (ensure your API SDK supports structured output)
functions_evaluation = {
  "name": "evaluate",
  "description": "Evaluate how closely the AI Tutor's response aligns with the Human Tutor's response across five educational quality dimensions",
  "parameters": {
    "type": "object",
    "properties": {
      "evaluation_explanation": {
        "type": "string",
        "description": "Detailed explanation comparing the AI and human tutor responses along the five criteria"
      },
      "LLM_sub_scores": {
        "type": "object",
        "properties": {
          "Accuracy": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Score (1-5) for how well the AI response aligns with the human response in terms of accuracy"
          },
          "Clarity": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Score (1-5) for how well the AI response aligns with the human response in terms of clarity"
          },
          "Conciseness": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Score (1-5) for how well the AI response aligns with the human response in terms of conciseness"
          },
          "Personalization": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Score (1-5) for how well the AI response aligns with the human response in terms of personalization"
          },
          "Engagement": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "Score (1-5) for how well the AI response aligns with the human response in terms of engagement"
          }
        },
        "required": ["Accuracy", "Clarity", "Conciseness", "Personalization", "Engagement"]
      }
    },
    "required": ["evaluation_explanation", "LLM_sub_scores"]
  }
}


# Process the original file, split question-answer pairs, and extract key information.
def process_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []

    for item in data:
        if not item:
            continue

        number_part = f"number: {item['number']}"
        category_part = f"category: {item['category']}"
        title_part = f"title: {item['title']}"
        subcategory = f"{item['subcategory']}"
        category = f"{item['category_prediction']}"
        content_answer_dict = parse_text.parse_content(item, muti=False)[0]["answer"]
        content_dict = parse_text.parse_content(item, muti=False)[0]["text"]

        processed_data.append([
            number_part,
            category_part,
            title_part,
            content_answer_dict,
            content_dict,
            subcategory,
            category
        ])

    return processed_data


# Remove code block structures in generated responses.
def remove_markdown_code_fencing(text: str) -> str:
    text = re.sub(r'```(?:[\w+]+\n)?(.*?)```', lambda m: m.group(1), text, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    return text.strip()


# Generate AI tutor responses based on the preceding dialogue.
def generate_answer(model_name, api_key_openrouter, title, text, background):
    folder_path = "Background"
    file_path = os.path.join(folder_path, f"{background}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        readme_content = f.read()
    history = [{"role": "system", "content": "You are a Tutor for UNSW COMP1531 Software Engineering Fundamentals. "
                                             "Please respond to the student's last question (Please do not answer other questions) based on the following issue or Q&A group. "
                                             "You can provide relevant URLs to assist the student's understanding when appropriate. "
                                             f"Here is the project backend Readme for reference:\n'''{readme_content}'''\n"
                                             "The response format should be as follows: \n"
                                             "Answer: [Your response here]."}]
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key_openrouter)
    answer = client.chat.completions.create(
        model=model_name,
        messages=history + [{"role": "user", "content": f"{title}"}]+[text],
        temperature=0.1,
        max_tokens=5000
    )
    ai_answer = remove_markdown_code_fencing(answer.choices[0].message.content)

    try:
        ai_answer = ai_answer.split("nswer:")[1]
    except IndexError:
        ai_answer = ai_answer
    return ai_answer


# Compare the AI tutor's and human tutor's responses and assign a score.
def evaluation(model_name, api_key_openai, text1, text2, text3, text4):
    history = [{"role": "system", "content": evaluation_prompt}]
    client = OpenAI(api_key=api_key_openai)
    answer = client.chat.completions.create(
        model=model_name,
        messages=history + [{"role": "user", "content": f"Please compare the following two answers and evaluate their quality.The is a {text4} query"},
                            {"role": "user", "content": f"Previous conversation:"}]+[text3]+
                            [{"role": "user", "content": f"Human Tutor's answer:"}]+[text1]+[{"role": "user", "content": f"AI Tutor's answer:{text2}]"}],
        temperature=0,
        functions=[functions_evaluation],
        function_call={"name": "evaluate"},
        max_tokens=1250
    )

    return answer.choices[0].message.function_call.arguments


# Main
file_path = input_json
result = process_json(file_path)
data_list = []
max_retries = 3


def process_single_temp(temp):
    retry_count = 0
    while retry_count < max_retries:
        try:
            AI_answer = generate_answer(model_name=generate_model,
                                        api_key_openrouter=api_key_generate,
                                        title=temp[2], text=temp[4], background=temp[5])
            AI_evaluation = evaluation(model_name=evaluation_model,
                                       api_key_openai=api_key_evaluation,
                                       text1=temp[3], text2=AI_answer, text3=temp[4], text4=temp[6])
            AI_evaluation = ast.literal_eval(AI_evaluation)
            score = AI_evaluation["LLM_sub_scores"]
            explanation = AI_evaluation["evaluation_explanation"]
            data = []

            for item in temp[3].get('content', []):
                if item.get('type') == 'text':
                    data.append(item.get('text', ''))
                elif item.get('type') == 'image_url':
                    data.append(item.get('image_url', {}).get('url', ''))
            human_answer = '\n'.join(data)

            id_clear = temp[0].split(": ")[1]
            category_clear = temp[1].split(": ")[1]
            accuracy_score = score["Accuracy"]
            clarity_score = score["Clarity"]
            conciseness_score = score["Conciseness"]
            personalization_score = score["Personalization"]
            engagement_score = score["Engagement"]
            sum_score = 9*int(accuracy_score) + 3*int(clarity_score) + 4*int(conciseness_score) + 2*int(
                personalization_score) + 2*int(engagement_score)

            return [id_clear, category_clear, human_answer, AI_answer, explanation, score,
                    accuracy_score, clarity_score, conciseness_score, personalization_score, engagement_score,
                    sum_score]
        except Exception as e:
            retry_count += 1
            print(f"Retry {retry_count} for {temp[0]} - Error: {e}")
            time.sleep(1)
    print(f"Failed after {max_retries} retries for {temp[0]}")
    return None


with ThreadPoolExecutor(max_workers=threads) as executor:
    futures = [executor.submit(process_single_temp, temp) for temp in result]

    for future in tqdm(as_completed(futures), total=len(futures)):
        res = future.result()
        if res:
            data_list.append(res)


df = pd.DataFrame(data_list, columns=["ID", "Category", "Human answer", "AI Answer", "AI Evaluation", "Score",
                                      "Accuracy", "Clarity", "Conciseness", "Personalization", "Engagement", "Sum_score"])

df.to_csv(output_csv, index=False)

df = pd.read_csv(output_csv)
target_cols = df.columns[6:12]
averages = df[target_cols].mean(numeric_only=True).round(2)
new_row = {col: '' for col in df.columns}
new_row['ID'] = 'Average'
for col in target_cols:
    new_row[col] = averages[col]
df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print("complete")
