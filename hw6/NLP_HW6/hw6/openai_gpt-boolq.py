import random
from datasets import load_dataset
from openai import OpenAI


dataset = load_dataset("boolq")

def extract_instances(data, num_yes=4, num_no=4):
    yes_cases = []
    no_cases = []

    for d in data:
        if d['answer']:
            yes_cases.append((d['passage'], d['question'], 1))
        else:
            no_cases.append((d['passage'], d['question'], 0))

    sample_yes = random.sample(yes_cases, num_yes)
    sample_no = random.sample(no_cases, num_no)

    ret = []
    for i in range(min(num_no, num_yes)):
        ret.append(sample_yes[i])
        ret.append(sample_no[i])
    return ret


samples = extract_instances(dataset['train'], 4, 4)

prompt = ''
for d in samples:
    prompt += f"Passage: {d[0]}\nQ: {d[1]}\nA: {d[2]}\n\n"

unseen_instances = random.sample(list(dataset['validation']), 30)
corr = 0

client = OpenAI(api_key="Openai_key_to_be_replaced")
for instance in unseen_instances:
    passage, question, answer = instance['passage'], instance['question'], 1 if instance['answer'] else 0
    full_prompt = prompt + f"Passage: {passage}\nQ: {question}\nA:"
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=full_prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    predict = response.choices[0].text.strip().lower()

    if str(predict) == str(answer):
        corr += 1

accu = corr / len(unseen_instances)
print(f"Evaluation accuracy: {accu:.2f}")
