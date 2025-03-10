from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM
import re
from trl import GRPOConfig, GRPOTrainer

dataset_id = 'openai/gsm8k'
ds = load_dataset(dataset_id, 'main')
train_dataset = ds['train']
test_dataset = ds['test']

prompt = """<|start_header_id|>system<|end_header_id|>
You are a helpful assistant solving math questions. Always answer in most accurate way.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there
will be 21 trees. How many trees did the grove workers plant today?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have
been 21 - 15 = 6. The answer is 6.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they
had 74 - 35 = 39. The answer is 39.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did
Jason give to Denny?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.
The answer is 8.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he
have now?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.
The answer is 9.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: There were nine computers in the server room. Five more computers were installed each day, from monday
to thursday. How many computers are now in the server room?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20
computers were added. 9 + 20 is 29. The answer is 29.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf
balls did he have at the end of wednesday?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he
had 35 - 2 = 33 golf balls. The answer is 33.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
<|eot_id|><|start_header_id|>assistant<|end_header_id|> 
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23
- 15 is 8. The answer is 8.
<|eot_id|><|start_header_id|>user<|end_header_id|> 
Q: {query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def prepare_data(data):
    return {"prompt": prompt.replace("{query}", data['question'])}

train_dataset = train_dataset.map(prepare_data)
test_dataset = test_dataset.map(prepare_data)
print(train_dataset[0])

model_id = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
)

FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
    )

def parse_response(response):
    text = response.replace("\n", " ")
    text = text.split('The answer is')[-1].strip()
    pred_answer_list = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
    if len(pred_answer_list) == 0:
        return None
    else:
        pred_answer = pred_answer_list[0]
        if pred_answer[-1] == ".":
            pred_answer = pred_answer[:-1]
        pred_answer = pred_answer.replace(",", "")
        pred_answer = pred_answer.replace("$", "")
        return pred_answer

def accuracy_reward(completions, **kwargs):
    solutions = kwargs['answer']
    completion_contents = [completion for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = solution.split('####')[-1].strip()
        answer_parsed = parse_response(content)
        if answer_parsed is not None:
            try:
                rewards.append(float(float(gold_parsed) == float(answer_parsed)))
            except Exception:
                rewards.append(-1.0)
        else:
            rewards.append(-1.0)
    return rewards

run_name = "sppo_off_policy"

training_args = GRPOConfig(
    run_name=run_name,
    output_dir=f"experiment/{run_name}",
    learning_rate=1e-7,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=8,
    bf16=True,
    temperature=0.35,
    use_sppo=True,
    per_device_train_batch_size=4,
    beta=0.0,

    # Parameters that control data preprocessing
    max_completion_length=512, # default: 256
    num_generations=1, # default: 8
    max_prompt_length=1024, # default: 512

    # Parameters related to reporting and saving
    report_to=["wandb"],
    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
)


trainer = GRPOTrainer(
    model=model,
    reward_funcs=[accuracy_reward],
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()
trainer.save_model(training_args.output_dir)
