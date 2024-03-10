from classification import *
from torch.profiler import profile, record_function, ProfilerActivity

args = dict(
    model_name='roberta-base',
    num_labels=2,
    type='prefix'
)

device = 'cuda'
model = CustomModelforSequenceClassification(**args).to(device)
loss_fn = nn.CrossEntropyLoss()
inputs = torch.randint(0, 100, (1, 128), dtype=torch.long).to(device)
attention_mask = torch.ones(1, 128, dtype=torch.long).to(device)
target = torch.randint(0, 2, (1,), dtype=torch.long).to(device)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             profile_memory=True, record_shapes=True) as prof:
    with record_function("forward"):
        output = model(input_ids=inputs, attention_mask=attention_mask)
        loss = loss_fn(output['logits'], target)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=4))
