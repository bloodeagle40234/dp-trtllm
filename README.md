# dp-trtllm
Dummy Plug for NVIDIA TensorRT-LLM used by transformers pipeline.

## Installation
```bash
git clone https://github.com/bloodeagle40234/dp-trtllm
cd dp-trtllm
pip install .
```

## Usage
Create DummyPlug instance via TensorRT-LLM engine directory, then
pass the instance to transformers pipeline

```python
from transformers import AutoTokenizer
from transformers.pipelines import pipeline

tokenizer = AutoTokenizer.from_pretrained(
    "<base model path>", device_map="cuda")
model = DummyPlug("<engine path>")

pipe = pipeline(
    "text-generation",
    model=model,
    framework="pt",
    tokenizer=tokenizer,
    eos_token=tokenizer.eos_token_id,
    pad_token=tokenizer.pad_token_id,
    max_new_tokens=1024,
    device_map="cuda"
)

pipe("Hello, how are you?")
```

### Example: llm-jp-eval
If you want to use DummyPlug for evaluation tool like llm-jp-eval,
set DummyPlug class for model target. Note that addtional configigurations
other than TensorRT-LLM's build config (e.g. dtype, device) will be ignored.


```yaml
model:
  _target_: dp_trtllm.DummyPlug
  pretrained_model_name_or_path: <engine path>
```


## References
- [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [transformers](https://github.com/huggingface/transformers/tree/main)
- [llm-jp-eval](https://github.com/llm-jp/llm-jp-eval)
