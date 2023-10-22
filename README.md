# Roy

[<img src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/JosefAlbers/Roy/blob/main/quickstart.ipynb)
[![DOI](https://zenodo.org/badge/699801819.svg)](https://zenodo.org/badge/latestdoi/699801819)

Roy is a lightweight alternative to `autogen` for developing advanced multi-agent systems using language models. It aims to simplify and democratize the development of emergent collective intelligence.

## Features

- **Model Agnostic**: Use any LLM, no external APIs required. Defaults to a 4-bit quantized wizard-coder-python model for efficiency.

- **Modular and Composable**: Roy decomposes agent interactions into reusable building blocks - templating, retrieving, generating, executing.

- **Transparent and Customizable**: Every method has a clear purpose. Easily swap out components or add new capabilities.

## Quickstart

```sh
git clone https://github.com/JosefAlbers/Roy
cd Roy
pip install -r requirements.txt
pip install -U transformers optimum accelerate auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

```python
from roy import Roy, Roys
roy = Roy()
s = '"What date is today? Which big tech stock has the largest year-to-date gain this year? How much is the gain?'
roy.generate(roy.format(s))
```

### **Rapid Benchmarking**

Roy provides a simple way to evaluate and iterate on your model architecture.. This allows you to:

- Easily swap out components, such as language models, prompt formats, agent architectures, etc

- Benchmark on different tasks like arithmetic, python coding, etc (default is OpenAI's HumanEval)

- Identify agent's areas of strengths and weaknesses

```python
from Roy.util import piecewise_human_eval

# Comparing different language models
piecewise_human_eval(0, lm_id='TheBloke/WizardCoder-Python-7B-V1.0-GPTQ') 
# -> {'pass@1': 0.6341463414634146}
piecewise_human_eval(0, lm_id='TheBloke/tora-code-7B-v1.0-GPTQ') 
# -> {'pass@1': 0.5609756097560976}
piecewise_human_eval(0, lm_id='TheBloke/Arithmo-Mistral-7B-GPTQ')
# -> {'pass@1': 0.5121951219512195}

# Testing a custom agent architecture
piecewise_human_eval(0, fx=<your_custom_Roy_agent>)
```

*Takes around 30 minutes each on a free Google Colab runtime.*

### **Constrained Beam Search**

Use templates to structure conversations (control output length, format, etc)

```python
roy.generate(s, ('\n```python', '\n```'))                      # Generate a python code block
roy.generate(s, (('\n```python', '\n```javascript'), '\n```')) # Generate python or javascript codes
roy.generate(s, ('\n```python', 100, '\n```'))                 # Generate a code block of size less than 100 tokens
```

### **Retrieval Augmented Generation**

Enhance generation with relevant knowledge.

```python
s = 'Create a text to image generator.'
r = roy.retrieve(s, n_topk=3, src='huggingface')
[roy.generate(s) for s in r]
```

### **Auto-Feedback**

Agents collaborate in tight loops to iteratively refine outputs to specification.

```python
# Iterative multiturn chat style
s = "Create a secure and unique secret code word with a Python script that involves multiple steps to ensure the highest level of confidentiality and protection.\n"
for i in range(2):
    c = roy.generate(s, prohibitions=['input'])
    s += roy.execute(c)

# Another way to implement auto-feedback
user_request = "Compare the year-to-date gain for META and TESLA."
ai_response = roy.generate(user_request, ('\n```python', ' yfinance', '\n```'))
for i in range(2):
    shell_execution = roy.execute(ai_response)
    if 'ModuleNotFoundError' in shell_execution:
        roy.execute(roy.generate(roy.format(f'Write a shell command to address the error encountered while running this Python code:\n\n{shell_execution}')))
    elif 'Error' in shell_execution:
        ai_response = roy.generate(roy.format(f'Modify the code to address the error encountered:\n\n{shell_execution}'))
    else:
        break
```

### **Multi-Agent**

Flexible primitives to build ecosystems of agents.

```python
roys = Roys()

# AutoFeedback
roys.create(agents = {'Coder': 'i = execute(generate(i))'})
roys.start(requests = {'i': 'Create a mobile application that can track the health of elderly people living alone in rural areas.'})

# Retrieval Augmented Generation
roys.create(
    agents = {
        'Retriever': 'r = retrieve(i)',
        'Generator': 'o = generate(r)',
        })
roys.start(requests = {'i': 'Create a Deutsch to English translator.'})

# Providing a custom tool to one of the agents using lambda
roys.create(
    agents = {
        'Coder': 'c = generate(i)',
        'Proxy': 'c = custom(execute(c))',
        },
    tools = {'custom': lambda x:f'Modify the code to address the error encountered:\n\n{x}' if 'Error' in x else None})
roys.start(requests = {'i': 'Compare the year-to-date gain for META and TESLA.'})

# Another way to create a custom tool for agents
def custom_switch(self, c):
    py_str = 'Modify the code to address the error encountered:\n\n'
    sh_str = 'Write a shell command to address the error encountered while running this Python code:\n\n'
    x = self.execute(c)
    if 'ModuleNotFoundError' in x:
        self.execute(self.generate(sh_str+x))
    elif 'Error' in x:
        self.dict_cache['i'] = [py_str+x]
    else:
        return '<<<Success>>>:\n\n'+x
    
roys.create(
    agents = {
        'Coder': 'c = generate(i)',
        'Proxy': '_ = protocol(c)',
        },
    tools = {'protocol': custom_switch})
roys.start(requests = {'i': 'Compare the year-to-date gain for META and TESLA.'})
```

## Emergent Multi-Agent Dynamics

Roy aims to facilitate the emergence of complex, adaptive multi-agent systems. It draws inspiration from biological and AI concepts to enable decentralized coordination and continual learning.

- **Survival of the Fittest** - Periodically evaluate and selectively retain high-performing agents based on accuracy, speed etc. Agents adapt through peer interactions.

- **Mixture of Experts** - Designate agent expertise, dynamically assemble specialist teams, and route tasks to optimal experts. Continuously refine and augment experts. 

These mechanisms facilitate the emergence of capable, adaptive, and efficient agent collectives.

## Get Involved

Roy is under active development. We welcome contributions - feel free to open issues and PRs!

## Support the Project

If you found this project helpful or interesting and want to support more of these experiments, feel free to buy me a coffee!

<a href="https://www.buymeacoffee.com/albersj66a" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="25" width="100"></a>
