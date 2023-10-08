# Roy

[<img src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/github/JosefAlbers/Roy/blob/main/quickstart.ipynb)
[![DOI](https://zenodo.org/badge/699801819.svg)](https://zenodo.org/badge/latestdoi/699801819)

Roy is a lightweight alternative to `autogen`, for crafting advanced multi-agent systems using language models.

## Features

- Flexibility: Roy is model-agnostic, eliminating the need for external API keys. By default, it employs the 4-bit quantized wizard-coder-python, but you can swap it out for any LLM of your choice.

- Composability: Roy refines the LLM interactions into modular, adaptable operations, each of which can be used in isolation or composed in myriad ways to create sophisticated workflows.

- Clarity: Opposed to frameworks that entangle operations behind multiple layers, Roy is transparent. Each method serves a distinct purpose, for instance, granting users complete oversight and control over the process.

## Quickstart

```sh
git clone https://github.com/JosefAlbers/Roy
cd Roy
pip install -r requirements.txt
pip install -U transformers optimum accelerate auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

```python
from roy import Roy
roy = Roy()
```

### **Template-Based Generation**

```python
s = '"What date is today? Which big tech stock has the largest year-to-date gain this year? How much is the gain?\n'
s = roy.format(s)
roy.generate(s)
```

### **Constrained Generation**

```python
roy.generate(s, ('```python', '```'))                    # Generate a python code block
roy.generate(s, (('```python', '```javascript'), '```')) # Generate python or javascript codes
roy.generate(s, ('```python', 100, '```'))               # Generate a code block of size less than 100 tokens
```

### **Retrieval Augmented Generation**

```python
s = 'Create a text to image generator.\n'
r = roy.retrieve(s, n_topk=3, src='huggingface')
r = roy.format('Modify the [Example Code] to fulfill the [User Request] using minimal changes. Keep the modifications minimal by making only the necessary modifications.\n\n[User Request]:\n"{user_request}"\n\n[Context]:\n{retrieved_docstr}\n\n[Example Code]:\n```python\n{retrieved_code}\n```', r)
[roy.generate(s) for s in r]
```

### **Auto Feedback**

```python
s = "Create a secure and unique secret code word with a Python script that involves multiple steps to ensure the highest level of confidentiality and protection.\n"

for i in range(3):
    c = roy.generate(s)
    s += roy.execute(c)
```

### **Auto Grind**

```python
user_request = "Compare the year-to-date gain for META and TESLA.\n"
ai_response = roy.generate(user_request)
for i in range(3):
    shell_execution = roy.execute(ai_response)
    if 'ModuleNotFoundError' in shell_execution:
        roy.execute(roy.generate(roy.format(f'Write a shell command to address the error encountered while running this Python code:\n\n{shell_execution}')), False)
    elif 'Error' in shell_execution:
        ai_response = roy.generate(roy.format(f'Modify the code to address the error encountered:\n\n{shell_execution}'))
    else:
        break
```

## Self-Organizing Multi-Agent System

Envision a dynamic group chat where agents, each bearing distinct roles and defined by its unique constraints, affinities, and behaviors, converge to collaborates towards a predefined objective. This opens up a realm of possibilities, from brainstorming sessions to problem-solving think tanks. Drawing inspiration from biology and machine learning, Roy aspires to pioneer a perpetually evolving multi-agent environment.

### Survival of the Fittest

Guided by the tenets of natural selection, this principle ensures the survival of only the most adept agents:

- **Fitness Metrics**: Evaluate agents based on information quality, response speed, relevance, and feedback.
  
- **Agent Evaluation**: Periodically assess agent performance.
  
- **Agent Reproduction**: Allow top-performing agents to spawn new ones with similar attributes.
  
- **Agent Removal**: Phase out underperformers to uphold quality.
  
- **Adaptive Learning**: Agents evolve through their interactions.
  
- **Dynamic Environment**: The chat setting adapts, pushing agents to be resilient and versatile.

### Mixture of Experts

This principle promotes efficiency by allocating tasks to the most qualified agents:

- **Expertise Definition**: Designate areas of expertise for each agent.
  
- **Dynamic Collaboration**: Align relevant agents to address specific tasks.
  
- **Expert Evaluation**: Constantly gauge the performance of expert agents.
  
- **Expert Refinement**: Retrain, adjust, or replace underperforming experts.
  
- **Learning from Peers**: Agents expand their knowledge horizons by learning from others.
  
- **Task Routing**: Route tasks to the best-suited experts.
  
- **Layered Expertise**: Deploy coordinators to guide collaborations for intricate tasks.
  
- **Human Expert Integration**: Infuse human knowledge into the chat, amplifying collective intelligence.

By utilizing these principles, our self-organizing chat group remains in a state of perpetual evolution, always questing for the zenith of outcomes and ongoing enhancement.

## Conclusion

Roy redefines the paradigm of LLM application development, emphasizing simplicity, versatility, and transparency. Whether your aim is a basic LLM interaction or an elaborate multi-agent system, Roy provides the architecture to realize it.

For those who've made it this far, I believe you share my passion. Dive into Roy, discover its potential, and join me in exploring the future of LLM applications.

If you found this project helpful or interesting and want to support more of these experiments, feel free to buy me a coffee!

<a href="https://www.buymeacoffee.com/albersj66a" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="25" width="100"></a>

