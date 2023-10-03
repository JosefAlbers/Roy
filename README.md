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
```

```python
from roy import Roy
roy = Roy()
```

### **Template-Based Generation**

```python
s = 'Compare the year-to-date gain for META and TESLA.'
s = roy.format(s)
s = roy.generate(s)
```

### **Retrieval Augmented Generation**

```python
s = 'Create a text to image generator.'
r = roy.retrieve(s, n_topk=3, src='huggingface')
r = roy.format('Modify the [Example Code] to fulfill the [User Request] using minimal changes...', r)
[roy.generate(s) for s in r]
```

### **Auto Feedback**

```python
s = 'Find arxiv papers that show how are people studying trust calibration in AI based systems'
for i in range(3):
    c = roy.generate(s)
    s += c
    s += roy.execute(c)
```

### **Auto Grind**

```python
def auto_grind(user_request):
    cache = {'user_request': user_request, 'py_code': roy.generate(user_request)}
    for i in range(3):
        cache['sh_out'] = roy.execute(cache['py_code'])
        if 'Error' in cache['sh_out']:
            feedback = roy.format('Debug the Code ("script.py") that had been written for this problem: "{user_request}"\n\n[Code]:\n```python\n{py_code}\n```\n\n[Error]:\n{sh_out}', cache)
            cache['py_code'] = roy.generate(feedback)
        else:
            break
    return cache

auto_grind("Plot a chart of TESLA's stock price change YTD and save to 'stock_price_ytd.png'.")
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

For those who've made it this far, we believe you share our passion. Dive into Roy, discover its potential, and collaborate with us to sculpt the future of LLM applications.


<a href="https://www.buymeacoffee.com/albersj66a" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>

