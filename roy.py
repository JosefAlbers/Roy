import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import faiss
import pandas as pd
import numpy as np
from textwrap import indent, dedent
import re
import subprocess
import shlex
import venv
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.styles import Style
from pygments.lexers.python import Python3Lexer
import argparse
from tqdm.auto import tqdm
from io import StringIO
from datetime import datetime, timedelta
import copy
import types
import inspect

LOG_LEVEL = 5
log_buffer = StringIO()

def get_timestamp():
    dt = datetime.utcnow() + timedelta(hours=9)
    return dt.strftime('%Y%m%d%H%M%S')

def log(s, log_level=5):
    if log_level < LOG_LEVEL+1:
        log_message = f'\n{get_timestamp()}\n\033[{31+log_level}m\n{s}\n\033[0m\n'
        print(log_message)
        log_buffer.write(log_message)

def dump_log(log_file="log.txt"):
    with open(log_file, "a") as file:
        file.write(log_buffer.getvalue())
    log_buffer.truncate(0)
    return log_file


def trace_method(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        log_in = '\n'.join(str(i) for i in args)
        log(f"{func.__name__}() receives:\n{indent(log_in, '    ')}", 3)
        log_out = '\n'.join(str(i) for i in result) if isinstance(result, (list, tuple)) else str(result)
        log(f"{func.__name__}() returns:\n{indent(log_out, '    ')}", 2)
        return result
    return wrapper
    
def process_code_string(s):
    if '>>>' not in s:
        return s
    def replace_line_prefix(match):
        prefix = match.group(1)
        if prefix in [">>> ", "... "]:
            return ""
        return "# " + match.group(0)
    pattern = r"^(>>> |... |\S+.*$)"
    return re.sub(pattern, replace_line_prefix, s, flags=re.MULTILINE)

def extract_code_block(s, is_python):
    s = s.replace('\r', '')
    pattern = r'```(?:\s*(\w+?)\s*\n)?(.*?)```'
    matches = re.findall(pattern, s, re.DOTALL)
    if len(matches) < 1:
        return ''
    code = ''
    for m in matches:
        is_python = identify_lang(m) if is_python is None else is_python
        code += m[1] if is_python else re.sub(r'^(?![!])', '!', m[1], flags=re.MULTILINE)
    return code.rstrip()

def process_markdown_data(df):
    df = df[~df['filepath'].str.contains('/zh/')]
    df['filepath'] = df['filepath'].str[7:]
    df['content'] = df['content'].str[:5000]
    df['retrieved_content'] = df.apply(lambda row: f"{row['filepath'].split('/')[-1]} ({row['filepath']}):\n'''\n{row['content']}...\n'''", axis=1)
    return df

def process_docstr_data(df):
    def truncate_string(row, char_limit, variable_str, constant_str):
        if not (isinstance(row[variable_str], str) and isinstance(row[constant_str], str)):
            return ""
        if len(row[constant_str]) >= char_limit:
            return ""
        trimmed_length = char_limit - len(row[constant_str])
        return row[variable_str][:trimmed_length]
    df = df[df['docstring'].str.contains('```')]
    df = df[~df['filepath'].apply(lambda x: x.split('/')[-1]).str.startswith('TF')]
    df.reset_index(drop=True, inplace=True)
    df['filepath'] = df['filepath'].str[7:].str.rstrip('/.')
    df['root_dir'] = df['filepath'].apply(lambda x: x.split('/')[0])
    df['retrieved_code'] = df['docstring'].apply(extract_code_block, args=(True,)).apply(process_code_string)
    df['docstring'] = df.apply(truncate_string, args=(5000,'docstring','retrieved_code'), axis=1)
    df['retrieved_docstr'] = df.apply(lambda row: f"{row['type']} `{row['filepath'].split('/')[-1]}` ({row['filepath']}):\n'''\n{row['docstring']}...\n'''", axis=1)
    return df

def edit_code_in_terminal(initial_text):
    kb = KeyBindings()
    result = {'text': initial_text}
    @kb.add('s-tab')
    def _(event):
        result['text'] = event.app.current_buffer.text
        event.app.exit()
    style = Style.from_dict({
        '': '#ffad00',
        'prompt': 'bg:#ff0000 #ffff00',
    })
    session = PromptSession(lexer=PygmentsLexer(Python3Lexer), key_bindings=kb, style=style)
    session.prompt('\n--- Press shift+tab when done ---\n', multiline=True, default=initial_text)
    result_text = result['text']
    return result_text

def identify_lang(match): # stub
    if 'py' in match[0]:
        is_python = True
    elif 'sh' in match[0]:
        is_python = False
    else:
        if '!pip install ' in match[1]:
            is_python = True
        elif 'pip install ' in match[1]:
            is_python = False
        else:
            log('Unable to identify code language')
            is_python = True
    return is_python
    
class VirtualEnvironment:
    def __init__(self, time_limit=20, venv_path='venvRoy'):
        self.venv_path = venv_path
        self.time_limit = time_limit
        try:
            if not os.path.exists(self.venv_path):
                venv.EnvBuilder(with_pip=True).create(self.venv_path)
            if os.name == 'nt':
                self.python_executable = os.path.join(venv_path, "Scripts", "python.exe")
                self.pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
            else:
                self.python_executable = os.path.join(venv_path, "bin", "python")
                self.pip_executable = os.path.join(venv_path, "bin", "pip")
            subprocess.run(f'{self.python_executable} -V')
            subprocess.run(f'{self.pip_executable} -V')
        except:
            log("Warning: Failed to create or locate virtual environment. Using default system python and pip.")
            self.python_executable = "python"
            self.pip_executable = "pip"

    def _run_cmd(self, command):
        replacements = {
            "python": self.python_executable,
            "pip": self.pip_executable,
            "pip3": self.pip_executable,
        }
        command_parts = shlex.split(command)
        command_parts = [replacements.get(part, part) for part in command_parts]
        adjusted_command = ' '.join(shlex.quote(part) for part in command_parts)

        try:
            output = subprocess.run(
                adjusted_command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=self.time_limit,
            ).stdout.decode()
        except subprocess.TimeoutExpired:
            output = "TimeoutError: Execution Exceeded Time Limit (Suspected Infinite Loop)"
        except subprocess.CalledProcessError as error:
            output = str(error.stdout.decode()).strip()
        return output

    def _run(self, code_string, script_name="script.py"):
        code_string = code_string.rstrip()
        ls = re.findall(r'^!(.*)$', code_string, re.MULTILINE)
        code_string = re.sub(r'^(!)', r'#\1', code_string, flags=re.MULTILINE)
        with open(script_name, 'w', encoding='utf-8') as f:
            f.write(code_string)
        ls.append(f"python {script_name}")
        return '\n'.join([self._run_cmd(s) for s in ls]).rstrip()

    def execute(self, s, is_python=None, join=True):
        x_in = extract_code_block(s, is_python)
        x_out = self._run(x_in)
        if join is True:
            return '[Code]:\n```python\n{x_in}\n```\n\n[Output]:\n```\n{x_out}\n```\n'.format(x_in=x_in, x_out=x_out)
        return f'```python\n{x_in}\n```', f'```\n{x_out}\n```'

class RM:
    def __init__(self, configs=None, model_id="BAAI/bge-small-en", query_instruction='Represent this sentence for searching relevant passages: '):
        default_config_for_RM = {
            'markdown': {
                'filename_key': 'hfmd_20230927192215',
                'process_data': process_markdown_data
            },
            'huggingface': {
                'filename_key': 'hfds_20230927191331',
                'process_data': process_docstr_data
            },
        }
        self.configs = default_config_for_RM if configs is None else configs
        self.resources = {}
        for src, config in self.configs.items():
            self._init_filenames(src)
            self._load_resources(src)
        self._init_model(model_id, query_instruction)

    def _init_filenames(self, src):
        config = self.configs[src]
        filename_key = config['filename_key']
        fn_index = f'index_{filename_key}.index'
        fn_df = f'df_{filename_key}.csv'
        self.resources[src] = {
            "fn_index": fn_index,
            "fn_df": fn_df
        }

    def _load_resources(self, src):
        res = self.resources[src]
        for fn_i in [res["fn_index"], res["fn_df"]]:
            hf_hub_download(
                repo_id="Accede/vecDB",
                filename=fn_i,
                repo_type='dataset',
                local_dir='.'
            )
        res["index"] = faiss.read_index(res["fn_index"])
        res["df"] = pd.read_csv(res["fn_df"])

    def _init_model(self, model_id, query_instruction):
        self.QUERY_INST = query_instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, device_map='cpu')
        self.device = torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode_queries(self, queries):
        query_formatted = [self.QUERY_INST + queries] if isinstance(queries, str) else ['{}{}'.format(self.QUERY_INST, q) for q in queries]
        query_tokenized = self.tokenizer(query_formatted, padding=True, truncation=True, return_tensors='pt').to(self.device)
        last_hidden_states = self.model(**query_tokenized, return_dict=True).last_hidden_state
        embeddings = last_hidden_states[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu().numpy()

    def retrieve(self, user_request, n_topk=3, src='huggingface', template='Modify the code below to solve this problem: {user_request}\n```python\n{retrieved_code}\n```'):
        config = self.configs[src]
        res = self.resources[src]
        index = res["index"]
        df = res["df"]
        q_embeddings = self._encode_queries([user_request])
        scores, indices = index.search(q_embeddings, n_topk*30)
        df_topk = df.iloc[indices[0]]
        process_func = config.get('process_data')
        if process_func:
            df_topk = process_func(df_topk)
        df_topk = df_topk.iloc[:n_topk]
        df_topk['user_request'] = user_request
        # return df_topk.reset_index(drop=True)
        ls_topk = df_topk.apply(lambda row: template.format(**row), axis=1).tolist()
        return ls_topk

class LM:
    @torch.no_grad()
    def __init__(self, model_id = 'TheBloke/WizardCoder-Python-7B-V1.0-GPTQ', default_prohibitions=True):
        if '-GPTQ' in model_id:
            log('LM(gptq)')
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").eval()
            self.model_device = self.model.device
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        # elif 'microsoft/phi-1' in model_id:
        #     log('LM(phi)')
        #     torch.set_default_device("cuda")
        #     self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
        #     self.model_device = self.model.device
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        # elif '-AWQ' in model_id:
        #     log('LM(awq)')
        #     from awq import AutoAWQForCausalLM
        #     self.model = AutoAWQForCausalLM.from_quantized(model_id, fuse_layers=True, trust_remote_code=False, safetensors=True) # ?eval()
        #     self.model_device = 'cuda'
        #     self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        else:
            log('LM(hf)')
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto").eval()
            # self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()
            self.model_device = self.model.device
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            
        self._init_tokenizer()
        self.default_prohibitions = []
        if default_prohibitions is True:
            self.default_prohibitions += self._subsentence_tokenizer(['\n\n\n', '\n\r\n\n', '\n\r\n\r'])
        elif default_prohibitions is False:
            pass
        else:
            self.default_prohibitions += self._subsentence_tokenizer(default_prohibitions)
        log(f'{self.default_prohibitions=}', 5)
        log(f'{self._lf=}', 5)

    def _init_tokenizer(self):
        if all(len(sublist) == 1 for sublist in self.tokenizer(['\n', '\n\n']).input_ids):
            self._lf = None
        else:
            self._lf = self.tokenizer('\n', add_special_tokens=False).input_ids

    def _subsentence_tokenizer(self, ls):
        if len(ls) < 1:
            return []
        ls = [ls] if isinstance(ls, str) else list(ls)
        if self._lf is None:
            return self.tokenizer(ls).input_ids
        ls = ['\n'+s for s in ls]
        ii = self.tokenizer(ls, add_special_tokens=False).input_ids
        ii = [i[len(self._lf):] for i in ii]
        return ii

    def _subsentence_decoder(self, ls):
        if self._lf is None:
            return self.tokenizer.decode(ls)
        return self.tokenizer.decode([self._lf[-1]] + list(ls))[1:]
    
    @torch.no_grad()
    def _constrained_beam(self, input_beam, constraint, prohibitions, num_beams, norm_factor = .0, patience_limit = 10):
        max_new_tokens, required_tokens = constraint
        required_tokens_pt = [torch.tensor(i).unsqueeze(0).to(self.model_device) for i in required_tokens]
        beams = [(input_beam[0], [], 0.0, input_beam[3])]
        best_postfixed = (torch.cat((input_beam[0], torch.tensor(required_tokens[0]).unsqueeze(0).to(self.model_device)), dim=1), required_tokens[0], input_beam[2], input_beam[3])
        patience = float('inf')
        for i in range(max_new_tokens):
            if patience < 0:
                break
            else:
                patience -= 1
            new_beams = []
            for beam in beams:
                beam_input_ids, beam_output_tokens, beam_score, beam_kv = beam
                new_outputs = self.model(beam_input_ids, use_cache=True, past_key_values=beam_kv)
                new_logits = new_outputs.logits[:, -1, :]
                new_kv = new_outputs.past_key_values
                topk = torch.topk(new_logits, num_beams)
                list_next_token_id = topk.indices[0]
                list_next_score = topk.values[0]
                for next_token_id, next_score in zip(list_next_token_id, list_next_score):
                    new_input_ids = next_token_id.unsqueeze(0).unsqueeze(0)
                    new_output_tokens = beam_output_tokens + [next_token_id.item()]
                    new_score = ((beam_score * (len(beam_output_tokens) + norm_factor)) + next_score.item()) / (len(new_output_tokens) + norm_factor)
                    if all(new_output_tokens[-len(p):] != p for p in prohibitions) and (next_token_id != self.tokenizer.eos_token_id):
                        new_beam = (new_input_ids, new_output_tokens, new_score, new_kv)
                        if any(new_beam[1][-len(sublist):] == sublist for sublist in required_tokens):
                            if new_beam[2] > best_postfixed[2]:
                                patience = patience_limit
                                best_postfixed = new_beam
                        else:
                            new_beams.append(new_beam)
                            new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:num_beams]
            beams = new_beams
        return best_postfixed

    @torch.no_grad()
    def _unconstrained_beam(self, input_beam, max_new_tokens, prohibitions, num_beams, norm_factor = .0, patience_limit = 10):
        beams = [(input_beam[0], [], 0.0, input_beam[3])]
        best_eos = (None, None, float('-inf'), None)
        patience = float('inf')
        for i in range(max_new_tokens):
            if patience < 0:
                break
            else:
                patience -= 1
            new_beams = []
            for beam in beams:
                beam_input_ids, beam_output_tokens, beam_score, beam_kv = beam
                new_outputs = self.model(beam_input_ids, use_cache=True, past_key_values=beam_kv)
                new_logits = new_outputs.logits[:, -1, :]
                new_kv = new_outputs.past_key_values
                topk = torch.topk(new_logits, num_beams)
                for next_token_id, next_score in zip(topk.indices[0], topk.values[0]):
                    new_input_ids = next_token_id.unsqueeze(0).unsqueeze(0)
                    new_output_tokens = beam_output_tokens + [next_token_id.item()]
                    new_score = ((beam_score * (len(beam_output_tokens) + norm_factor)) + next_score.item()) / (len(new_output_tokens) + norm_factor)
                    if (next_token_id == self.tokenizer.eos_token_id) and (new_score > best_eos[2]):
                        best_eos = beam
                        patience = patience_limit
                    elif all(new_output_tokens[-len(p):] != p for p in prohibitions):
                        new_beams.append((new_input_ids, new_output_tokens, new_score, new_kv))
                        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:num_beams]
            beams = new_beams
        result = best_eos if best_eos[1] else beams[0]
        return result


    def _get_constraints(self, template, default_padding=1, default_interval=500):
        if len(template) == 1:
            if isinstance(template[0], int):
                return [(template[0], [])]
            return [(default_padding, self._subsentence_tokenizer(template[0]))]
        template = list(template)
        template = [default_padding] + template if not isinstance(template[0], int) else template
        template = template + [''] if isinstance(template[-1], int) else template
        fixed_template = []
        expect_int = True
        for i in template:
            if (expect_int is True):
                if (isinstance(i, int)):
                    fixed_template.append(i)
                    expect_int = False
                else:
                    fixed_template.extend([default_interval, i])
                    expect_int = True
            else:
                fixed_template.append(i)
                expect_int = True
        assert len(fixed_template) % 2 == 0
        constraints = [(fixed_template[i], self._subsentence_tokenizer(fixed_template[i+1])) for i in range(0, len(fixed_template), 2)]
        return constraints

    def _get_prohibitions(self, ls):
        if ls is None:
            return self.default_prohibitions
        ls = ls + [' '+i for i in ls if not i[0].isspace()]
        return self.default_prohibitions + self._subsentence_tokenizer(ls)

    @torch.no_grad()
    def _generate(self, input_txt, constraints, prohibitions, num_beams):
        log(f'{constraints=}\n{prohibitions=}')
        input_ids = self.tokenizer(input_txt, add_special_tokens=True, return_tensors='pt').input_ids
        beam = (input_ids.to(self.model_device), [], .0, None)
        result = []
        for constraint in constraints:
            if len(constraint[1]) < 1:
                beam = self._unconstrained_beam(beam, max_new_tokens = constraint[0], prohibitions=prohibitions, num_beams=num_beams)
                to_decode = beam[1]
                result.append(to_decode)
            else:
                beam = self._constrained_beam(beam, constraint = constraint, prohibitions=prohibitions, num_beams=num_beams)
                to_decode = beam[1]
                result.extend([[to_decode[:-len(p)], p] for p in constraint[1] if to_decode[-len(p):] == p][0])
            # print(self._subsentence_decoder(to_decode), '\n---------\n')# debug
        result = [self._subsentence_decoder(i) for i in result]
        return result
        
    @torch.no_grad()
    def generate(self, input_txt, template = (('\n```python', '\n```sh'), '\n```'), constraints = None, prohibitions = None, num_beams = 3, join = True):
        constraints = self._get_constraints(template) if constraints is None else constraints
        prohibitions = self._get_prohibitions(prohibitions)
        result = self._generate(input_txt, constraints, prohibitions, num_beams)
        torch.cuda.empty_cache()
        if join is True:
            return ''.join(result)
        return result

class Roy:
    def __init__(self, config=None):
        if config is None:
            config = {}
        self.template = config.get('template', "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:")
        self._venv = config.get('venv', None)
        self._lm = config.get('lm', None)
        self._rm = config.get('rm', None)

    def format(self, instruction, data=None):
        if data is None:
            data={}
        template = self.template.format(instruction=instruction.rstrip())
        if len(data) < 1:
            return template
        elif isinstance(data, pd.DataFrame):
            return data.apply(lambda row: template.format(**row), axis=1).tolist()
        elif isinstance(data, (dict, pd.Series)):
            return template.format(**data)
        else:
            raise ValueError("Unsupported data type. Data must be a dict, Series, or DataFrame.")

    def add_tool(self, fxn, key = None):
        key = fxn.__name__ if key is None else key
        setattr(self, key, types.MethodType(fxn, self))

    @property
    def venv(self):
        if self._venv is None:
            self._venv = VirtualEnvironment()
        return self._venv

    @property
    def lm(self):
        if self._lm is None:
            self._lm = LM()
        return self._lm

    @property
    def rm(self):
        if self._rm is None:
            self._rm = RM()
        return self._rm

    @trace_method
    def execute(self, *args, **kwargs):
        return self.venv.execute(*args, **kwargs)

    @trace_method
    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)

    @trace_method
    def retrieve(self, *args, **kwargs):
        return self.rm.retrieve(*args, **kwargs)

class Roys(Roy):
    def create(self, agents, tools=None):
        df_agents = pd.DataFrame(agents.items(), columns=['name', 'signature'])
        df_agents['chopchop'] = df_agents['signature'].apply(lambda x: [item.strip() for item in re.split(r'[=()]', x) if item.strip()])
        df_agents['in'] = df_agents['chopchop'].apply(lambda x: x[-1] if x else None)
        df_agents['to'] = df_agents['chopchop'].apply(lambda x: x[0] if x else None)
        df_agents['fxn'] = df_agents['chopchop'].apply(lambda x: x[1:-1] if x else None)
        df_agents = df_agents.drop(columns = ['chopchop'])
        self.df_agents = df_agents
        if tools is not None:
            for key, val in tools.items():
                if 'self' in inspect.signature(val).parameters:
                    self.add_tool(val, key)
                else:
                    setattr(self, key, trace_method(val))

    def _map_fxn(self, ls_fxn, ls_i):
        ls_i = [ls_i] if isinstance(ls_i, str) else ls_i
        ls_o = []
        for i in ls_i:
            t = i
            for f in ls_fxn[::-1]:
                t = getattr(self, f)(t)
            if isinstance(t, list):
                ls_o.extend(t)
            elif isinstance(t, str):
                ls_o.append(t)
            else:
                continue
        return ls_o

    def start(self, requests):
        self.dict_cache = {key: [value] if isinstance(value, str) else value for key, value in requests.items()}
        for turn in range(2):
            if '_' in self.dict_cache:
                break
            log(f'Turn {turn}', 1)
            snapshot = copy.deepcopy(self.dict_cache)
            for _, row_agent in self.df_agents.iterrows():
                key_i = row_agent['in']
                key_o = row_agent['to']
                ls_fxn = row_agent['fxn']
                if key_i in snapshot.keys():
                    agent_output = self._map_fxn(ls_fxn, snapshot[key_i])
                    self.dict_cache[key_o] = agent_output
                    _log = '\n'.join(agent_output)
                    log(f"<<<{row_agent['name']}>>>:\n{_log}", 0)
        return self.dict_cache
