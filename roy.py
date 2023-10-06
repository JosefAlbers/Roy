import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
import faiss
import pandas as pd
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
import sys
from tqdm.auto import tqdm

LOG_LEVEL = 5

def log(s, log_level=5):
    if log_level < LOG_LEVEL+1:
        print(f'\033[{31+log_level}m{s}\033[0m')

def trace_method(func):
    def wrapper(*args, **kwargs):
        input_str = args[0] if isinstance(args[0], str) else args[1]
        log(f"{func.__name__}() receives:\n{indent(input_str, '    ')}", 2)
        result = func(*args, **kwargs)
        if isinstance(result, str):
            log(f"{func.__name__}() returns:\n{indent(result, '    ')}", 1)
        else:
            log(f"{func.__name__}() returns an object of type: {type(result)}", 1)
        return result
    return wrapper
        
def extract_code_block(text_w_code, join_all = True):
    pattern = r'```(?:\s*(\w+?)\s*\n)?(.*?)```'
    matches = re.findall(pattern, text_w_code, re.DOTALL)
    if len(matches) < 1:
        return ''
    codes = [i[1] for i in matches]
    if join_all:
        code = '\n\n'.join(codes)
    else:
        code = codes[-1]
    return code
    
def trim_str(row, char_limit, variable_str, constant_str):
    if not (isinstance(row[variable_str], str) and isinstance(row[constant_str], str)):
        return ""
    if len(row[constant_str]) >= char_limit:
        return ""
    trimmed_length = char_limit - len(row[constant_str])
    return row[variable_str][:trimmed_length]

def process_markdown_data(df):
    df = df[~df['filepath'].str.contains('/zh/')]
    df['filepath'] = df['filepath'].str[7:]
    df['content'] = df['content'].str[:5000]
    df['retrieved_content'] = df.apply(lambda row: f"{row['filepath'].split('/')[-1]} ({row['filepath']}):\n'''\n{row['content']}...\n'''", axis=1)
    return df

def process_docstr_data(df):
    df = df[df['docstring'].str.contains('```')]
    df = df[~df['filepath'].apply(lambda x: x.split('/')[-1]).str.startswith('TF')]
    df.reset_index(drop=True, inplace=True)
    df['filepath'] = df['filepath'].str[7:].str.rstrip('/.')
    df['root_dir'] = df['filepath'].apply(lambda x: x.split('/')[0])
    df['retrieved_code'] = df['docstring'].apply(extract_code_block)
    df['docstring'] = df.apply(trim_str, args=(5000,'docstring','retrieved_code'), axis=1)
    df['retrieved_docstr'] = df.apply(lambda row: f"{row['type']} `{row['filepath'].split('/')[-1]}` ({row['filepath']}):\n'''\n{row['docstring']}...\n'''", axis=1)
    return df

def process_string(s):
    if '>>>' not in s:
        return s

    def replace_line_prefix(match):
        prefix = match.group(1)
        if prefix in [">>> ", "... "]:
            return ""
        return "# " + match.group(0)
    
    pattern = r"^(>>> |... |\S+.*$)"
    return re.sub(pattern, replace_line_prefix, s, flags=re.MULTILINE)

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

class VirtualEnvironment:
    def __init__(self, venv_path='venv4gen'):
        self.venv_path = venv_path
        try:
            if not os.path.exists(self.venv_path):
                venv.EnvBuilder(with_pip=True).create(self.venv_path)
            if os.name == 'nt':
                self.python_executable = os.path.join(venv_path, "Scripts", "python.exe")
                self.pip_executable = os.path.join(venv_path, "Scripts", "pip.exe")
            else:
                self.python_executable = os.path.join(venv_path, "bin", "python")
                self.pip_executable = os.path.join(venv_path, "bin", "pip")
        except:
            log("Warning: Failed to create or locate virtual environment. Using default system python and pip.")
            self.python_executable = "python"
            self.pip_executable = "pip"

    def _run_sh(self, command):
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
            ).stdout.decode()
        except subprocess.CalledProcessError as error:
            output = str(error.stdout.decode()).strip()
        return output
    
    def _run_py(self, code_string, script_name="script.py"):
        lines_with_exclamation = re.findall(r'^!(.*)$', code_string, re.MULTILINE)
        code_string = re.sub(r'^(!)', r'#\1', code_string, flags=re.MULTILINE)
        with open(script_name, 'w') as f:
            f.write(code_string)
        lines_with_exclamation.append(f"python {script_name}")
        return '\n' +  '\n'.join([self._run_sh(s) for s in lines_with_exclamation]) + '\n'
    
    def execute(self, code_string, is_python=True):
        code_string = extract_code_block(code_string)
        if is_python is True:
            return self._run_py(code_string)
        return '\n' + '\n'.join([self._run_sh(s) for s in code_string.splitlines()]) + '\n'
        
class RM:
    def __init__(self, configs=default_config_for_RM, model_id="BAAI/bge-small-en", query_instruction='Represent this sentence for searching relevant passages: '):
        self.configs = configs
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
    
    def retrieve(self, user_request, n_topk=3, src='huggingface'):
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
        return df_topk.reset_index(drop=True)

class LM:
    def __init__(self, model_id = 'TheBloke/WizardCoder-Python-7B-V1.0-GPTQ'):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", revision="main")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    @torch.no_grad()
    def _beam(self, input_beam, constraint, prohibits, num_beams, cache_fn, norm_factor = .0):
        fn_to_save = os.path.join(os.getcwd(), cache_fn)
        fn_to_load = os.path.join(os.getcwd(), input_beam[3]) if input_beam[3] is not None else None
        beams = [(input_beam[0].to(self.model.device), [], 0.0, torch.load(fn_to_load) if (fn_to_load is not None) else None )]
        max_new_tokens, required_tokens = constraint
        required_tokens_pt = torch.tensor(required_tokens).unsqueeze(0).to(self.model.device)
        best_postfixed = (None, None, float('-inf'), None)
        best_compatibility = float('-inf')
        for i in range(max_new_tokens):
            best_voluntary = (None, None, float('-inf'), None)
            new_beams = []
            for beam in beams:
                beam_input_ids, beam_output_tokens, beam_score, beam_kv = beam
                new_outputs = self.model(beam_input_ids, use_cache=True, past_key_values=beam_kv)
                new_logits = new_outputs.logits[:, -1, :]
                new_kv = new_outputs.past_key_values
                topk = torch.topk(new_logits, num_beams)
                list_next_token_id = topk.indices[0]
                list_next_score = topk.values[0]
                list_next_token_id = torch.cat((list_next_token_id, required_tokens_pt[:, 0]), dim=0)
                head_score = new_logits[:, required_tokens[0]]
                diff_score = head_score - torch.mean(list_next_score)
                list_next_score = torch.cat((list_next_score, head_score), dim=0)
                if diff_score > best_compatibility:
                    if len(required_tokens) > 1:
                        candidate_output = self.model(required_tokens_pt[:, :-1], use_cache=False, past_key_values=new_kv)
                        rest_score = torch.sum(torch.cat([candidate_output.logits[:,i,required_tokens[i+1]] for i in range(len(required_tokens)-1)]) , dim = 0)
                    else:
                        rest_score = .0
                    force_score = ((beam_score * (len(beam_output_tokens) + norm_factor)) + head_score + rest_score) / (len(beam_output_tokens) + len(required_tokens) + norm_factor)
                    if force_score > best_postfixed[2]:
                        best_postfixed = (required_tokens_pt, beam_output_tokens + required_tokens, force_score, new_kv)
                        best_compatibility = diff_score
                for next_token_id, next_score in zip(list_next_token_id, list_next_score):
                    new_input_ids = next_token_id.unsqueeze(0).unsqueeze(0)
                    new_output_tokens = beam_output_tokens + [next_token_id.item()]
                    new_score = ((beam_score * (len(beam_output_tokens) + norm_factor)) + next_score.item()) / (len(new_output_tokens) + norm_factor)
                    if next_token_id == self.tokenizer.eos_token_id:
                        continue
                    elif all(new_output_tokens[-len(p):] != p for p in prohibits):
                        new_beam = (new_input_ids, new_output_tokens, new_score, new_kv)
                        new_beams.append(new_beam)
                        new_beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:num_beams]
                        if (new_output_tokens[-len(required_tokens):] == required_tokens) and (new_score > best_voluntary[2]):
                            best_voluntary = new_beam
            if best_voluntary[2] >= new_beams[-1][2]:
                torch.save(best_voluntary[-1], fn_to_save)
                return (*best_voluntary[:-1], cache_fn)
            beams = new_beams
        torch.save(best_postfixed[-1], fn_to_save)
        return (*best_postfixed[:-1], cache_fn)
    
    @torch.no_grad()
    def generate(self, input_txt, constraints = [(20, [13, 28956, 4691]), (900, [13, 28956])], prohibits = ([13, 13, 13], [13, 30004, 13, 13], [13, 30004, 13, 30004]), num_beams = 3, cache_fn = 'kv'): 
        os.remove(cache_fn) if os.path.exists(cache_fn) else None
        i_beam = (self.tokenizer(input_txt, add_special_tokens=True, return_tensors='pt').input_ids, None, None, None)
        result = []
        for i, constraint in enumerate(constraints):
            i_beam = self._beam(i_beam, constraint = constraints[i], prohibits=prohibits, num_beams=num_beams, cache_fn=cache_fn)
            torch.cuda.empty_cache()
            result += i_beam[1]
        return self.tokenizer.decode(result)
    
class Roy:
    def __init__(self, config=None):
        if config is None:
            config = {}
            
        self.template = config.get('template', "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:")
        
        self._venv = None
        self._lm = None
        self._rm = None

        self.execute = trace_method(config.get('execute', self.venv.execute))
        self.generate = trace_method(config.get('generate', self.lm.generate))
        self.retrieve = trace_method(config.get('retrieve', self.rm.retrieve))

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
    def format(self, instruction, data={}):
        template = self.template.format(instruction=instruction)
        if isinstance(data, pd.DataFrame):
            return data.apply(lambda row: template.format(**row), axis=1).tolist()
        elif isinstance(data, (dict, pd.Series)):
            return template.format(**data)
        else:
            raise ValueError("Unsupported data type. Data must be a dict, Series, or DataFrame.")
