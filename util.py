from huggingface_hub import login, hf_hub_download, HfApi
from inspect import getsource
from .roy import Roy, LM, log, dump_log, get_timestamp
from .human_eval import evaluate

LM_ID = 'TheBloke/WizardCoder-Python-7B-V1.0-GPTQ' 
HF_CFG = None

def grind(self, request):
    num_beams = 3
    num_iter = 3

    prompt_template = '{system}\n\n### Instruction:\n{query}\n\n### Response:'
    prompt_code = prompt_template.format(system='', query='{request}').lstrip()
    prompt_test = prompt_template.format(system='Create a Python script to verify the correctness of the code.', query='[Problem]:\n{request}\n\n[Code]:\n{code}')
    prompt_debug = prompt_template.format(system='Identify the cause of the error and debug the code.', query='[Problem]:\n{request}\n\n[Code]:\n{code}\n\n[Error]:\n{out}')

    template_code = (50, '\n```python', 300, '\n```')
    template_test = (50, '\n```python\nassert', 100, '\n```')
    template_debug = (50, '\n```python', 300, '\n```')

    prohibitions_test = ['input']
    prohibitions_debug = ['\n```Python']

    cache={}
    cache['request'] = f'```python\n{request.strip()}\n```'
    cache['code'] = ''.join(self.generate(prompt_code.format(**cache), template=template_code, join=False, num_beams=num_beams)[1:])
    cache['test'] = ''.join(self.generate(prompt_test.format(**cache), template=template_test, prohibitions=prohibitions_test, join=False, num_beams=num_beams)[1:])

    for i in range(num_iter):
        _, cache['out'] = self.execute('{request}\n{code}\n{test}'.format(**cache), join=False)
        if 'Error' in cache['out']:
            cache['code'] = ''.join(self.generate(prompt_debug.format(**cache), template=template_debug, prohibitions=prohibitions_debug, join=False, num_beams=num_beams)[1:])
        else:
            break

    final_code, final_out = self.execute('{request}\n{code}\n{test}'.format(**cache), join=False)
    if 'Error' in final_out:
        print(f'FAIL\n{final_code}\n{final_out}')
    else:
        print('PASS')

    return cache['code']

def aggregate_human_eval_results(key, d, hf_cfg=HF_CFG):
    if hf_cfg is None:
        print('Huggingface user token, repo ID, and file key are required.')
        return None
    login(hf_cfg['token'])
    correct_files = [f'{key}_{n}_{d}_correct.npy' for n in range(d)]
    downloaded_files = []
    for fn_i in correct_files:
        try:
            hf_hub_download(
                repo_id=hf_cfg['repo'],
                filename=fn_i,
                repo_type='dataset',
                local_dir='.'
            )
            downloaded_files.append(fn_i)
        except:
            print(f'{fn_i} does not exist.')
    list_npy = []
    for fn_i in downloaded_files:
        _npy = np.load(fn_i)
        print(f'{np.sum(_npy)/len(_npy)} = {np.sum(_npy)}/{len(_npy)} for {fn_i}')
        list_npy.append(_npy)
    all_npy = np.concatenate(list_npy)
    all_scr = np.sum(all_npy)/len(all_npy)
    print(f'{all_scr} = {np.sum(all_npy)}/{len(all_npy)} for ALL')
    return downloaded_files, list_npy, all_npy, all_scr

def piecewise_human_eval(n=0, d=4, lm_id=LM_ID, fx=grind, hf_cfg=HF_CFG):
    key = hf_cfg.get('key', None)
    log(f'{key}: {n}/{d}\n{lm_id=}\n----- BEGIN -----')
    roy = Roy({'lm': LM(lm_id)})
    roy.add_tool(fx, 'fx')
    list_files = evaluate(roy.fx, (n,d))
    log(f'{key}: {n}/{d}\n{lm_id=}\n{getsource(fx)}\n----- END -----', 0)
    list_files.append(dump_log())
    if hf_cfg is None:
        return list_files
    api = HfApi(token=hf_cfg['token'])
    for file_i in list_files:
        api.upload_file(
            path_or_fileobj=file_i,
            path_in_repo=hf_cfg['key']+file_i,
            repo_id=hf_cfg['repo'],
            repo_type='dataset'
        )

