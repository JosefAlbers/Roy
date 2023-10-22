# This file contains codes adapted from:
# - abacaj's code-eval (https://github.com/abacaj/code-eval) 
# - OpenAI's human-eval (https://github.com/openai/human-eval)

# Copyright (c) abacaj
# Licensed under The MIT License (https://github.com/abacaj/code-eval/blob/main/LICENSE)
# Copyright (c) OpenAI
# Licensed under The MIT License (https://github.com/openai/human-eval)

import glob
import torch
import gzip
import json
from tqdm.auto import tqdm
import re
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
import numpy as np
from typing import Iterable, Optional, Callable, Dict
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile

CWD = os.getcwd()
        
def read_problems(evalset_file):
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}

def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))

def custom_sort_key(key):
    parts = key.split('/')
    return (parts[0], int(parts[1]))

def generate_raw(fx, debug, eval_file):
    out_path = os.path.join(CWD, 'generated.jsonl')
    problems = read_problems(eval_file)
    samples = []
    pbar = tqdm(total=len(problems))

    sorted_keys = sorted(problems.keys(), key=custom_sort_key)
    if debug is not None:
        numerator, denominator = debug
        sublists_idx = [sorted_keys[i:i + len(sorted_keys)//denominator] for i in range(0, len(sorted_keys), len(sorted_keys)//denominator)]
        list_id = sublists_idx[numerator]
    else:
        list_id = sorted_keys

    for task_id in list_id:
        print(task_id)
        prompt = problems[task_id]["prompt"]

        batch_completions = [fx(prompt)]

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]
        pbar.update(1)

    write_jsonl(out_path, samples)
    return out_path


def extract_code(eval_file):

    in_path = os.path.join(CWD, 'generated.jsonl')
    out_path = os.path.join(CWD, 'extracted.jsonl')
    
    problems = read_problems(eval_file)

    output = []
    a = 0
    codes = [c for c in stream_jsonl(in_path)]
    for code in codes:
        task_id = code["task_id"]
        prompt = problems[task_id]["prompt"]
        completion = code["completion"]
        completion = completion.replace("\r", "")
        if "```python" in completion:
            def_line = completion.index("```python")
            completion = completion[def_line:].strip()
            completion = completion.replace("```python", "")
            try:
                next_line = completion.index("```")
                completion = completion[:next_line].strip()
            except:
                a += 1
                print(completion)
                print("================\n")
        if '__name__ == "__main__"' in completion:
            next_line = completion.index('if __name__ == "__main__":')
            completion = completion[:next_line].strip()

        if "# Example usage" in completion:
            next_line = completion.index("# Example usage")
            completion = completion[:next_line].strip()

        code["completion"] = completion

    output += codes

    write_jsonl(out_path, output)
    print(a)
    return out_path

def check_correctness(problem: Dict, completion: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.

    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Construct the check program and run it.
            check_program = (
                problem["prompt"] + '\n' + completion + "\n" +
                # completion + '\n' +
                problem["test"] + "\n" +
                f"check({problem['entry_point']})"
            )
            print(check_program)

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        print(exec(check_program, exec_globals))
                print('PASS')
                result.append("passed")
            except TimeoutException:
                print('TIMEOUT')
                result.append("timed out")
            except BaseException as e:
                print('FAIL')
                print(e)
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"],
        passed=result[0] == "passed",
        result=result[0],
        completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield

@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname

class TimeoutException(Exception):
    pass

class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'

@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    problem_file,
    k = [1],
    n_workers = 1,
    timeout = 5.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    sample_file = os.path.join(CWD, 'extracted.jsonl')
    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        # assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm(combine_results(), total=n_samples))
    print(f'{pass_at_k=}')
    print(f'{total=}')
    print(f'{correct=}')

    correct_file = 'correct.npy'
    np.save(correct_file, correct)

    return out_file, correct_file

def evaluate(fx, debug=None, eval_file='heval/HumanEval.jsonl.gz'):
    raw_file = generate_raw(fx, debug=debug, eval_file=eval_file)
    code_file = extract_code(eval_file=eval_file)
    result_file, correct_file = evaluate_functional_correctness(problem_file=eval_file)
    return [raw_file, code_file, result_file, correct_file]

