import os
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.append(str((Path.cwd() / "../../UtilsYF").resolve()))
CACHE_DIR = (HERE / "../../Cache").resolve()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["VLLM_USE_FLASH_ATTENTION"] = "false"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_DIR)
os.environ["HF_HOME"] = str(CACHE_DIR)

from transformers import AutoTokenizer
from normal_utils import load_openai_key, setup_openai, setup_gemini,setup_huggingface
from google.genai import types

class Runner:
    def __init__(self, backend: str = "openai"):
        b = backend.lower()
        if b == "openai":
            self.impl = self._OpenAIImpl()
        elif b == "gemini":
            self.impl = self._GeminiImpl()
        elif b == "qwen":
            self.impl = self._QwenImpl()
        elif b == "llama":
            self.impl = self._LlamaImpl()
        else:
            raise ValueError("backend must be one of: openai | gemini | qwen")

    def set_model(self, **kwargs):
        return self.impl.set_model(**kwargs)

    def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
        return self.impl.query(system_prompt, user_prompt, max_tokens)
    
    def query_multi_turn(self, messages: list, max_tokens: int = 512) -> str:
        return self.impl.query_multi_turn(messages, max_tokens)
    


    # -------------------- OpenAI --------------------
    class _OpenAIImpl:
        def __init__(self):
            self.model = "gpt-4.1-mini"
            self.client = None

        def set_model(self, **kwargs):
            from openai import OpenAI
            setup_openai()
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            return self  # 允许链式

        def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
            if self.client is None:
                raise RuntimeError("Call get_model() first.")
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_output_tokens=max_tokens,
            )
            return resp.output_text
        def query_multi_turn(self, messages: list, max_tokens: int = 512) -> str:
            if self.client is None:
                raise RuntimeError("Call get_model() first.")
            resp = self.client.responses.create(
                model=self.model,
                input=messages,
                max_output_tokens=max_tokens,
            )
            return resp.output_text
        
        def query_mm(self, system_prompt: str, user_prompt: str, image_url: str, max_tokens: int = 512) -> str:
            if self.client is None:
                raise RuntimeError("Call get_model() first.")

            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": user_prompt},
                            {"type": "input_image", "image_url": image_url}, 
                        ],
                    },
                ],
                max_output_tokens=max_tokens,
            )
            return resp.output_text
    # -------------------- Gemini --------------------
    class _GeminiImpl:
        def __init__(self):
            self.model = "gemini-2.5-flash"
            self.client = None

        def set_model(self, **kwargs):
            from google import genai
            setup_gemini()
            self.client = genai.Client()
            return self

        def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 512) -> str:
            if self.client is None:
                raise RuntimeError("Call get_model() first.")
            resp = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction= system_prompt,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    maxOutputTokens= max_tokens),
                contents=user_prompt
            )
            return resp.text
        
        def query_multi_turn(self, messages: list, max_tokens: int = 512) -> str:
                if self.client is None:
                    raise RuntimeError("Call set_model() first.")
                
                chat_contents = []
                system_instruction = ""

                for msg in messages:
                    if msg["role"] == "system":
                        # 提取系统指令
                        system_instruction = msg["content"]
                    else:
                        # 转换角色：assistant -> model
                        role = "user" if msg["role"] == "user" else "model"
                        chat_contents.append(
                            types.Content(
                                role=role,
                                parts=[types.Part(text=msg["content"])]
                            )
                        )

                # 使用 generate_content 处理多轮历史
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=chat_contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        max_output_tokens=max_tokens,
                    )
                )
                return resp.text
    # -------------------- Qwen (vLLM 本地) --------------------
    class _QwenImpl:
        def __init__(self):
            self.model_id = "Qwen/Qwen3-8B"
            self.tok = None
            self.llm = None
            self.SamplingParams = None
            setup_huggingface()

        def set_model(self, **kwargs):
            from vllm import LLM, SamplingParams
            self.SamplingParams = SamplingParams
            self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            self.llm = LLM(
                model=self.model_id,
                dtype="bfloat16",
                tensor_parallel_size=1,
                max_model_len=kwargs.get("max_length", 4096),
                trust_remote_code=False
            )
            return self

        def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
            if self.llm is None or self.tok is None:
                raise RuntimeError("Call get_model() first.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            outs = self.llm.generate([prompt], self.SamplingParams(max_tokens=max_tokens, temperature=0.0),use_tqdm=False)
            return outs[0].outputs[0].text.strip()
        
        def query_multi_turn(self, messages: list, max_tokens: int = 256) -> str:
            if self.llm is None or self.tok is None:
                raise RuntimeError("Call get_model() first.")
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            outs = self.llm.generate([prompt], self.SamplingParams(max_tokens=max_tokens, temperature=0.0),use_tqdm=False)
            return outs[0].outputs[0].text.strip()
        

    class _LlamaImpl:
        def __init__(self):
            self.model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.tok = None
            self.llm = None
            self.SamplingParams = None
            setup_huggingface()

        def set_model(self, **kwargs):
            from vllm import LLM, SamplingParams
            self.SamplingParams = SamplingParams
            self.tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            self.llm = LLM(
                model=self.model_id,
                dtype="bfloat16",
                tensor_parallel_size=1,
                max_model_len=kwargs.get("max_length", 4096),
                trust_remote_code=False
            )
            return self

        def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 256) -> str:
            if self.llm is None or self.tok is None:
                raise RuntimeError("Call get_model() first.")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outs = self.llm.generate([prompt], self.SamplingParams(max_tokens=max_tokens, temperature=0.0),use_tqdm=False)
            return outs[0].outputs[0].text.strip()
        
        def query_multi_turn(self, messages: list, max_tokens: int = 256) -> str:
            if self.llm is None or self.tok is None:
                raise RuntimeError("Call get_model() first.")
            prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outs = self.llm.generate([prompt], self.SamplingParams(max_tokens=max_tokens, temperature=0.0),use_tqdm=False)
            return outs[0].outputs[0].text.strip()