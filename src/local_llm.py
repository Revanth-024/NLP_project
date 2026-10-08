"""Small replaceable local generation adapters. No RAG/model downloading here."""

class LocalGenerationError(RuntimeError):
    """Readable generation failure; retrieval remains independently usable."""


class RecordingTestLLM:
    """Deterministic TEST ONLY backend; fixed text, never an application fallback.

    Records the exact prompt for structural integration tests. It does not infer
    answers or validate evidence. Its response must be labelled as simulated.
    """
    def __init__(self,response='TEST BACKEND: simulated response; no model inference.'):
        self.response=response
        self.prompts=[]
    def generate(self,prompt):
        if not isinstance(prompt,str) or not prompt.strip():
            raise ValueError('A non-empty prompt is required.')
        self.prompts.append(prompt)
        return self.response


class TransformersLocalLLM:
    """Starter CPU adapter: model/tokenizer supplied by the resource helper.

    Replaces only generate(); rag.py remains independent of backend details.
    Render one user message with the checkpoint's chat template. Do not silently
    truncate an oversized prompt. Decode only newly generated tokens.
    """
    def __init__(self,model,tokenizer,*,max_new_tokens=96,max_input_tokens=3072):
        self.model=model
        self.tokenizer=tokenizer
        self.max_new_tokens=max_new_tokens
        self.max_input_tokens=max_input_tokens
    def generate(self,prompt):
        if not isinstance(prompt,str) or not prompt.strip():
            raise LocalGenerationError('Enter a non-empty prompt.')
        import torch
        try:
            inputs=self.tokenizer.apply_chat_template([{'role':'user','content':prompt}],
                add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
            length=inputs['input_ids'].shape[1]
            if length>self.max_input_tokens or length+self.max_new_tokens>self.model.config.max_position_embeddings:
                raise LocalGenerationError(f'Prompt has {length} generation-model tokens; input budget is {self.max_input_tokens}. Reduce context explicitly; no silent truncation.')
            with torch.inference_mode():
                output=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=False,
                    num_beams=1,use_cache=True,pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(output[0,length:],skip_special_tokens=True).strip()
        except LocalGenerationError:
            raise
        except Exception as error:
            raise LocalGenerationError(f'CPU generation failed: {type(error).__name__}: {error}') from error
