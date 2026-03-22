from model_adapters.base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:

    if 'qwen' in model_path.lower():
        from model_adapters.qwen import QwenModel
        return QwenModel(model_path)
    if 'llama-3' in model_path.lower():
        from model_adapters.llama3 import Llama3Model
        return Llama3Model(model_path)
    elif 'llama' in model_path.lower():
        from model_adapters.llama2 import Llama2Model
        return Llama2Model(model_path)
    elif 'gemma' in model_path.lower():
        from model_adapters.gemma import GemmaModel
        return GemmaModel(model_path)
    elif 'yi' in model_path.lower():
        from model_adapters.yi import YiModel
        return YiModel(model_path)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
