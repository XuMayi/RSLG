from vllm import SamplingParams
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def llm_runner_chat(llm, lora, inputs_all, temperature, max_length_cot, n=1, stop="\n\nQ:"):

    temperature=temperature

    sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_length_cot, n=n,stop=stop)
    prompts_re_generate = []
    for input in inputs_all:
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": "What evidence do we need to answer the question given the current evidence?" + "\n" + input
            },
        ]

        prompts_re_generate.append(conversation)


    preds = []
    outputs_re_generate = llm.chat(prompts_re_generate, sampling_params,lora_request=lora)
    if n == 1:
        for test_idx, output in enumerate(outputs_re_generate):
            prompt = output.prompt
            pred = output.outputs[0].text
            preds.append(pred)
    else:
        for test_idx, output in enumerate(outputs_re_generate):
            pred_ = []
            for idx_n in range(n):
                pred = output.outputs[idx_n].text
                pred_.append(pred)
            preds.append(pred_)
    return preds

def llm_runner_chat2(llm, inputs_all, temperature, max_length_cot, n=1, stop="\n\nQ:"):

    temperature=temperature

    sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_length_cot, n=n,stop=stop)
    prompts_re_generate = []
    for input in inputs_all:
        conversation = [
            {
                "role": "system",
                "content": "You are a helpful assistant"
            },
            {
                "role": "user",
                "content": input
            },
        ]

        prompts_re_generate.append(conversation)


    preds = []
    outputs_re_generate = llm.chat(prompts_re_generate, sampling_params)
    if n == 1:
        for test_idx, output in enumerate(outputs_re_generate):
            prompt = output.prompt
            pred = output.outputs[0].text
            preds.append(pred)
    else:
        for test_idx, output in enumerate(outputs_re_generate):
            pred_ = []
            for idx_n in range(n):
                pred = output.outputs[idx_n].text
                pred_.append(pred)
            preds.append(pred_)
    return preds