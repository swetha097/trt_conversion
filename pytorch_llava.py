import torch
from transformers import pipeline
from PIL import Image
from transformers import BitsAndBytesConfig
import os

def generate_image_text_response(image_path, text_prompt, model_id="llava-hf/llava-1.5-7b-hf"):
    """Generates a text response for an image and a given prompt."""

    quantization_config = BitsAndBytesConfig(load_in_4bit=False, bnb_4bit_compute_dtype=None, llm_int8_enable_fp32_cpu_offload=True)
    pipe = pipeline("image-text-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config}, device_map="auto", torch_dtype=torch.float32)

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_prompt}, ]}]
    output = pipe(text=messages, max_length=2024, max_new_tokens=1024)
    return output[0]["generated_text"]

def process_images_and_prompts(image_paths, prompts, output_file="image_text_responses.txt"):
    """Processes multiple images with multiple prompts and saves responses to a file."""

    with open(output_file, "w") as f:
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            f.write(f"Image: {image_path}\n")

            for prompt in prompts:
                print(f"  Prompt: {prompt}")
                response = generate_image_text_response(image_path, prompt)
                if response:
                    f.write(f"    Prompt: {prompt}\n")
                    f.write(f"    Response: {response}\n")
                else:
                    f.write(f"    Prompt: {prompt}\n")
                    f.write("    Response: Could not generate response.\n")
            f.write("-" * 80 + "\n")

image_paths = [
    "/home/mcw/swetha/new_models/PromptMRG/data/iu_xray/iu_xray/images/CXR3536_IM-1729/0.png",
    "/home/mcw/swetha/new_models/PromptMRG/data/iu_xray/iu_xray/images/CXR3536_IM-1729/1.png",
]

prompts = [
    "Write a radiologic report on the given chest radiograph, including information about cardiomegaly, lung opacity, lung lesion, edema, focal consolidation, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, fractured and enlarged cardiomediastinum.",
    "What kind of disease is shown in the image?",
    "Describe the abnormalities in this chest X-ray.",
]

process_images_and_prompts(image_paths, prompts)