import torch
from transformers import pipeline
from PIL import Image
from transformers import BitsAndBytesConfig
import os
import csv

def generate_image_text_response(image_path, text_prompt, model_id="llava-hf/llava-1.5-7b-hf"):
    """Generates a text response for an image and a given prompt."""

    quantization_config = BitsAndBytesConfig(load_in_4bit=True) #using 8 bit quantization
    pipe = pipeline("image-text-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config}, device_map="auto", torch_dtype=torch.float32)

    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_prompt}, ]}]
    output = pipe(text=messages, max_length=2024, max_new_tokens=1024)
    return output[0]["generated_text"]

def process_images_and_prompts_to_csv(image_paths, prompts, output_file="image_text_responses_int4.csv"):
    """Processes multiple images with multiple prompts and saves responses to a CSV file."""

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image Path", "Prompt", "Response"])  # Write header

        for prompt in prompts:
            print(f"  Prompt: {prompt}")
            for image_path in image_paths:
                print(f"Processing: {image_path}")
                response = generate_image_text_response(image_path, prompt)
                if response:
                    writer.writerow([image_path, prompt, response])
                else:
                    writer.writerow([image_path, prompt, "Could not generate response."])

# Example usage:
image_paths = [
    "/home/mcw/swetha/pics/test_dataset_17_images/Picture1.jpg",
    "/home/mcw/swetha/pics/test_dataset_17_images/Picture2.jpg",
    "/home/mcw/swetha/pics/test_dataset_17_images/Picture3.jpg",
    "/home/mcw/swetha/pics/test_dataset_17_images/Picture4.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2386_IM-0942/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2871_IM-1277/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2416_IM-0961/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1524_IM-0339/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1524_IM-0339/1.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1112_IM-0078/1.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2661_IM-1142/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1013_IM-0013/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR219_IM-0799/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR219_IM-0799/1.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR3848_IM-1946-1001/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1125_IM-0082/0.png",
    "/home/mcw/swetha/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR3536_IM-1729/1.png",
]

prompts = [
    "Write a radiologic report on the given chest radiograph, including information about cardiomegaly, lung opacity, lung lesion, edema, focal consolidation, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, fractured and enlarged cardiomediastinum.",
    "Given the following list of medical conditions, classify which is depicted in the image: Conditions such as 'cardiomegaly' or 'lung opacity' or 'lung lesion' or 'edema' or 'consolidation' or 'pneumonia' or 'atelectasis' or 'pneumothorax' or 'pleural effusion' or 'pleural other' or 'fractured' or 'no finding' or 'enlarged cardiomediastinum']. Please analyze the image and select the conditions from above given 14 conditions that would describe the disease's shown in the image. And generate a radiology report.",
]

process_images_and_prompts_to_csv(image_paths, prompts)