#!/bin/bash

# Configuration
MODEL_NAME="llava-1.5-7b-hf"
OUTPUT_CSV="output_test_17_int4_awq_calib_2048.csv"
TRT_LLM_DIR="/media/TensorRT-LLM"
RUN_PY="/media/TensorRT-LLM/examples/multimodal/run.py"

# Ensure the output CSV exists and add headers if it's new
if [ ! -f "$OUTPUT_CSV" ]; then
  echo "Image Path,Prompt,Output Answer" > "$OUTPUT_CSV"
fi

# Function to run the command and extract the answer
run_inference() {
  local image_path="$1"
  local prompt="$2"

  local output=$(mpirun -n 2 --allow-run-as-root python "$RUN_PY" \
    --max_new_tokens 300 \
    --hf_model_dir "tmp/hf_models/${MODEL_NAME}" \
    --visual_engine_dir "tmp/trt_engines/${MODEL_NAME}/vision_encoder/" \
    --llm_engine_dir "tmp/trt_engines/${MODEL_NAME}/int4_awq/calib_data_2048/2-gpu/" \
    --input_text "$prompt" \
    --image_path "$image_path" 2>&1) # Redirect stderr to stdout

  # Extract the answer (adjust extraction logic if needed)
  local answer=$(echo "$output" | grep -oP '\[A]:\s*\K.*')
  echo "In the function run_inference()"
  echo "Raw Output: $output"  

  # echo "Extracted Answer: $answer"
  # Escape commas in the answer for CSV format
  answer=$(echo "$answer" | sed 's/,/\\,/g')

  # Append to CSV
  echo "\"$image_path\",\"$prompt\",\"$answer\"" >> "$OUTPUT_CSV"
}

# Example Inputs (replace with your actual data)
# Example image paths and prompts
image_paths=(
  "/media/pics/test_dataset_17_images/Picture1.jpg"
  "/media/pics/test_dataset_17_images/Picture2.jpg"
  "/media/pics/test_dataset_17_images/Picture3.jpg"
  "/media/pics/test_dataset_17_images/Picture4.png"
  # "/media/pics/test_dataset_17_images/pic0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2386_IM-0942/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2871_IM-1277/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2416_IM-0961/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1524_IM-0339/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1524_IM-0339/1.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1112_IM-0078/1.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR2661_IM-1142/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1013_IM-0013/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR219_IM-0799/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR219_IM-0799/1.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR3848_IM-1946-1001/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR1125_IM-0082/0.png"
  "/media/new_models/kir/PromptMRG/data/iu_xray/iu_xray/images/CXR3536_IM-1729/1.png"

)

prompts=(
  "Given the following list of medical conditions, analyse and carefully classify which are depicted in the image: Conditions such as 'cardiomegaly', 'lung opacity', 'lung lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 'pleural other', 'fractured',  'enlarged cardiomediastinum' or 'no finding' "
  # "Describe the abnormalities in the image, if any."
  # "What are the possible diagnoses based on this X-ray for chest and heart - from conditions such as - Conditions such as 'cardiomegaly', 'lung opacity', 'lung lesion', 'edema', 'consolidation', 'pneumonia', 'atelectasis', 'pneumothorax', 'pleural effusion', 'pleural other', 'fractured',  'enlarged cardiomediastinum' or 'no finding"
  "Input : You will receive a chest X-ray image as input. Output Requirements: Provide a structured radiologic report that includes the following sections - Examination: Specify the type of imaging study (e.g - Chest X-ray, AP view). - Indication: Summarize the reason for the examination - e.g., Evaluation for potential findings such as cardiomegaly, lung opacity, lung lesion, edema, consolidation, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, fractured, enlarged cardiomediastinum or no finding at all. - Findings: Describe all observed abnormalities or normal findings in detail. - Impression: Provide a concise summary of the most critical observations or diagnoses. - If a section does not apply or no information is available, return it as null. Avoid referencing prior studies or comparisons unless explicitly provided in the input. Focus solely on observations from the provided image."
  # Add more prompts here, matching the number of images or providing a default.
  #  "Input : You will receive a chest X-ray image as input. Output Requirements: Provide a structured radiologic report that includes the following sections - Examination: Specify the type of imaging study (e.g - Chest X-ray, AP view). - Indication: Summarize the reason for the examination (e.g., Evaluation for potential findings such as Lung-related: lung opacity, lung lesion, edema, consolidation, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, no finding  - Heart-related: cardiomegaly, enlarged cardiomediastinum. - Other - Fractures). - Findings: Describe all observed abnormalities or normal findings in detail. - Impression: Provide a concise summary of the most critical observations or diagnoses. - If a section does not apply or no information is available, return it as null. Avoid referencing prior studies or comparisons unless explicitly provided in the input. Focus solely on observations from the provided image."
#  "Input : You will receive a chest X-ray image as input. Output Requirements: Provide a structured radiologic report that includes the following sections - Examination: Specify the type of imaging study (e.g - Chest X-ray, AP view). - Indication: Summarize the reason for the examination (e.g., Evaluation for potential findings such as pneumonia, lung opacity, lung lesion, edema, focal consolidation, atelectasis, pneumothorax, pleural effusion, pleural other, cardiomegaly, enlarged cardiomediastinum. - Other - Fractures - or no finding). - Findings: Describe all the observed abnormalities or normal findings in detail. - Impression: Provide a concise summary of the most critical observations or diagnoses. - If a section does not apply or no information is available, tell about not finding the abnormality.Focus solely on observations from the provided image."

  #Lung-related: lung opacity, lung lesion, edema, consolidation, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, no finding.
)

# Run inference for each image and prompt combination 

for prompt in "${prompts[@]}"; do
  for image_path in "${image_paths[@]}"; do
    run_inference "$image_path" "$prompt"
  done
done

echo "Inference completed. Results saved to $OUTPUT_CSV"

 


