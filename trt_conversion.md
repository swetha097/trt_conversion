

**TensorRT conversion**
    Convert Checkpoints to a format required by trt-llm:
        ```python
        python ../llama/convert_checkpoint.py         --model_dir tmp/hf_models/${MODEL_NAME}         --output_dir tmp/trt_models/${MODEL_NAME}/fp16_calib_2gpu         --dtype float16 --tp_size 2 --calib_dataset /media/pics/test_folder/
        ```
   
    Build the tensorrt engine:
        ```python
        trtllm-build         --checkpoint_dir tmp/trt_models/${MODEL_NAME}/fp16_calib_2gpu   --output_dir tmp/trt_engines/${MODEL_NAME}/fp16_calib_2gpu  --gemm_plugin float16 --use_fused_mlp=enable         --max_batch_size 1  --max_input_len 2048         --max_seq_len 2560         --max_multimodal_len 576 --context_fmha disable --profiling_verbosity --workers 2 --monitor_memory --enable_debug_output 

        ```
    **[ NOTE: Make sure to use --context_fmha disable in the cluster20]**

   To run the inference with mutliple gpus (change -n to number of gpus present in the system):
   ```python
   mpirun -n 2 --allow-run-as-root python run.py --max_new_tokens 300         --hf_model_dir tmp/hf_models/${MODEL_NAME}  --visual_engine_dir tmp/trt_engines/${MODEL_NAME}/vision_encoder/  --llm_engine_dir tmp/trt_engines/llava-1.5-7b-hf/fp16_calib_2gpu/ --input_text "Given the following list of medical conditions, classify which is depicted in the image: Conditions such as 'cardiomegaly' or 'lung opacity' or 'lung lesion' or 'edema' or 'consolidation' or 'pneumonia' or 'atelectasis' or 'pneumothorax' or 'pleural effusion' or 'pleural other' or 'fractured' or 'no finding' or 'enlarged cardiomediastinum'] Please analyze the image and select the conditions from above given 14 conditions that would describe the disease's shown in the image.  And generate a radiology report." --image_path /media/pics/Picture2.jpg  --log_level verbose
   ```




**Accuracy Computations with mmlu:**

   export MODEL_NAME="llava-1.5-7b-hf"
   cd examples/multimodal/
   ls
   mkdir data
   tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
   mpirun -n 2 --allow-run-as-root python mmlu.py --hf_model_dir ./tmp/hf_models/llava-1.5-7b-hf/ --engine_dir ./tmp/trt_engines/llava-1.5-7b-hf/fp16/2-gpu/ --test_trt_llm
   mpirun -n 2 --allow-run-as-root python ../mmlu.py --hf_model_dir ./tmp/hf_models/llava-1.5-7b-hf/ --engine_dir ./tmp/trt_engines/llava-1.5-7b-hf/fp16/2-gpu/ --test_trt_llm
   mpirun -n 2 --allow-run-as-root python ../mmlu.py --hf_model_dir ./tmp/hf_models/llava-1.5-7b-hf/ --engine_dir ./tmp/trt_engines/llava-1.5-7b-hf/fp16/2-gpu/ --test_hf

