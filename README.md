# trt_conversion

Use the docker - swethabs/tensorrt_llm for tensorrt conversion (docker compiled with right CUDA arch).


Metric used for Radiology Report Generation:
    - https://github.com/MAGIC-AI4Med/RaTEScore (RaTEScore)
    - https://github.com/rajpurkarlab/CXR-Report-Metric/tree/main (BLEU, BERTScore, Chexbert vector similarity)

Forked repo;
    - https://github.com/swetha097/CXR-Report-Metric/tree/swe_fork - In addition to the above, included ROUGE1 and ROUGEL scores.

Look at the file trt_conversion.md file for CMDs used for conversion.

**Documenting the results in -** 
1.  [llava-cxr](https://multicorewareinc1-my.sharepoint.com/:x:/r/personal/siva_ayyadurai_multicorewareinc_com/_layouts/15/Doc.aspx?sourcedoc=%7B6E98D5B4-C875-5FE2-88FE-41C6EF4958EF%7D&file=VLMs%20Testing.xlsx&action=default&mobileredirect=true&DefaultItemOpen=1&web=1) 
2. [Eval-trt-models](https://multicorewareinc1-my.sharepoint.com/:x:/r/personal/swetha_multicorewareinc_com/_layouts/15/Doc.aspx?sourcedoc=%7BA473D144-2727-4E05-806F-CDF1DF333159%7D&file=Evaluation%20-%20Trt%20models.xlsx&action=default&mobileredirect=true&DefaultItemOpen=1)


