# RepoBench 数据文件

此目录应包含 RepoBench 数据集的 JSONL 文件。

## 需要的文件

请将以下文件放在 `python_v1_1/` 子目录中：

- `repobench_python_v1.1-cross_file_first.jsonl` (~501MB)
- `repobench_python_v1.1-cross_file_random.jsonl` (~465MB) 
- `repobench_python_v1.1-in_file.jsonl` (~485MB)

## 下载数据

您可以从以下来源获取 RepoBench 数据集：

1. **官方来源**: [RepoBench GitHub](https://github.com/Leolty/repobench)
2. **Hugging Face**: [RepoBench Dataset](https://huggingface.co/datasets/tianyang/repobench)

## 目录结构

```
repobench/
├── README.md (本文件)
└── python_v1_1/
    ├── repobench_python_v1.1-cross_file_first.jsonl
    ├── repobench_python_v1.1-cross_file_random.jsonl
    └── repobench_python_v1.1-in_file.jsonl
```

## 使用方法

数据文件下载完成后，RepoBench 任务配置文件将自动使用这些数据进行评估。

相关的任务配置文件位于：
- `lm_eval/tasks/repobench/repobench_cross_file_first_down_sampling_200.yaml`
- `lm_eval/tasks/repobench/repobench_cross_file_random_down_sampling_200.yaml`
- `lm_eval/tasks/repobench/repobench_in_file_down_sampling_200.yaml`