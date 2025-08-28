# 代码评估工具使用说明

本项目包含用于执行 CCEval 和 RepoBench 代码生成评估任务的脚本。以下是关于如何配置和运行这些评估的详细说明。

## 目录结构概览

- `lm-evaluation-harness/`: 包含评估框架和任务定义。
  - `run_cceval.sh`: 用于运行 CCEval 多语言评估的脚本。
  - `run_repobench.sh`: 用于运行 RepoBench 评估的脚本。
  - `lm_eval/tasks/cceval/`: CCEval 任务的 YAML 配置文件。
  - `lm_eval/tasks/repobench/`: RepoBench 任务的 YAML 配置文件。
- `output/`: 评估结果将输出到此目录。

## CCEval 评估

CCEval 评估支持多种编程语言（如 Python, Java, C# 等）。评估通过 `run_cceval.sh` 脚本启动。

### 修改参数

1. **模型参数**:
   - 打开 `run_cceval.sh` 脚本。
   - 修改 `MODEL_ARGS` 变量以匹配你的模型配置。例如，如果你使用不同的模型名称或 API 端点，需要在此处进行更改。
     ```bash
     MODEL_ARGS="model=your-model-name,base_url=your-api-endpoint,num_concurrent=1,max_retries=10,tokenized_requests=False"
     ```

2. **支持的语言**:
   - 在 `run_cceval.sh` 脚本中，`LANGUAGES` 数组定义了要评估的语言。
     ```bash
     LANGUAGES=("python" "java" "csharp" "typescript")
     ```
   - 如果你需要添加或删除语言，直接修改此数组即可。确保对应的任务 YAML 文件存在于 `lm_eval/tasks/cceval/` 目录中。

3. **任务 YAML 文件**:
   - CCEval 任务的配置文件位于 `lm_eval/tasks/cceval/` 目录下，文件名格式为 `cceval_${language}_*.yaml`。
   - 每个 YAML 文件定义了一个特定的评估任务。你可能需要根据实际情况修改这些文件中的参数，例如数据集路径、提示模板等。

### 运行评估

确保你在 `lm-evaluation-harness` 目录下，然后执行以下命令：

```bash
bash run_cceval.sh
```

评估结果将保存在 `output/${language}` 目录中。

## RepoBench 评估

RepoBench 评估通过 `run_repobench.sh` 脚本启动。

### 修改参数

1. **模型参数**:
   - 打开 `run_repobench.sh` 脚本。
   - 修改 `--model_args` 参数以匹配你的模型配置。
     ```bash
     --model_args model=your-model-name,base_url=your-api-endpoint,num_concurrent=1,max_retries=10,tokenized_requests=False
     ```

2. **任务 YAML 文件**:
   - RepoBench 任务的配置文件位于 `lm_eval/tasks/repobench/` 目录下。
   - 你可能需要根据实际情况修改这些文件中的参数。

### 运行评估

确保你在 `lm-evaluation-harness` 目录下，然后执行以下命令：

```bash
bash run_repobench.sh
```

评估结果将保存在 `output/` 目录中。

## 注意事项

- 确保所有路径配置正确，特别是任务 YAML 文件的路径。
- 在运行评估之前，请确认模型服务已启动并且可以通过指定的 `base_url` 访问。
- 脚本中使用了 `--confirm_run_unsafe_code` 参数，这意味着评估过程中可能会执行不安全的代码。请确保在受控环境中运行。