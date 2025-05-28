# 从魔搭社区下载数据集

[医疗问诊数据_SFT格式 · 数据集](https://modelscope.cn/datasets/BRZ911/Medical_consultation_data_SFT/quickstart)

```python
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('BRZ911/Medical_consultation_data_SFT')
# 链接：https://modelscope.cn/datasets/BRZ911/Medical_consultation_data_SFT/quickstart
```

# 安装LLaMa Factory

```python
# 使用git克隆项目
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
# 安装依赖库
pip install -e ".[torch,metrics,modelscope]"
```

1. 运行 `llamafactory-cli version` 进行验证。若显示当前  LLaMA-Factory 版本，则说明安装成功

```Bash
----------------------------------------------------------
| Welcome to LLaMA Factory, version 0.9.2                |
|                                                        |
| Project page: https://github.com/hiyouga/LLaMA-Factory |
----------------------------------------------------------
```

启动服务开始模型微调

```
llamafactory-cli webui
```

环境变量解释：

- CUDA_VISIBLE_DEVICES：指定使用的显卡序号，默认全部使用
- USE_MODELSCOPE_HUB：使用国内魔搭社区加速模型下载，默认不使用



# 云资源配置

![image-20250527171408503](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250527171408503.png)

# Lora微调Qwen2.5

## 训练

分类任务（数据集159）

```shell
llamafactory-cli train \
--stage sft \
--do_train True \
--model_name_or_path /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
--preprocessing_num_workers 16 \
--finetuning_type lora \
--template qwen \
--flash_attn auto \
--dataset_dir /gemini/code \
--dataset cate-train \
--cutoff_len 2048 \
--learning_rate 5e-05 \
--num_train_epochs 10.0 \
--max_samples 50000 \
--per_device_train_batch_size 5 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type linear \
--max_grad_norm 1.0 \
--logging_steps 10 \
--save_steps 10 \
--warmup_steps 5 \
--packing False \
--enable_thinking True \
--report_to none \
--freeze_vision_tower True \
--freeze_multi_modal_projector True \
--image_max_pixels 589824 \
--image_min_pixels 1024 \
--video_max_pixels 65536 \
--video_min_pixels 256 \
--output_dir /gemini/output/fenlei-checkpoint \
--bf16 True \
--plot_loss True \
--trust_remote_code True \
--ddp_timeout 180000000 \
--include_num_input_tokens_seen True \
--optim adamw_torch \
--lora_rank 16 \
--lora_alpha 16 \
--lora_dropout 0.01 \
--lora_target all
```

SPO任务（数据集500）

```shell
llamafactory-cli train \
--stage sft \
--do_train True \
--model_name_or_path /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
--preprocessing_num_workers 16 \
--finetuning_type lora \
--template qwen \
--flash_attn auto \
--dataset_dir /gemini/code \
--dataset spo-train \
--cutoff_len 2048 \
--learning_rate 5e-05 \
--num_train_epochs 10.0 \
--max_samples 50000 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--logging_steps 1 \
--save_steps 30 \
--warmup_steps 5 \
--packing False \
--enable_thinking True \
--report_to none \
--freeze_vision_tower True \
--freeze_multi_modal_projector True \
--image_max_pixels 589824 \
--image_min_pixels 1024 \
--video_max_pixels 65536 \
--video_min_pixels 256 \
--output_dir /gemini/output/spo-checkpoint \
--bf16 True \
--plot_loss True \
--trust_remote_code True \
--ddp_timeout 180000000 \
--include_num_input_tokens_seen True \
--optim adamw_torch \
--lora_rank 16 \
--lora_alpha 16 \
--lora_dropout 0.01 \
--lora_target all
```

医疗智能问答项目（数据集43万6千）---太慢了，通过max_samples参数裁剪5万吧

```shell
llamafactory-cli train \
--stage sft \
--do_train True \
--model_name_or_path /root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
--preprocessing_num_workers 16 \
--finetuning_type lora \
--template qwen \
--flash_attn auto \
--dataset_dir /gemini/code \
--dataset yiliao \
--cutoff_len 2048 \
--learning_rate 5e-05 \
--num_train_epochs 10.0 \
--max_samples 50000 \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 5 \
--lr_scheduler_type cosine \
--max_grad_norm 1.0 \
--logging_steps 10 \
--save_steps 1000 \
--warmup_steps 1000 \
--packing False \
--enable_thinking True \
--report_to none \
--freeze_vision_tower True \
--freeze_multi_modal_projector True \
--image_max_pixels 589824 \
--image_min_pixels 1024 \
--video_max_pixels 65536 \
--video_min_pixels 256 \
--output_dir /gemini/output/yiliao-checkpoint \
--bf16 True \
--plot_loss True \
--trust_remote_code True \
--ddp_timeout 180000000 \
--include_num_input_tokens_seen True \
--optim adamw_torch \
--lora_rank 16 \
--lora_alpha 16 \
--lora_dropout 0.01 \
--lora_target all
```



## 训练日志（已医疗数据集为例）

数据加载：

![image-20250528134835908](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528134835908.png)

分批情况：![image-20250528134923487](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528134923487.png)

每1000步保存一次

![image-20250528135232850](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528135232850.png)

![image-20250528135512463](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528135512463.png)

参数更新日志

![image-20250528102104687](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528102104687.png)



![image-20250528112855047](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528112855047.png)

最终结果

![image-20250528140121192](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528140121192.png)

## 损失函数图

![image-20250528144723313](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528144723313.png)

观察损失函数，整体处于下降趋势，符合正常现象，

有2个问题：

①原始损失函数曲线（`original`）锯齿状严重的问题。（可优化方向：可增加语料丰富程度）

②损失函数下降不够多（优化方向：可进一步增加训练轮次）



### **常用优化策略**

训练优化问题，通常可从数据处理、训练参数调整、优化器选择等多方面进行优化：

#### 1. **数据层面优化**

**增大 Batch Size**

小批量（如`batch_size=2`）会导致梯度估计方差大，使损失波动剧烈。可逐步增大`batch_size`（如 8-16），并相应调整`gradient_accumulation_steps`以保持总批量不变。

**数据打乱与混洗**

确保数据在每个 Epoch 中充分打乱（`shuffle=True`），避免批次间数据分布偏差过大。

#### 2. **学习率调整**

**降低学习率**
过高的学习率会导致参数更新步长过大，损失函数震荡。(不符合本案例，损失没有左右震荡还好，还需要进一步下降)

**使用学习率调度器**
推荐使用**余弦退火调度器**（`cosine_with_restarts`），在训练后期自动降低学习率，使模型更稳定地收敛。

**预热步数调整**
增加`warmup_steps`（如从 10 增至 50），让模型在训练初期缓慢学习，避免剧烈震荡。

#### 3. **优化器选择**

- **切换至 AdamW 优化器**
  AdamW 在 Adam 基础上加入权重衰减，能有效减少震荡。若已使用 AdamW，可微调`weight_decay`（如设为`0.01`）。（可以尝试）

```bash
--optim adamw_torch \
--weight_decay 0.01 \
```

#### 4. **梯度平滑与裁剪**

- **梯度累积**
  保持`gradient_accumulation_steps=8`或更高，通过多步累积降低梯度噪声。
- **梯度裁剪**（可尝试进一步缩小该值）
  设置`max_grad_norm`限制梯度的最大范数，防止梯度爆炸导致的剧烈波动。

**示例**：

```bash
--max_grad_norm 0.5 \  # 降低梯度裁剪阈值
```

## 数据小插曲

原本43万6千中文语料，可惜太慢了，太费钱了，暂时放弃了：

![image-20250527170401196](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250527170401196.png)

# 效果验证

## 部署

![image-20250528152449979](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528152449979.png)

**方式一**：可以安装vllm引擎同时加载多个Lora模型。访问vllm api，参考代码：

```shell
# vllm启动命令
CUDA_VISIBLE_DEVICES=0 API_PORT=8000 vllm serve
/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-0.5B-Instruct \
--enable-lora \
--lora-modules
cot=/gemini/pretrain3/fenlei-checkpoint
spo=/gemini/pretrain/sop-checkpoint
yiliao=/gemini/pretrain2/yiliao-checkpoint
```

![image-20250528152839961](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528152839961.png)

**方式二**：也可以用LLama Factory UI加载启动。

![image-20250528153042704](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528153042704.png)

> 这里使用LLama Factory Web UI看下效果

## 训练前效果

![image-20250528154120258](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528154120258.png)

## 训练后效果

![image-20250528154455678](C:\Users\cc723\AppData\Roaming\Typora\typora-user-images\image-20250528154455678.png)

# 拓展学习

官网：[hiyouga/LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs (ACL 2024)](https://github.com/hiyouga/LLaMA-Factory)

多模态文旅项目微调：https://gallery.pai-ml.com/#/preview/deepLearning/nlp/llama_factory_qwen2vl
