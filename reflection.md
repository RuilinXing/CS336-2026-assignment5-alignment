# Alignment 实现反思

## 遇到的问题与修复

1. `question_only` prompt 曾复用 R1 的奖励函数和 `</answer>` 停止词，导致合法的 `\boxed{...}` 回答被判为错误。
   - 修复：将 prompt 格式与奖励函数、停止词绑定；自定义 prompt 必须显式指定 `--rollout-format`。

2. Off-policy 训练先对完整 rollout 计算 `old_log_probs`，随后切出较短的小批次重新 token 化，造成两者序列宽度不一致。
   - 修复：在训练步中校验旧 log-prob 的批次维度，并裁掉仅由完整 rollout padding 引入的右侧列，使其与当前 token 化宽度对齐。

3. GRPO/CISPO 的 clip fraction 曾计入 prompt、padding，以及跨 microbatch 时按序列而非 response token 加权。
   - 修复：仅用 `response_mask` 覆盖的 token 统计，并按有效 response token 数聚合；GSPO 继续按序列统计。

4. 标准 on-policy GRPO 允许较小的 `train_batch_size`，但每个 rollout 只更新一次，可能静默丢弃后半个 rollout。
   - 修复：标准模式要求 `train_batch_size == rollout_batch_size`；off-policy 模式要求更新恰好覆盖完整 rollout。

5. 安全评测把 GSM8K 数字答案按字符串比较，`18.0` 会被误判为不同于 `18`；验证 response 长度也按字符而非 token 统计。
   - 修复：使用 `Decimal` 比较数值答案，并使用 `completion.token_ids` 计算生成长度。

6. `max_examples=0` 会错误读取一条 GSM8K 数据。
   - 修复：在读文件前处理零上限，并拒绝负数上限。

7. GRPO 训练时，同一题的多个 rollout 实际上是完全重复的 response，导致组内奖励没有方差，所有训练 step 都得到零梯度。
   - 复现证据：在一次双 H800、`group_size=8` 的 50-step 运行中，训练日志从 step 1 到 step 50 都显示 `mean_reward=0.0`、`mean_group_reward=0.0`、`loss=0.0`、`grad_norm=0.0`。同一次运行的验证集仍得到 `val_reward=0.001953125`（1024 个样本中有 2 个正确），因此评分函数和 vLLM 服务本身能够产生正奖励。
   - 直接检查：对 `outputs/grpo_seed0/rollouts/step_0040.json` 中同一题的前 8 条 response 做 SHA-256 哈希，得到 8 个相同的 `19fec579`。这证明它们不是恰好得到相同奖励，而是文本本身完全一致。
   - 根因：训练循环先构造 `repeated_prompts = [p0] * G + [p1] * G + ...`，然后通过一次 vLLM completion 请求发送这些重复 prompt。与此同时，采样参数固定为 `n=1`，并在同一请求内为每个重复 prompt 传入相同的 `seed`。vLLM 因而为同一题的每个副本初始化相同的随机采样过程，生成完全相同的 response。
   - 为什么这会使 GRPO 失效：标准 GRPO 对每个 prompt group 减去组均值。若一个 group 的 8 条 completion 相同，则它们的 reward 也相同；即使该答案恰好正确，减去组均值后所有 advantage 仍为 0。`grpo_train_step` 正确跳过零 advantage 样本，于是表现为零 loss 和零梯度，而不是一次显式报错。
   - 修复：新增 `generate_grouped_responses`。训练时只向 vLLM 发送每步的唯一 prompt 列表，并将请求参数改为 `n=group_size`，让 vLLM 对每个 prompt 返回一个独立的 completion group。训练侧仍保留 `repeated_prompts` 和 `repeated_ground_truths`，其顺序与 vLLM 返回的 prompt-major、group-minor completion 顺序对齐。`generate_responses` 的返回数检查同时改为 `len(prompts) * n`，避免未来错误地接受缺失的 group completion。
   - 防御性校验：在任何取模或 vLLM 请求前拒绝 `group_size <= 0`，避免 `group_size=0` 触发 `ZeroDivisionError`，或把负值作为非法的 vLLM `n` 传出。
   - 回归测试：新增 stub vLLM 测试，断言请求只收到唯一 prompt、`n` 从 1 覆盖为 group size、原始 seed 被保留，且返回的 completion 数为 `prompt 数 × group size`。另新增 `group_size=0` 与 `group_size=-1` 的参数校验测试。测试先在缺少 helper 和校验时失败，再在修复后通过；完整测试套件共 50 项全部通过。

## 经验

- 单元测试通过并不表示训练脚本路径正确：完整 rollout、padding、切分小批次和指标聚合必须有专门回归测试。
- 对齐训练中，token mask 的含义应在 loss、统计指标和跨 microbatch 聚合中保持一致。
- 先写能稳定复现真实输入形状的测试，再做最小修复，可以避免 PyTorch 广播等行为掩盖问题。
- 对 GRPO 而言，“重复同一个 prompt”不等于“得到多个独立 rollout”。需要验证同组 response 的文本或哈希确实不同；否则奖励方差为零时，训练会安静地退化为零更新。
