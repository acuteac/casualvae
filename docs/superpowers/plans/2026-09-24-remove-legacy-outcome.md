# Remove Legacy Outcome Paths Implementation Plan

**Goal:** 删除 encoded_y、OutcomeEncoder、MixedOutcomeHead 和无 outcome 备用路径；模型必须包含正维度 outcome，支持 zero / gaussian，默认 zero。

**Architecture:** 保留现有组学共享/私有编码器、SCM、OutcomeDecoder、损失及 CSV 格式。SCM 的 A 使用 [target, source] 索引，统一求解 (I-A)z=eps+context。外部 forward_joint 参数暂保持兼容，但真实 Y 不参与生成。

**Tech Stack:** Python / PyTorch，Conda casualvae_01。按用户要求仅做小范围测试，不跑全套测试、真实数据训练或多种子实验；本会话内执行。

## 1. 明确约束并更新定向测试

- [x] 修改 tests/test_outcome_exogenous_mode.py：以拒绝 encoded_y、缺失/零维度 outcome 的测试替代旧模式兼容测试。
- [x] 确认新增约束测试在清理前失败。
- [x] 保留 zero / gaussian 的输入独立性、Y loss 梯度、方向掩码和小型 CSV 导出检查。

## 2. 清理核心模型

- [x] 修改 src/casualvae/module_aclf_partial_multimodal.py：删除两个旧类及其初始化、encode_outcome、返回字段、outcome KL。
- [x] 模型与 BlockSCM 初始化检查 outcome 维度为正整数；无 outcome 直接 ValueError。
- [x] 删除 transpose_adjacency，SCM 使用 self.eye - a；删除无 outcome 解码、投影、损失及统计分支。
- [x] 删除 outcome_hidden_dims 构造参数和配置读取，CLI 必须提供正维度 outcome。
- [x] 保持结局方向 mask、OutcomeDecoder、MSE/BCE、best_tau、eval/no_grad 和 CSV 输出约定。

## 3. 同步配置、调用及说明

- [x] 清理 configs/aclf_full_multimodal_config.json 与 scripts/preprocessing/generate_full_config.py 中废弃的两个 outcome hidden 配置；生成器显式写 zero / sigma。
- [x] 更新 tests/test_best_tau_and_validation_mode.py、tests/test_seed_separation.py 的模型构造调用（仅适配，不扩大测试范围）。
- [x] 更新 readme.md：固定 outcome 架构、两种模式、旧 checkpoint 多余参数及同 seed 初始化序列变化的说明。

## 4. 简单验证与交付

- [x] 仅运行 conda run --no-capture-output -n casualvae_01 python -m unittest discover -s tests -p test_outcome_exogenous_mode.py -v。
- [x] 两种模式各做一个小批次 joint_loss 前向/反向及 optimizer step，确认有限 loss。
- [x] 用训练脚本 --help 验证 CLI 可导入；静态检查无残留旧分支，git diff --check 检查补丁。
- [x] 汇报范围、测试结果和旧权重加载限制；不改写历史实验文件，不自动迁移旧 checkpoint。

## 验证记录

- 解释器：D:/ProgramData/Anaconda_envs/envs/casualvae_01/python.exe。
- 新约束清理前按预期失败；清理后该测试文件 10/10 通过。
- 小批次训练：zero loss=4.929253，gaussian loss=4.836913，有限梯度和参数更新通过。
- CLI --help 成功；当前配置 mode=zero、outcome_dim=3，两个废弃配置均已移除。
- 未运行全套测试或完整实验。
