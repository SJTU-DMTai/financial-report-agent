# verifier相关的项目结构
- `financial-report-agent\src\agents\verifier.py` — 包含了verifier中几个agent的创建
- `financial-report-agent\src\utils\multi_types_verification.py` — verifier主要逻辑函数
- `financial-report-agent\src\prompt\__init__.py` - 包含了verifier用到的所有prompt
- `financial-report-agent\src\agents\verifier.py` - 包含了verifier agent的创建

# 问题
1. 我想要改回之前的三路检验模式
2. 去掉numericchecker,只保留fact\numeric\temporal verifier，通过路由到每个verifier
3. 仍然使用系统置信度，通过来源计算置信度

### 我希望我的代码简洁有力
