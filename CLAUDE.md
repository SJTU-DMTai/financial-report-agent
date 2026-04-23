# verifier相关的项目结构
- `financial-report-agent\src\agents\verifier.py` — 包含了verifier中几个agent的创建
- `financial-report-agent\src\utils\multi_types_verification.py` — verifier主要逻辑函数
- `financial-report-agent\src\prompt\__init__.py` - 包含了verifier用到的所有prompt
- `financial-report-agent\src\agents\verifier.py` - 包含了verifier agent的创建

# 下面是chatgpt指出的问题
1. 严重】issue merge 仍然不“语义去重”

你现在：

key = (iss.type, iss.claim_id)

问题：

❌ 同一个 claim 上，同一 type 的不同错误会被“合并”

举例

一个 claim：

Revenue = 100M USD in 2023

可能有两个问题：

value mismatch
unit mismatch

现在：

(type="numeric_error", claim_id="c1")

👉 被 merge 成一个 issue

正确做法（论文级）
key = (
    iss.claim_id,
    iss.type,
    normalize(iss.description)
)

否则：

❌ 你在 silently 丢信息

2. 严重】confidence fallback 仍然是错的

你这里：

except Exception:
    source_keys.add(f"cite:{cid}")

问题我之前说过，但你还没改：

❌ resolve 失败 → 不应该当作新 source

正确做法
except Exception:
    return 0.5  # 或直接标记 low confidence

3. 【中等】segment_score 仍然是 noisy signal

你现在：

segment_score = avg_score * 20

问题：

❌ claim 数量变化 → score 不稳定

举例
Round1: 5 claims
Round2: 7 claims

👉 avg_score 不可比

更稳做法（简单版）
segment_score = weighted by issue counts

或者：

👉 直接减少对 score 的依赖（你 loop 那边更关键）
### 我希望我的代码简洁有力
