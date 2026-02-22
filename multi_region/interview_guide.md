# 🎤 Interview Guide: k8s_operator & multi_region

> **How to use this guide**
> 1. **Open with the WHY** — always explain the business/engineering problem *before* the solution
> 2. **Use STAR** for challenge questions — Situation → Task → Action → Result
> 3. **Mention tradeoffs** — show you made deliberate decisions, not just the first thing that came to mind

---

## 📦 Project 1: TrainingJob Kubernetes Operator (`k8s_operator`)

### One-Sentence Pitch

**EN:** "I built a Kubernetes Operator that lets data scientists run distributed GPU training with a single YAML file, abstracting away all the orchestration complexity."

**🇨🇳:** "我做了一个 Kubernetes Operator，让算法工程师用一个 YAML 文件就能跑分布式 GPU 训练，底层编排复杂性完全透明。"

---

### 1. Start with the WHY

**English:**

> "At scale, ML training is not just a Python script — it's a distributed systems problem. You have multiple GPU nodes that need to coordinate, checkpoints that need to survive failures, and jobs that need to automatically retry without human intervention. The engineering problem was: **how do you give data scientists a simple, declarative interface to run distributed training, while hiding all the Kubernetes complexity from them?**
>
> The answer was a Kubernetes Operator. I built a custom `TrainingJob` resource — a data scientist writes ten lines of YAML describing their model and hyperparameters, and the operator takes care of orchestrating the actual Kubernetes Jobs, headless Services, ConfigMaps, retries, and progress monitoring. It's the same pattern Kubeflow uses, but built from scratch so I could understand every component."

**🇨🇳 中文：**

> "在大规模场景下，ML 训练不只是跑一个 Python 脚本——它本质上是一个分布式系统问题。多个 GPU 节点需要互相协调，checkpoint 要能在故障后存活，任务失败后要能自动重试，不能依赖人工介入。工程上的核心问题是：**怎样给算法工程师一个简单的声明式接口来跑分布式训练，同时把底层所有 Kubernetes 的复杂性都屏蔽掉？**
>
> 答案就是 Kubernetes Operator。我定义了一个自定义资源 `TrainingJob`——算法同学只需写十行 YAML 描述模型和超参数，Operator 就负责编排底层的 Kubernetes Job、Headless Service、ConfigMap，处理自动重试和进度监控。这和 Kubeflow 的思路一致，但我从零实现，目的是深刻理解每一个组件。"

---

### 2. STAR: Hardest Technical Challenge

**Q: "What was the most difficult technical challenge?"**

**English:**

> **Situation:** When I first built the timer-based reconciler, I had a bug where if the [create](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#70-145) event handler and the 30-second timer both fired at the same time, they would both try to create the same Kubernetes Job simultaneously — getting an `AlreadyExists` conflict error and putting the job into `Failed` state even though nothing was actually wrong.
>
> **Task:** I needed to make the reconciliation loop fully **idempotent and safe for concurrent execution** without using distributed locks.
>
> **Action:** I restructured the state machine so that [create](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#70-145) and [update](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#147-218) Kopf handlers only set the **initial status** (state = `Pending`) — they never touch Kubernetes resources directly. All resource creation happens exclusively in the timer handler. Before creating a Job, the controller calls `get_job()` first and only proceeds if the job doesn't already exist. This is the Kubernetes-native "desired state vs. actual state" pattern.
>
> **Result:** The system became fully level-triggered and idempotent. The same reconcile cycle can run 100 times on the same object safely. This is the same design pattern used by the official Kubernetes controller-runtime library.

**🇨🇳 中文：**

> **Situation（背景）：** 第一版调和循环有个 bug——[create](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#70-145) 事件 handler 和 30 秒 timer handler 同时触发时，会同时尝试创建同一个 Kubernetes Job，导致 `AlreadyExists` 冲突报错，任务被置为 `Failed`，但实际上什么都没有出错。
>
> **Task（任务）：** 需要让调和循环完全**幂等且并发安全**，同时不引入分布式锁。
>
> **Action（行动）：** 重构了状态机职责：[create](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#70-145) 和 [update](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/k8s_operator/src/operator/main.py#147-218) 的 Kopf handler 只负责设置**初始状态**（state = `Pending`），绝不直接操作 Kubernetes 资源。所有资源创建集中在 timer handler 里，创建 Job 前先调用 `get_job()` 检查是否已存在，只有不存在才创建。这就是 Kubernetes 原生的"期望状态 vs 实际状态"模式。
>
> **Result（结果）：** 系统变成完全水平触发和幂等的。同一对象的调和循环跑100次也安全，和官方 Kubernetes controller-runtime 库设计模式一致。

---

### 3. Key Tradeoffs

**English:**

> **① Kopf (Python) vs. client-go (Go)**
> I chose Python + Kopf over Go + client-go, even though Go operators are more common in production and more performant. Reason: ML teams run Python everywhere — data scientists, training scripts, model serving. Having the operator in Python means the team can actually read and contribute to it, rather than treating it as a black box.

> **② Timer-driven vs. pure event-driven reconciliation**
> I chose 30-second periodic reconciliation *plus* event handlers, not purely event-driven. Risk of pure event-driven: if an event is missed (network partition, pod restart), the system gets stuck. The timer is a safety net that always drives the system toward desired state, regardless of missed events.

> **③ Operator doesn't stream checkpoint data**
> The operator tracks epoch counts and path metadata in the CRD status, but relies on the training container itself to write checkpoints to a mounted volume. Having the operator stream checkpoint data would create a massive I/O bottleneck. Separation of concerns wins here.

**🇨🇳 中文：**

> **① Kopf (Python) vs. client-go (Go)**
> 选择 Python + Kopf 而非更主流的 Go + client-go。虽然 Go Operator 性能更好、生态更成熟，但 ML 团队整个栈都是 Python。用 Python 写 Operator 意味着团队真的能读懂、能参与维护，而不是把它当成平台组的黑盒。

> **② Timer 驱动 vs. 纯事件驱动**
> 选择 30 秒定时调和 + 事件 handler 混合模式，而非纯事件驱动。纯事件驱动的风险：如果事件丢失（网络分区、Pod 重启），系统就会卡住。Timer 是兜底机制，确保无论是否有事件，系统都能向期望状态收敛。

> **③ Operator 不直接传输 checkpoint 数据**
> Operator 只在 CRD status 里追踪 epoch 数量和路径元数据，实际写入由训练容器自己挂载 Volume 完成。让 Operator 流式传输 checkpoint 数据会形成巨大 I/O 瓶颈，关注点分离更合理。

---

### 4. Quick Q&A

| Question | English Answer | 中文回答 |
|----------|----------------|----------|
| What is a Kubernetes Operator? | A controller that extends the K8s API with custom resources and encodes domain-specific operational logic | 把领域知识编码进 K8s 控制器的模式，让复杂有状态应用可以声明式管理 |
| How does distributed training coordinate? | A Headless Service gives each worker a stable DNS name; `MASTER_ADDR` points to it so PyTorch can discover peers automatically | Headless Service 给每个 worker 稳定的 DNS 名；`MASTER_ADDR` 指向它，PyTorch 自动发现节点 |
| How are failures handled? | Timer checks every 30s; if `failed >= backoffLimit`, state → `Failed`; next cycle resets to `Pending` if `restartCount < backoffLimit` | 每 30 秒检查；失败超阈值 → `Failed`；如重试次数未超限，下个周期重置为 `Pending` |

---

---

## ☁️ Project 2: Multi-Region ML Platform (`multi_region`)

### One-Sentence Pitch

**EN:** "I built a unified multi-cloud ML serving platform across AWS, GCP, and Azure with automated failover in under 60 seconds and ~30% cost savings."

**🇨🇳:** "我搭建了一个跨 AWS、GCP、Azure 的多云 ML 推理平台，自动故障转移低于 60 秒，节省约 30% 成本。"

---

### 1. Start with the WHY

**English:**

> "The problem we were solving was: **single-region ML serving is fragile and expensive at global scale.** If you serve a latency-sensitive prediction API from one AWS region, users in Asia get 200ms+ of network latency before the model even starts computing. And if that region goes down — which *does* happen — your entire ML product is unavailable.
>
> The business requirement was 99.95% availability, sub-300ms global P99 latency, and no more than 60 seconds of disruption during a regional failure. The engineering challenge: how do you build this across *three different cloud providers* without creating a maintenance nightmare?
>
> I built a unified async Python platform that treats AWS, GCP, and Azure as interchangeable backends — one service orchestrates failover, model replication, cost analysis, and monitoring across all three clouds."

**🇨🇳 中文：**

> "我们要解决的问题是：**在全球规模下，单区域 ML 推理服务既脆弱又昂贵。** 如果把预测 API 只部署在一个 AWS 区域，亚洲用户在模型开始计算之前就要承受 200ms+ 的网络延迟。而一旦那个区域出故障——这种事真的会发生——整个 ML 产品就全线不可用。
>
> 业务要求是：99.95% 的可用性，全球 P99 延迟低于 300ms，区域故障时切换时间不超过 60 秒。工程挑战：如何跨越**三个不同云厂商**来实现这些目标，同时不让系统变成运维噩梦？
>
> 我构建了一个统一的异步 Python 平台，把 AWS、GCP 和 Azure 当作可互换的后端处理，一个服务统一编排跨三云的故障转移、模型同步、成本分析和监控。"

---

### 2. STAR: Hardest Technical Challenge

**Q: "Tell me about a time you had to design for resilience."**

**English:**

> **Situation:** Early on, the failover controller was purely reactive — it triggered failover on any detected failure. In testing, a degraded region would briefly recover, cancel the failover, fail again seconds later, and cycle repeatedly. This caused constant DNS churn — the system felt *less* stable than single-region.
>
> **Task:** Make failover decisions stable and deliberate, not reactive to every transient blip, while still meeting the 60-second recovery SLA.
>
> **Action:** I implemented two mechanisms. First, a **consecutive failures threshold** — failover only triggers after 3 consecutive failed health checks. Second, a **degradation counter** — even if checks pass, if a region stays `DEGRADED` for 5+ consecutive cycles (consistently slow, not just briefly unhealthy), failover still triggers. This distinguishes brief network hiccups from genuine regional problems.
>
> **Result:** Flapping was eliminated entirely. Median failover time in simulated tests was under 35 seconds (well within SLA), with zero false-positive failovers over a 72-hour stability test.

**🇨🇳 中文：**

> **Situation（背景）：** 早期，故障转移控制器是纯响应式的——检测到故障就触发切换。测试中发现**"抖动"问题**：降级区域短暂恢复，控制器取消切换，几秒后又再次失败，如此循环，导致不断 DNS 变更，系统比单区域更不稳定。
>
> **Task（任务）：** 让故障转移决策更稳定，不对每次瞬间抖动做出反应，同时满足 60 秒恢复 SLA。
>
> **Action（行动）：** 实现了两个机制：①**连续失败阈值**——连续 3 次健康检查失败才触发故障转移；②**降级计数器**——即使健康检查通过，若某区域连续 5 次处于 `DEGRADED` 状态（持续高延迟），也会触发切换。这样能区分短暂网络抖动和真正的区域性故障。
>
> **Result（结果）：** 抖动问题被彻底消除。模拟故障测试中，中位切换时间低于 35 秒（远在 SLA 以内），72 小时稳定性测试中误报率为零。

---

### 3. Key Tradeoffs

**English:**

> **① Async Python monolith vs. Go microservices**
> I chose a single async Python process over separate Go microservices per subsystem. Python async is less performant under CPU load, but all the work here is I/O-bound (health checks, cloud API calls, Prometheus queries) — the difference is negligible. One process is dramatically simpler to operate and debug than five separate services. At this scale, operational simplicity wins.

> **② Active-passive failover vs. active-active**
> I implemented active-passive (one primary, failover to secondary) rather than active-active (all regions serving simultaneously). Active-active gives better latency everywhere but introduces **split-brain risk** — if regions disagree on model versions, you corrupt your serving state. For ML inference, serving a slightly stale but *consistent* model is better than serving inconsistent outputs.

> **③ Pull-based replication vs. push-based**
> [ModelReplicator](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#291-655) uses a **pull model** — periodically scans each region and copies whatever is missing. Push-based (source pushes to targets immediately) sounds faster, but if a target is down during the push, you need a complex retry queue. Pull is more resilient: if a target is down, the next pull cycle catches it automatically. The cost: up to 5 minutes of replication lag — acceptable for this use case.

**🇨🇳 中文：**

> **① 异步 Python 单体 vs. Go 微服务**
> 选择单个异步 Python 进程，而不是为每个子系统分别建一个 Go 微服务。Python async 在高 CPU 负载下性能不如 Go，但这里所有工作都是 I/O 密集型（健康检查、云 API 调用、Prometheus 查询），性能差距可以忽略。一个进程的运维复杂度远低于五个独立微服务，在这个规模下运维简单性更重要。

> **② 主从故障转移 vs. 多活**
> 实现主从模式，而非多活。多活延迟更低，但引入**脑裂风险**——各区域对模型版本看法不一致，推理状态就会被破坏。对 ML 推理来说，提供稍旧但一致的模型，远好过提供不一致的输出。

> **③ 拉取式复制 vs. 推送式**
> [ModelReplicator](file:///c:/Users/Chaoran%20Zhou/Downloads/ai-infra/project%20myself/multi_region/src/replication/model_replicator.py#291-655) 采用**拉取模型**——定期扫描各区域，复制缺失内容。推送式看起来更快，但如果目标区域在推送时宕机，就需要复杂的重试队列。拉取更有韧性：目标宕机时，下一个拉取周期自然会补上。代价是最多 5 分钟复制延迟——在这个场景下可以接受。

---

### 4. Quick Q&A

| Question | English Answer | 中文回答 |
|----------|----------------|----------|
| Why three clouds? | Vendor lock-in avoidance, regulatory compliance (GDPR/data sovereignty), cost arbitrage via spot instances (~30% savings) | 避免厂商锁定、满足合规要求（GDPR/数据主权）、通过 Spot 实例进行成本套利（约节省30%） |
| How is model consistency guaranteed? | SHA256 checksum on source + size verification on target after upload — two layers of integrity checking | 源端 SHA256 校验 + 上传后目标端文件大小验证，双重完整性校验 |
| How does cost analysis work? | Parallel async calls to AWS Cost Explorer, GCP BigQuery billing export, Azure Cost Management; aggregated with anomaly detection (2σ) and trend analysis | 并行异步调用三家云的账单 API；聚合后用标准差检测异常，计算趋势 |
| How do you work without cloud credentials locally? | Lazy init: SDK clients created on first use inside try/except; None if credentials missing; all consumers check `if client is None: return []` | 懒加载：SDK 客户端首次使用时在 try/except 里创建；缺少凭据置为 None；所有调用方先判断 `if client is None: return []` |

---

---

## 📊 Key Metrics to Remember

| Metric | Value |
|--------|-------|
| Failover time | **< 60 seconds** (median ~35s in tests) |
| Global P99 latency | **< 300ms** |
| Replication lag | **< 5 minutes** |
| Cost savings vs. single-region | **~30%** |
| Platform availability target | **99.95%** |
| Operator reconcile interval | **30 seconds** |
| Health check interval | **10 seconds** |
| Failure threshold before failover | **3 consecutive failures** |

---

## 🏁 Practice Checklist

- [ ] Can you explain the WHY for each project in under 60 seconds?
- [ ] Can you tell the STAR story for each project's hardest challenge without notes?
- [ ] Can you name 3 tradeoffs for each project and explain *why* you chose what you chose?
- [ ] Can you recite the key metrics table from memory?
- [ ] Can you whiteboard the `TrainingJob` state machine? (`Pending → Initializing → Running → Completed / Failed`)
- [ ] Can you whiteboard the failover decision logic? (health check → threshold → target selection → DNS update)
