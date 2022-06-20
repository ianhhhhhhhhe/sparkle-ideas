# Flink TaskManager 内存分配与 ExecutionGraph 管理

Flink 基本结构如下

<img src="https://nightlies.apache.org/flink/flink-docs-release-1.15/fig/processes.svg" width="800" height="350" />

## TaskManager 

<img src="https://nightlies.apache.org/flink/flink-docs-master/fig/detailed-mem-model.svg" width="800" height="350" />

JVM Heap : taskmanager 执行 application 时 JVM 分配内存。其中 Framework Heap 是 JVM 分配给 Flink TaskManager 框架的，Task Heap 是 JVM 分配给 Flink 运行 application 的。

Flink 的 JobManager -- TaskManager 结构为一种主从结构，JobManager 和 ResourceManager 负责管理 TaskManager 的资源以及 ExecutionJobVertex 的调度，TaskManager 之间仅做 ExecutionJobVertex 的通信，资源相互隔离。TaskManager 内部资源（内存、CPU）共享。

ExecutionJobVertex 分配时，会保证同一个 Slot 内部包含的 ExecutionJobVertex 来源于同一个 Job，同一个 TaskManager 内的多个 Slot 可能会包含来源于不同 Job 的 ExecutionJobVertex。

## ExecutionGraph

Flink 程序运行前，需要讲程序转化成 ExecutionGraph 之后部署在 TaskManager 内运行，在转化时需要经历以下几个步骤：

1. Program -> StreamGraph
2. StreamGraph -> JobGraph
3. JobGraph -> ExecutionGraph

在这三步骤中，会发生一系列的 Job 衔接合并，最终得到一个最简的 ExecutionJobVertex DAG。这些 JobVertex 会被分配到不同的 TaskManager Slot 中。再分配时会尽量将不同类别的 JobVertex 分配至同一个 slot 中，相同类别的 JobVertex 分配至不同的 slot 中。

考虑一个具有数据源、MapFunction和ReduceFunction的程序。source和MapFunction以4的并行度执行，而ReduceFunction以3的并行度执行。管道由序列Source-Map-Reduce组成。在具有2个TaskManager（每个TaskManager有3个插槽）的集群上，程序将按如下所述执行。

<img src="https://nightlies.apache.org/flink/flink-docs-release-1.15/fig/slots.svg" width="800" height="300" />

## TaskManager 重启的可能原因

目前测试环境所遇到过的 TaskManager 的重启大多是因为 ExecutionJobVertex 内存泄漏，造成某个/某些 TaskManager 的 JVM Heap 被占满所产生的 OOM。这个报错会导致 TaskManager 内所有 ExecutionJobVertex 获取不到足够的资源从而抛出 OOM 异常信息。另外一种导致 TaskManager OOM 的原因是 TaskManager 内部 Slot 过多，每个 Slot 内部的 ExecutionJobVertex 正常运行时，所需求的 JVM heap 总和依旧超出了 TaskManager 被分配的 JVM heap 的容量，此时需要对 TaskManager 所配置的 Slot 数量进行缩减，并适量增加 TaskManager 数量。

## Standalone 和 Application 模式部署的不同

Standalone 部署模式下，所有 ExecutionGraph 共享 JobManager 和 TaskManager。一旦 JobManager 或者 TaskManager 出现异常，所有相关服务都会收到影响。部分资源共享，对资源的需求较少但稳定性较差。

Application 部署模式下，所有 ExecutionGraph 独享 JobManager 和 TaskManager。JobManager 或者 TaskManager 的异常只会影响部署在其上的服务，不会影响其他的服务。资源不共享，对资源需求略高，但稳定性强。



## ExecutionJobVertex 数量对 Flink 集群的影响

Kubernetes 里面 pods 多会对 Kubernetes 的稳定性造成影响吗……

参考链接

- [Job-Scheduling](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/internals/job_scheduling/)
- [Flink-Architecture](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/concepts/flink-architecture/#flink-application-cluster)
- [DeployMent](https://nightlies.apache.org/flink/flink-docs-release-1.15/docs/deployment/overview/)