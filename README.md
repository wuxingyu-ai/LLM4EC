# Interdisciplinary Research on LLM and Evolutionary Computation

A list of awesome papers and resources of the intersection of Large Language Models and Evolutionary Computation.

🎉 ***News: Our survey has been released.***
[Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap](https://arxiv.org/pdf/2401.10034)

***The related work and projects will be updated soon and continuously.***

<div align="center">
	<img src="https://github.com/wuxingyu-ai/LLM4EC/blob/main/Framework.png" alt="Editor" width="600">
</div>

If our work has been of assistance to you, please feel free to cite our survey. Thank you.
```
@article{wu2024evolutionary,
  title={Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap},
  author={Wu, Xingyu and Wu, Sheng-hao and Wu, Jibin and Feng, Liang and Tan, Kay Chen},
  journal={CoRR},
  volume={abs/2401.10034},
  year={2024}
}
```

# Table of Contents

- [Interdisciplinary Research on LLM and Evolutionary Computation](#interdisciplinary-research-on-llm-and-evolutionary-computation)
- [Table of Contents](#table-of-contents)
  - [LLM-enhanced EA](#llm-enhanced-ea)
    - [LLM-assisted Black-box Optimization](#llm-assisted-black-box-optimization)
    - [LLM-assisted Optimization Algorithm Generation](#llm-assisted-optimization-algorithm-generation)
    - [LLM Empower EA for Other Capabilities](#llm-empower-ea-for-other-capabilities)
  - [EA-enhanced LLM](#ea-enhanced-llm)
    - [EA-based Prompt Engineering](#ea-based-prompt-engineering)
    - [EA-based LLM Architecture Search](#ea-based-llm-architecture-search)
    - [EA Empower LLM for Other Capabilities](#ea-empower-llm-for-other-capabilities)
  - [Integrated Synergy and Application of LLM and EA](#integrated-synergy-and-application-of-llm-and-ea)
    - [Code Generation](#code-generation)
    - [Software Engineering](#software-engineering)
    - [Neural Architecture Search](#neural-architecture-search)

## LLM-enhanced EA

### LLM-assisted Black-box Optimization

| **Category** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Name                   |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| Evaluation | [Exploring the True Potential: Evaluating the Black-Box Optimization Capability of Large Language Models](https://arxiv.org/pdf/2404.06290) | arXiv | 2024 | N/A | N/A |
| Evaluation | [Towards Optimizing with Large Language Models](https://arxiv.org/pdf/2310.05204) | arXiv | 2023 | N/A | N/A |
| Single-objective | [Large Language Models as Optimizers](https://arxiv.org/pdf/2309.03409) | ICLR | 2024 | [Python](https://github.com/google-deepmind/opro) | OPRO |
| Single-objective | [Language Model Crossover: Variation through Few-Shot Prompting](https://arxiv.org/pdf/2302.12170) | arXiv | 2023 | N/A | LMX |
| Single-objective | [Large Language Models as Evolutionary Optimizers](https://arxiv.org/pdf/2310.19046) | CEC | 2024 | N/A | LMEA |
| Single-objective | [Large Language Models As Evolution Strategies](https://arxiv.org/pdf/2402.18381) | arXiv | 2024 | N/A | EvoLLM |
| Single-objective | [How Multimodal Integration Boost the Performance of LLM for Optimization: Case Study on Capacitated Vehicle Routing Problems](https://arxiv.org/pdf/2403.01757) | arXiv | 2024 | N/A | Huang et al. |
| Single-objective | [Large Language Model-Based Evolutionary Optimizer: Reasoning with Elitism](https://arxiv.org/pdf/2403.02054) | arXiv | 2024 | N/A | LEO |
| Single-objective (Application) | [CUDA-Accelerated Soft Robot Neural Evolution with Large Language Model Supervision](https://arxiv.org/pdf/2405.00698) | arXiv | 2024 | N/A | Zhang |
| Single-objective (Application) | [Evolving Interpretable Visual Classifiers with Large Language Models](https://arxiv.org/pdf/2404.09941) | arXiv | 2024 | N/A | LLM-Mutate |
| Multi-objective | [Quality-Diversity through AI Feedback](https://arxiv.org/pdf/2310.13032v4) | Workshop in NeurIPS | 2023 | [Python](https://openreview.net/attachment?id=nr0w6CH7v4&name=supplementary_material) | QDAIF |
| Multi-objective | [Large Language Model for Multi-objective Evolutionary Optimization](https://arxiv.org/pdf/2310.12541) | arXiv | 2023 | [Python](https://github.com/FeiLiu36/LLM4MOEA) | LLM4MOEA |
| Multi-objective | [Large Language Models as In-context AI Generators for Quality-Diversity](https://arxiv.org/pdf/2404.15794) | arXiv | 2024 | N/A | In-context QD |
| Multi-objective | [Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization](https://arxiv.org/pdf/2405.05767) | arXiv | 2024 | N/A | CMOEA-LLM |


### LLM-assisted Optimization Algorithm Generation

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Generated Algorithm                   |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| Pluhacek et al. | [Leveraging Large Language Models for the Generation of Novel Metaheuristic Optimization Algorithms](https://dl.acm.org/doi/abs/10.1145/3583133.3596401) | GECCO | 2023 | N/A | Hybrid swarm intelligence optimization algorithm |
| OptiMUS | [OptiMUS: Optimization Modeling Using MIP Solvers and Large Language Models](https://arxiv.org/pdf/2310.06116) | arXiv | 2023 | [Python](https://github.com/teshnizi/OptiMUS) | Mixed-integer linear programming problem |
| AEL | [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design using Large Language Model](https://arxiv.org/pdf/2401.02051) | ICML | 2024 | [Python](https://github.com/FeiLiu36/EoH) | Heuristic algorithm |
| ReEvo | [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://arxiv.org/pdf/2402.01145) | arXiv | 2024 | [Python](https://github.com/ai4co/LLM-as-HH) | Heuristic algorithm |
| LLM_GP | [Evolving Code with A Large Language Model](https://arxiv.org/pdf/2401.07102) | arXiv | 2024 | N/A | Genetic Programming |
| ZSO | [Leveraging Large Language Model to Generate a Novel Metaheuristic Algorithm with CRISPE Framework](https://arxiv.org/pdf/2403.16417) | arXiv | 2024 | [Python](https://github.com/RuiZhong961230/ZSO) | Zoological search optimization algorithm |
| SR-EAD | [Evolution Transformer: In-Context Evolutionary Optimization](https://arxiv.org/pdf/2403.02985) | arXiv | 2024 | [Python](https://github.com/RobertTLange/evosax) | Evolutionary strategy or evolution transformer |
| EvolCAF | [Evolve Cost-aware Acquisition Functions Using Large Language Models](https://arxiv.org/pdf/2404.16906) | arXiv | 2024 | [Python](https://github.com/RobertTLange/evosax) | Cost-aware Bayesian optimization |
| OpenELM | [The OpenELM Library: Leveraging Progress in Language Models for Novel Evolutionary Algorithms](https://arxiv.org/pdf/2404.16906) | arXiv | 2024 | [Python](https://github.com/CarperAI/OpenELM) | Open-source Python library |

### LLM Empower EA for Other Capabilities

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Enhancement Aspect                 |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| OptiChat | [Diagnosing Infeasible Optimization Problems Using Large Language Models](https://arxiv.org/pdf/2308.12923) | arXiv | 2023 | [Python](https://github.com/li-group/OptiChat) | Identify potential sources of infeasibility |
| AS-LLM | [Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation](https://arxiv.org/pdf/2311.13184) | IJCAI | 2024 | [Python](https://github.com/wuxingyu-ai/AS-LLM) | Algorithm representation and algorithm selection |
| GP4NLDR | [Explaining Genetic Programming Trees Using Large Language Models](https://arxiv.org/pdf/2403.03397) | arXiv | 2024 | N/A |  Provide explainability for results of EA |

## EA-enhanced LLM

### EA-based Prompt Engineering

| **Category** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Discrete Prompt Optimization | GrIPS | [GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models](https://arxiv.org/pdf/2203.07281) | EACL | 2023 | [Python](https://github.com/archiki/GrIPS) |
| Discrete Prompt Optimization | GPS | [GPS: Genetic Prompt Search for Efficient Few-shot Learning](https://arxiv.org/pdf/2210.17041) | EMNLP | 2022 | [Python](https://github.com/hwxu20/GPS) |
| Discrete Prompt Optimization | EvoPrompt | [Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers](https://arxiv.org/pdf/2309.08532) | ICLR | 2024 | [Python](https://github.com/beeevita/EvoPrompt) |
| Discrete Prompt Optimization | Plum | [Plum: Prompt Learning Using Metaheuristic](https://arxiv.org/pdf/2311.08364) | arXiv | 2024 | [Python](https://github.com/research4pan/Plum) |
| Discrete Prompt Optimization | SPELL | [SPELL: Semantic Prompt Evolution Based on A LLM](https://arxiv.org/pdf/2310.01260) | arXiv | 2023 | N/A |
| Discrete Prompt Optimization | EoT prompting | [Zero-Shot Chain-of-Thought Reasoning Guided by Evolutionary Algorithms in Large Language Models](https://arxiv.org/pdf/2402.05376) | arXiv | 2024 | [Python](https://github.com/stan-anony/Zero-shot-EoT-Prompting) |
| Discrete Prompt Optimization | PhaseEvo | [PhaseEvo: Towards Unified In-Context Prompt Optimization for Large Language Models](https://arxiv.org/pdf/2402.11347) | arXiv | 2024 | N/A |
| Discrete Prompt Optimization | InstOptima | [InstOptima: Evolutionary Multi-objective Instruction Optimization via Large Language Model-based Instruction Operators](https://arxiv.org/pdf/2310.17630) | EMNLP | 2023 | [Python](https://github.com/yangheng95/InstOptima) |
| Discrete Prompt Optimization | EMO-Prompts | [Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments](https://arxiv.org/pdf/2401.09862) | EvoApplications | 2024 | [Python]() |
| Gradient-Free Soft Prompt Optimization | BBT | [Black-Box Tuning for Language-Model-as-a-Service](https://proceedings.mlr.press/v162/sun22e/sun22e.pdf) | ICML | 2022 | [Python](https://github.com/txsun1997/Black-Box-Tuning) |
| Gradient-Free Soft Prompt Optimization | BBTv2 | [BBTv2: Towards a Gradient-Free Future with Large Language Models](https://arxiv.org/pdf/2205.11200) | EMNLP | 2022 | [Python](https://github.com/txsun1997/Black-Box-Tuning) |
| Gradient-Free Soft Prompt Optimization | Clip-Tuning | [Clip-Tuning: Towards Derivative-free Prompt Learning with a Mixture of Rewards](https://aclanthology.org/2022.findings-emnlp.8.pdf) | EMNLP | 2022 | N/A |
| Prompt Generation for Data Augmentation | Evol-Instruct | [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/pdf/2304.12244) | ICLR | 2024 | [Python](https://github.com/nlpxucan/WizardLM) |
| Prompt Generation for Data Augmentation | Sun et al. | [Dial-insight: Fine-tuning Large Language Models with High-Quality Domain-Specific Data Preventing Capability Collapse](https://arxiv.org/pdf/2403.09167) | arXiv | 2024 | N/A |
| Prompt Generation for Security | AutoDAN | [AutoDAN: Generating Stealthy Jailbreak Prompts on Aligned Large Language Models](https://arxiv.org/pdf/2310.04451) | ICLR | 2024 | [Python](https://github.com/SheltonLiu-N/AutoDAN) |
| Prompt Generation for Security | Jailbreak Attacks | [Open Sesame! Universal Black Box Jailbreaking of Large Language Models](https://arxiv.org/pdf/2309.01446) | arXiv | 2024 | N/A |
| Prompt Generation for Security | SMEA | [Is the System Message Really Important to Jailbreaks in Large Language Models?](https://arxiv.org/pdf/2402.14857) | arXiv | 2024 | N/A |
| Prompt Generation for Security | Shi et al. | [Red Teaming Language Model Detectors with Language Models](https://arxiv.org/pdf/2305.19713) | TACL | 2024 | [Python](https://github.com/shizhouxing/LLM-Detector-Robustness) |

### EA-based LLM Architecture Search

Note: Approaches discussed here primarily focus on LLM architecture search, and their techniques are based on EAs.

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      LLM                 |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| AutoBERT-Zero | [AutoBERT-Zero: Evolving BERT Backbone from Scratch](https://arxiv.org/pdf/2107.07445) | AAAI | 2022 | [Python](https://github.com/JunnYu/AutoBERT-Zero-pytorch/tree/main) | BERT |
| SuperShaper | [SuperShaper: Task-Agnostic Super Pre-training of BERT Models with Variable Hidden Dimensions](https://arxiv.org/pdf/2110.04711) | arXiv | 2021 | N/A | BERT |
| AutoTinyBERT | [AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient Pre-trained Language Models](https://arxiv.org/pdf/2107.13686) | ACL | 2021 | [Python](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/AutoTinyBERT) | BERT |
| LiteTransformerSearch | [LiteTransformerSearch: Training-free Neural Architecture Search for Efficient Language Models](https://arxiv.org/pdf/2203.02094) | NeurIPS | 2022 | [Python](https://github.com/microsoft/archai/tree/neurips-lts/archai/nlp) | GPT-2 |
| Klein et al. | [Structural Pruning of Large Language Models via Neural Architecture Search](https://openreview.net/pdf?id=SHlZcInS6C) | AutoML | 2023 | N/A | BERT |
| Choong et al. | [Jack and Masters of All Trades: One-Pass Learning of a Set of Model Sets from Foundation AI Models](https://arxiv.org/pdf/2205.00671) | IEEE CIM | 2023 | N/A | M2M100-418M, ResNet-18 |

### EA Empower LLM for Other Capabilities

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Enhancement Aspect                 |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| Length-Adaptive Transformer Model | [Length-Adaptive Transformer: Train Once with Length Drop, Use Anytime with Search](https://aclanthology.org/2021.acl-long.508.pdf) | ACL | 2021 | [Python](https://github.com/clovaai/length-adaptive-transformer) | Automatically adjust the sequence length according to different computational resource constraints |
| HexGen | [HexGen: Generative Inference of Large-Scale Foundation Model over Heterogeneous Decentralized Environment](https://arxiv.org/pdf/2311.11514) | arXiv | 2023 | [Python](https://github.com/Relaxed-System-Lab/HexGen) | Deploy generative inference services for LLMs in a heterogeneous distributed environment |
| LongRoPE | [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/pdf/2402.13753) | arXiv | 2023 | [Python](https://github.com/microsoft/LongRoPE) | Extend the context window of LLMs to 2048k tokens |
| Evolutionary Model Merge | [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/pdf/2403.13187) | arXiv | 2024 | [Python](https://github.com/SakanaAI/evolutionary-model-merge) | Utilize CMA-ES algorithm to optimize merged LLM in both parameter and data flow space |
| BLADE | [BLADE: Enhancing Black-box Large Language Models with Small Domain-Specific Models](https://arxiv.org/pdf/2403.18365) | arXiv | 2024 | N/A | Find soft prompts that optimizes the consistency between the outputs of two models |
| Self-evolution in LLM | [A Survey on Self-Evolution of Large Language Models](https://arxiv.org/pdf/2404.14387) | arXiv | 2024 | [Summary](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Awesome-Self-Evolution-of-LLM) | Some studies for LLM self-evolution also adopted the ideas of EAs |

## Integrated Synergy and Application of LLM and EA

### Code Generation

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Applicable scenarios                   |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
|  | []() |  |  | [Python]() |  |

### Software Engineering

### Neural Architecture Search

Note: Methods reviewed here leverage the synergistic combination of EAs and LLMs, which are more versatile and not limited to LLM architecture search alone, applicable to a broader range of NAS tasks..

### Others Generative Tasks



Hope our conclusion can help your work.

