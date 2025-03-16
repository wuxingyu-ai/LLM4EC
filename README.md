# Interdisciplinary Research on LLM and Evolutionary Computation

A list of awesome papers and resources of the intersection of Large Language Models and Evolutionary Computation.

🎉 ***News: Our survey has been accepted by IEEE Transactions on Evolutionary Computation (TEVC).***
[Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap](https://ieeexplore.ieee.org/document/10767756)

***The related work and projects will be updated soon and continuously.***

<div align="center">
	<img src="https://github.com/wuxingyu-ai/LLM4EC/blob/main/Framework.png" alt="Editor" width="600">
</div>

If our work has been of assistance to you, please feel free to cite our survey. Thank you.
```
@article{wu2024evolutionary,
  title={Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap},
  author={Wu, Xingyu and Wu, Sheng-hao and Wu, Jibin and Feng, Liang and Tan, Kay Chen},
  journal={IEEE Transactions on Evolutionary Computation},
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
  - [Applications Driven by Integrated Synergy of LLM and EA](#applications-driven-by-integrated-synergy-of-llm-and-ea)
    - [Code Generation](#code-generation)
    - [Software Engineering](#software-engineering)
    - [Neural Architecture Search](#neural-architecture-search)
	- [Others Generative Tasks](#others-generative-tasks)

## LLM-enhanced EA

### LLM-assisted Black-box Optimization

| **Category** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Evaluation | N/A | [Exploring the True Potential: Evaluating the Black-Box Optimization Capability of Large Language Models](https://arxiv.org/pdf/2404.06290) | arXiv | 2024 | N/A |
| Evaluation | N/A | [Towards Optimizing with Large Language Models](https://arxiv.org/pdf/2310.05204) | arXiv | 2023 | N/A |
| Evaluation | N/A | [A Critical Examination of Large Language Model Capabilities in Iteratively Refining Differential Evolution Algorithm](https://dl.acm.org/doi/abs/10.1145/3638530.3664179) | GECCO | 2024 | N/A |
| Evaluation | N/A | [Can Large Language Models Be Trusted as Black-Box Evolutionary Optimizers for Combinatorial Problems?](https://arxiv.org/pdf/2501.15081) | arXiv | 2025 | N/A |
| Single-objective | OPRO | [Large Language Models as Optimizers](https://arxiv.org/pdf/2309.03409) | ICLR | 2024 | [Python](https://github.com/google-deepmind/opro) |
| Single-objective | LMX | [Language Model Crossover: Variation through Few-Shot Prompting](https://arxiv.org/pdf/2302.12170) | arXiv | 2023 | N/A |
| Single-objective | LMEA | [Large Language Models as Evolutionary Optimizers](https://arxiv.org/pdf/2310.19046) | CEC | 2024 | N/A |
| Single-objective | EvoLLM | [Large Language Models As Evolution Strategies](https://arxiv.org/pdf/2402.18381) | arXiv | 2024 | N/A |
| Single-objective | Huang et al. | [How Multimodal Integration Boost the Performance of LLM for Optimization: Case Study on Capacitated Vehicle Routing Problems](https://arxiv.org/pdf/2403.01757) | SSCI | 2025 | N/A |
| Single-objective | LEO | [Large Language Model-Based Evolutionary Optimizer: Reasoning with Elitism](https://arxiv.org/pdf/2403.02054) | arXiv | 2024 | N/A |
| Single-objective| LAEA | [Large Language Models as Surrogate Models in Evolutionary Algorithms: A Preliminary Study](https://www.sciencedirect.com/science/article/abs/pii/S2210650224002797) | SWEVO | 2024 | [Python](https://github.com/hhyqhh/LAEA.git) |
| Single-objective| PAIR | [PAIR: A Novel Large Language Model-Guided Selection Strategy for Evolutionary Algorithms](https://arxiv.org/pdf/2503.03239) | arXiv | 2025 | [Python](https://github.com/SHIXOOM/PAIR) |
| Single-objective (Application) | Zhang et al. | [CUDA-Accelerated Soft Robot Neural Evolution with Large Language Model Supervision](https://arxiv.org/pdf/2405.00698) | arXiv | 2024 | N/A |
| Single-objective (Application) | LLM-Mutate | [Evolving Interpretable Visual Classifiers with Large Language Models](https://arxiv.org/pdf/2404.09941) | arXiv | 2024 | N/A |
| Multi-objective | QDAIF | [Quality-Diversity through AI Feedback](https://arxiv.org/pdf/2310.13032v4) | Workshop at NeurIPS | 2023 | [Python](https://openreview.net/attachment?id=nr0w6CH7v4&name=supplementary_material) |
| Multi-objective | LLM4MOEA | [Large Language Model for Multi-objective Evolutionary Optimization](https://arxiv.org/pdf/2310.12541) | arXiv | 2023 | [Python](https://github.com/FeiLiu36/LLM4MOEA) |
| Multi-objective | In-context QD | [Large Language Models as In-context AI Generators for Quality-Diversity](https://arxiv.org/pdf/2404.15794) | arXiv | 2024 | N/A |
| Multi-objective | CMOEA-LLM | [Large Language Model-Aided Evolutionary Search for Constrained Multiobjective Optimization](https://arxiv.org/pdf/2405.05767) | arXiv | 2024 | N/A |
| Multi-objective |  LLM-assisted MOEA | [Large Language Model Aided Multi-objective Evolutionary Algorithm: a Low-cost Adaptive Approach](https://arxiv.org/pdf/2410.02301) | arXiv | 2024 | N/A |
| Multi-objective (Application) | IlmPC-NSGA-II | [Generative Evolution Attacks Portfolio Selection](https://ieeexplore.ieee.org/abstract/document/10612020) | CEC | 2024 | N/A |


### LLM-assisted Optimization Algorithm Generation

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Generated Algorithm                   |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| Pluhacek et al. | [Leveraging Large Language Models for the Generation of Novel Metaheuristic Optimization Algorithms](https://dl.acm.org/doi/abs/10.1145/3583133.3596401) | GECCO | 2023 | N/A | Hybrid swarm intelligence optimization algorithm |
| OptiMUS | [OptiMUS: Optimization Modeling Using MIP Solvers and Large Language Models](https://arxiv.org/pdf/2310.06116) | arXiv | 2023 | [Python](https://github.com/teshnizi/OptiMUS) | Mixed-integer linear programming problem |
| ZSO | [Leveraging Large Language Model to Generate a Novel Metaheuristic Algorithm with CRISPE Framework](https://arxiv.org/pdf/2403.16417) | arXiv | 2024 | [Python](https://github.com/RuiZhong961230/ZSO) | Zoological search optimization algorithm |
| AEL | [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design using Large Language Model](https://arxiv.org/pdf/2401.02051) | ICML | 2024 | [Python](https://github.com/FeiLiu36/EoH) | Heuristic algorithm |
| ReEvo | [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://arxiv.org/pdf/2402.01145) | arXiv | 2024 | [Python](https://github.com/ai4co/LLM-as-HH) | Heuristic algorithm |
| LLM_GP | [Evolving Code with A Large Language Model](https://arxiv.org/pdf/2401.07102) | arXiv | 2024 | N/A | Genetic Programming |
| SR-EAD | [Evolution Transformer: In-Context Evolutionary Optimization](https://arxiv.org/pdf/2403.02985) | arXiv | 2024 | [Python](https://github.com/RobertTLange/evosax) | Evolutionary strategy or evolution transformer |
| EvolCAF | [Evolve Cost-aware Acquisition Functions Using Large Language Models](https://arxiv.org/pdf/2404.16906) | arXiv | 2024 | [Python](https://github.com/RobertTLange/evosax) | Cost-aware Bayesian optimization |
| Kramer | [Large Language Models for Tuning Evolution Strategies](https://arxiv.org/pdf/2405.10999v1) | arXiv | 2024 | N/A | Evolution Strategies |
| LLaMEA | [LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics](https://ieeexplore.ieee.org/abstract/document/10752628) | TEVC | 2024 | N/A | Heuristic algorithm |
| Evaluation | [Understanding the Importance of Evolutionary Search in Automated Heuristic Design with Large Language Models](https://link.springer.com/chapter/10.1007/978-3-031-70068-2_12) | PPSN | 2024 | [Python](https://github.com/zhichao-lu/llm-eps) | Combinatorial Optimization Problem |
| Pluhacek et al. | [Using LLM for Automatic Evolvement of Metaheuristics from Swarm Algorithm SOMA](https://dl.acm.org/doi/abs/10.1145/3638530.3664181) | GECCO | 2024 | N/A | Self-Organizing Migrating Algorithm |
| Pang et al. | [Large Language Model-Based Benchmarking Experiment Settings for Evolutionary Multi-Objective Optimization](https://arxiv.org/pdf/2502.21108) | arXiv | 2025 | N/A | Multi-Objective EA |


### LLM Empower EA for Other Capabilities

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Enhancement Aspect                 |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| OptiChat | [Diagnosing Infeasible Optimization Problems Using Large Language Models](https://arxiv.org/pdf/2308.12923) | arXiv | 2023 | [Python](https://github.com/li-group/OptiChat) | Identify potential sources of infeasibility |
| AS-LLM | [Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation](https://arxiv.org/pdf/2311.13184) | IJCAI | 2024 | [Python](https://github.com/wuxingyu-ai/AS-LLM) | Algorithm representation and algorithm selection |
| GP4NLDR | [Explaining Genetic Programming Trees Using Large Language Models](https://arxiv.org/pdf/2403.03397) | arXiv | 2024 | N/A |  Provide explainability for results of EA |
| Singh et al. | [Enhancing Decision-Making in Optimization through LLM-Assisted Inference: A Neural Networks Perspective](https://arxiv.org/pdf/2405.07212) | IJCNN | 2024 | N/A |  Provide explainability for results of EA |
| Custode et al. | [An Investigation on the Use of Large Language Models for Hyperparameter Tuning in Evolutionary Algorithms](https://dl.acm.org/doi/pdf/10.1145/3638530.3664163) | GECCO | 2024 | [Python](https://github.com/DIOL-UniTN/llm_step_size_adaptation) | Hyperparameter Tuning |


## EA-enhanced LLM

### EA-based Prompt Engineering

| **Category** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Discrete Prompt Optimization | GPS | [GPS: Genetic Prompt Search for Efficient Few-shot Learning](https://arxiv.org/pdf/2210.17041) | EMNLP | 2022 | [Python](https://github.com/hwxu20/GPS) |
| Discrete Prompt Optimization | GrIPS | [GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models](https://arxiv.org/pdf/2203.07281) | EACL | 2023 | [Python](https://github.com/archiki/GrIPS) |
| Discrete Prompt Optimization | EvoPrompt | [Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers](https://arxiv.org/pdf/2309.08532) | ICLR | 2024 | [Python](https://github.com/beeevita/EvoPrompt) |
| Discrete Prompt Optimization | Plum | [Plum: Prompt Learning Using Metaheuristic](https://arxiv.org/pdf/2311.08364) | arXiv | 2023 | [Python](https://github.com/research4pan/Plum) |
| Discrete Prompt Optimization | PromptBreeder | [Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution](https://arxiv.org/pdf/2309.16797) | arXiv | 2023 | N/A |
| Discrete Prompt Optimization | SPELL | [SPELL: Semantic Prompt Evolution Based on A LLM](https://arxiv.org/pdf/2310.01260) | arXiv | 2023 | N/A |
| Discrete Prompt Optimization | EoT prompting | [Zero-Shot Chain-of-Thought Reasoning Guided by Evolutionary Algorithms in Large Language Models](https://arxiv.org/pdf/2402.05376) | arXiv | 2024 | [Python](https://github.com/stan-anony/Zero-shot-EoT-Prompting) |
| Discrete Prompt Optimization | iPrompt | [Explaining Patterns in Data with Language Models via Interpretable Autoprompting](https://arxiv.org/pdf/2210.01848) | arXiv | 2023 | [Python](https://openreview.net/attachment?id=GvMuB-YsiK6&name=supplementary_material) |
| Discrete Prompt Optimization | PhaseEvo | [PhaseEvo: Towards Unified In-Context Prompt Optimization for Large Language Models](https://arxiv.org/pdf/2402.11347) | arXiv | 2024 | N/A |
| Discrete Prompt Optimization | InstOptima | [InstOptima: Evolutionary Multi-objective Instruction Optimization via Large Language Model-based Instruction Operators](https://arxiv.org/pdf/2310.17630) | EMNLP | 2023 | [Python](https://github.com/yangheng95/InstOptima) |
| Discrete Prompt Optimization | EMO-Prompts | [Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments](https://arxiv.org/pdf/2401.09862) | EvoApplications | 2024 | [Python](https://github.com/ollama/ollama) |
| Discrete Prompt Optimization | RSBench | [Language Model Evolutionary Algorithms for Recommender Systems: Benchmarks and Algorithm Comparisons](https://arxiv.org/pdf/2411.10697) | arXiv | 2024 | N/A |
| Discrete Prompt Optimization | evoPrompt | [Exploring the Prompt Space of Large Language Models through Evolutionary Sampling](https://dl.acm.org/doi/pdf/10.1145/3638529.3654049) | GECCO | 2024 | [Python](https://github.com/Martisal/evoPrompt) |
| Discrete Prompt Optimization | PREDO | [Prompt Evolutionary Design Optimization with Generative Shape and Vision-Language models](https://ieeexplore.ieee.org/abstract/document/10611898) | CEC | 2024 | N/A |
| Gradient-Free Soft Prompt Optimization | BBT | [Black-Box Tuning for Language-Model-as-a-Service](https://proceedings.mlr.press/v162/sun22e/sun22e.pdf) | ICML | 2022 | [Python](https://github.com/txsun1997/Black-Box-Tuning) |
| Gradient-Free Soft Prompt Optimization | BBTv2 | [BBTv2: Towards a Gradient-Free Future with Large Language Models](https://arxiv.org/pdf/2205.11200) | EMNLP | 2022 | [Python](https://github.com/txsun1997/Black-Box-Tuning) |
| Gradient-Free Soft Prompt Optimization | Clip-Tuning | [Clip-Tuning: Towards Derivative-free Prompt Learning with a Mixture of Rewards](https://aclanthology.org/2022.findings-emnlp.8.pdf) | EMNLP | 2022 | N/A |
| Gradient-Free Soft Prompt Optimization | Shen et al. | [Reliable Gradient-free and Likelihood-free Prompt Tuning](https://github.com/maohaos2/SBI_LLM) | ACL | 2023 | [Python](https://github.com/txsun1997/Black-Box-Tuning) |
| Gradient-Free Soft Prompt Optimization | BPT-VLM | [Black-box Prompt Tuning for Vision-Language Model as a Service](https://www.ijcai.org/proceedings/2023/0187.pdf) | IJCAI | 2023 | [Python](https://github.com/BruthYU/BPT-VLM) |
| Gradient-Free Soft Prompt Optimization | Fei et al. | [Gradient-Free Textual Inversion](https://dl.acm.org/doi/pdf/10.1145/3581783.3612599) | MM | 2023 | [Python](https://github.com/feizc/Gradient-Free-Textual-Inversion) |
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

### EA-based LLM Merging

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Merging Recipes | [Evolutionary Optimization of Model Merging Recipes](https://www.nature.com/articles/s42256-024-00975-8) | NMI | 2025 | [Python](https://github.com/SakanaAI/evolutionary-model-merge) |
| GENOME+ | [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/pdf/2403.13187) | arXiv | 2025 | [Python](https://github.com/ZhangYiqun018/GENOME) |
| EEM-TISP | [Evolutionary Expert Model Merging with Task-Adaptive Iterative Self-Improvement Process for Large Language Modeling on Aspect-Based Sentiment Analysis](https://ieeexplore.ieee.org/abstract/document/10799461) | IoTaIS | 2024 | N/A |



### EA Empower LLM for Other Capabilities

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Enhancement Aspect                 |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| Length-Adaptive Transformer Model | [Length-Adaptive Transformer: Train Once with Length Drop, Use Anytime with Search](https://aclanthology.org/2021.acl-long.508.pdf) | ACL | 2021 | [Python](https://github.com/clovaai/length-adaptive-transformer) | Automatically adjust the sequence length according to different computational resource constraints |
| HexGen | [HexGen: Generative Inference of Large-Scale Foundation Model over Heterogeneous Decentralized Environment](https://arxiv.org/pdf/2311.11514) | arXiv | 2023 | [Python](https://github.com/Relaxed-System-Lab/HexGen) | Deploy generative inference services for LLMs in a heterogeneous distributed environment |
| LongRoPE | [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/pdf/2402.13753) | arXiv | 2023 | [Python](https://github.com/microsoft/LongRoPE) | Extend the context window of LLMs to 2048k tokens |
| BLADE | [BLADE: Enhancing Black-box Large Language Models with Small Domain-Specific Models](https://arxiv.org/pdf/2403.18365) | arXiv | 2024 | N/A | Find soft prompts that optimizes the consistency between the outputs of two models |
| Self-evolution in LLM | [A Survey on Self-Evolution of Large Language Models](https://arxiv.org/pdf/2404.14387) | arXiv | 2024 | [Summary](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/Awesome-Self-Evolution-of-LLM) | Some studies for LLM self-evolution also adopted the ideas of EAs |
| OPTISHEAR | [OPTISHEAR: Towards Efficient and Adaptive Pruning of Large Language Models via Evolutionary Optimization](https://arxiv.org/pdf/2502.10735) | arXiv | 2025 | N/A | An efficient evolutionary optimization framework for adaptive LLM pruning using NSGA-III |

## Applications Driven by Integrated Synergy of LLM and EA

### Code Generation

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Applicable scenarios                   |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| ELM | [Evolution through Large Models](https://arxiv.org/pdf/2206.08896) | arXiv | 2022 | [Python](https://github.com/CarperAI/OpenELM) | Universal code generation |
| OpenELM | [The OpenELM Library: Leveraging Progress in Language Models for Novel Evolutionary Algorithms](https://arxiv.org/pdf/2404.16906) | arXiv | 2024 | [Python](https://github.com/CarperAI/OpenELM) | Open-source Python library |
| Pinna et al. | [Enhancing Large Language Models-Based Code Generation by Leveraging Genetic Improvement](https://link.springer.com/chapter/10.1007/978-3-031-56957-9_7) | ECGP | 2024 | [Python](https://github.com/dravalico/LLMGIpy) | Universal code generation |
| WizardCoder | [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://openreview.net/pdf?id=UnUwSIgK5W) | ICLR | 2024 | [Python](https://github.com/nlpxucan/WizardLM/tree/main/WizardCoder) | Universal code generation |
| SEED | [SEED: Domain-Specific Data Curation With Large Language Models](https://arxiv.org/pdf/2310.00749) | arXiv | 2023 | N/A | Data cleaning tasks |
| EUREKA | [Eureka: Human-Level Reward Design via Coding Large Language Models](https://arxiv.org/pdf/2310.12931) | ICLR | 2024 | [Python](https://github.com/eureka-research/Eureka) | Design reward in reinforcement learning |
| EROM | [Evolutionary Reward Design and Optimization with Multimodal Large Language Models](https://openreview.net/pdf?id=PwlKdPDZK4) | Workshop at ICRA | 2023 | N/A | Design reward in reinforcement learning |
| Zhang et al. | [A Simple Framework for Intrinsic Reward-Shaping for RL using LLM Feedback](https://alexzhang13.github.io/assets/pdfs/Reward_Shaping_LLM.pdf) | Github | 2024 | N/A | Design reward in reinforcement learning |
| Diffusion-ES | [Diffusion-ES: Gradient-free Planning with Diffusion for Autonomous Driving and Zero-Shot Instruction Following](https://arxiv.org/pdf/2402.06559) | arXiv | 2024 | N/A |Design reward in reinforcement learning  |
| GPT4AIGChip | [GPT4AIGChip: Towards Next-Generation AI Accelerator Design Automation via Large Language Models](https://arxiv.org/pdf/2309.10730) | ICCAD | 2023 | N/A | Design AI accelerator |
| FunSearch | [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6.pdf) | Nature | 2023 | [Python](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-023-06924-6/MediaObjects/41586_2023_6924_MOESM2_ESM.zip) | For mathematical and algorithmic discovery |
| L-AutoDA | [L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks](https://arxiv.org/pdf/2401.15335) | arXiv | 2024 | N/A | For decision-based adversarial attacks |
| LLM-SR | [LLM-SR: Scientific Equation Discovery via Programming with Large Language Models](https://arxiv.org/pdf/2404.18400) | arXiv | 2024 | [Python](https://github.com/deep-symbolic-mathematics/llm-sr) | For scientific equation discovery from data |
| Shojaee et al. | [Identify Critical Nodes in Complex Network with Large Language Models](https://arxiv.org/pdf/2403.03962) | arXiv | 2024 | [Python](https://anonymous.4open.science/r/LLM4CN-6520/README.md) | Node importance scoring functions in complex networks |
| DeceptPrompt | [DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions](https://arxiv.org/pdf/2312.04730) | arXiv | 2023 | N/A | For LLM Security |
| G3P with LLM | [Program Synthesis with Generative Pre-trained Transformers and Grammar-Guided Genetic Programming Grammar](https://ieeexplore.ieee.org/document/10409384) | LA-CCI | 2023 | N/A | For LLM Security |

### Software Engineering

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |      Applicable scenarios                    |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| Kang et al. | [Towards Objective-Tailored Genetic Improvement Through Large Language Models](https://ieeexplore.ieee.org/abstract/document/10190823) | Workshop at ICSE | 2023 | N/A | Software Optimization |
| Brownlee et al. | [Enhancing Genetic Improvement Mutations Using Large Language Models](https://arxiv.org/pdf/2310.19813) | SSBSE | 2023 | N/A | Software Optimization |
| ARJA-CLM | [Revisiting Evolutionary Program Repair via Code Language Model](https://arxiv.org/pdf/2408.10486) | arXiv | 2024 | N/A | Software Optimization (Program Repair) |
| TitanFuzz | [Large Language Models Are Zero-Shot Fuzzers: Fuzzing Deep-Learning Libraries via Large Language Models](https://dl.acm.org/doi/abs/10.1145/3597926.3598067) | ISSTA | 2023 | N/A | Software Testing |
| CodaMOSA | [CODAMOSA: Escaping Coverage Plateaus in Test Generation with Pre-trained Large Language Models](https://ieeexplore.ieee.org/document/10172800) | ICSE | 2023 | [Python](https://github.com/microsoft/codamosa) | Software Testing |
| SBSE | [Search-based Optimisation of LLM Learning Shots for Story Point Estimation](https://dl.acm.org/doi/abs/10.1007/978-3-031-48796-5_9) | SSBSE | 2023 | N/A | Software Project Planning |

### Neural Architecture Search

Note: Methods reviewed here leverage the synergistic combination of EAs and LLMs, which are more versatile and not limited to LLM architecture search alone, applicable to a broader range of NAS tasks..

| **Name** | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |       Role of LLM                    |
| -------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ | --------------------- |
| GPT-NAS | [GPT-NAS: Evolutionary Neural Architecture Search with the Generative Pre-Trained Model](https://arxiv.org/pdf/2305.05351) | arXiv | 2023 | N/A | Representation Capability |
| LLMatic | [LLMatic: Neural Architecture Search via Large Language Models and Quality Diversity Optimization](https://arxiv.org/pdf/2306.01102) | GECCO | 2024 | [Python](https://openreview.net/attachment?id=iTrd5xyHLP&name=supplementary_material) | Generation Capability |
| Evoprompting | [EvoPrompting: Language Models for Code-Level Neural Architecture Search](https://arxiv.org/pdf/2302.14838) | NeurIPS | 2023 | [Python](https://github.com/algopapi/EvoPrompting_Reinforcement_learning) | Generation Capability |
| Guided Evolution | [LLM Guided Evolution -- The Automation of Models Advancing Models](https://arxiv.org/pdf/2403.11446) | arXiv | 2023 | [Python](https://github.com/clint-kristopher-morris/llm-guided-evolution) | Generation Capability |
| GPTN-SS | [Discovering More Effective Tensor Network Structure Search Algorithms via Large Language Models (LLMs)](https://arxiv.org/pdf/2402.02456) | arXiv | 2024 | N/A | Generation Capability |
| GENIUS | [Can GPT-4 Perform Neural Architecture Search?](https://arxiv.org/pdf/2304.10970) | arXiv | 2023 | [Python](https://github.com/mingkai-zheng/GENIUS) | Generation Capability |
| GPT4GNAS | [Graph Neural Architecture Search with GPT-4](https://arxiv.org/pdf/2310.01436) | arXiv | 2023 | N/A | Generation Capability |
| Jawahar et al. | [LLM Performance Predictors are good initializers for Architecture Searc](https://arxiv.org/pdf/2310.16712) | arXiv | 2023 | N/A | Reasoning Capability |
| ReStruct | [Large Language Model-driven Meta-structure Discovery in Heterogeneous Information Network](https://arxiv.org/pdf/2402.11518) | arXiv | 2024 | N/A | Reasoning Capability |

### Others Generative Tasks

| **Task** |      Name                   | **Paper**                                                    | **Venue** | **Year** | **Code**                                               |
| -------- | --------------------- | ------------------------------------------------------------ | --------- | -------- | ------------------------------------------------------------ |
| Text Generation | NSG | [Enhancing LLM with Evolutionary Fine Tuning for News Summary Generation](https://arxiv.org/pdf/2307.02839) | Journal of Intelligent and Fuzzy Systems | 2024 | N/A |
| Text Generation | SCAPE | [SCAPE: Searching Conceptual Architecture Prompts using Evolution](https://arxiv.org/pdf/2402.00089) | CEC | 2024 | [Python](https://github.com/soolinglim/webllm/) |
| Text Generation | Mario-GPT | [Prompt-Guided Level Generation](https://dl.acm.org/doi/abs/10.1145/3583133.3590656) | GECCO | 2024 | [Python](https://github.com/shyamsn97/mario-gpt) |
| Text Generation | Cai et al. | [Simulation of Language Evolution under Regulated Social Media Platforms: A Synergistic Approach of Large Language Models and Genetic Algorithms](https://arxiv.org/pdf/2502.19193) | arXiv | 2025 | N/A |
| Text-to-image Generation | StableYolo | [StableYolo: Optimizing Image Generation for Large Language Models](https://link.springer.com/chapter/10.1007/978-3-031-48796-5_10) | SSBSE | 2023 | [Python](https://github.com/SOLAR-group/StableYolo) |
| Natural Science | Researchers from McGill University | [14 Examples of How LLMs Can Transform Materials Science and Chemistry: A Reflection on a Large Language Model Hackathon](https://arxiv.org/pdf/2306.06283) | Digital Discovery | 2023 | N/A |
| Natural Science | LLM-GA | [Integrating Genetic Algorithms and Language Models for Enhanced Enzyme Design](https://chemrxiv.org/engage/chemrxiv/article-details/65f0746b9138d23161510400) | ChemRxiv | 2024 | N/A |
| Natural Science | MLDE | [Protein Design by Directed Evolution Guided by Large Language Models](https://ieeexplore.ieee.org/abstract/document/10628050) | TEVC | 2024 | [Python](https://github.com/HySonLab/Directed_Evolution) |
| Natural Science | MolLEO | [Efficient Evolutionary Search Over Chemical Space with Large Language Models](https://openreview.net/forum?id=Bvlw0pFRS0) | ICML Workshop | 2024 | [Python](https://github.com/zoom-wang112358/MOLLEO) |
| Natural Science | Reinhart et al. | [Large Language Models Design Sequence-defined Macromolecules via Evolutionary Optimization](https://www.nature.com/articles/s41524-024-01449-6.pdf) | NPJ Computational Materials | 2024 | N/A |
| Social Science | Suzuki et al. | [An Evolutionary Model of Personality Traits Related to Cooperative Behavior Using A Large Language Model](https://www.nature.com/articles/s41598-024-55903-y) | Scientific Reports | 2024 | N/A |
| LLM as Agent | FoA | [Fleet of Agents: Coordinated Problem Solving with Large Language Models using Genetic Particle Filtering](https://arxiv.org/abs/2405.06691) | arXiv | 2024 | N/A |
| LLM as Agent | EvoAgnet | [EVOAGENT: Towards Automatic Multi-Agent Generation via Evolutionary Algorithms](https://arxiv.org/pdf/2406.14228?) | NeurIPS Workshop | 2024 | [Python](https://github.com/siyuyuan/evoagent) |
| Machine Learning | ELLM-FT | [Evolutionary Large Language Model for Automated Feature Transformation](https://arxiv.org/pdf/2405.16203) | arXiv | 2024 | [Python](https://github.com/NanxuGong/ELLM-FT) |

Hope our conclusion can help your work.

