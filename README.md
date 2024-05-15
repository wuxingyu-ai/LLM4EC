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
| AS-LLM | [Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation](https://arxiv.org/pdf/2311.13184) | IJCAI | 2024 | [Python](https://github.com/wuxingyu-ai/AS-LLM) | Algorithm representation and algorithm selection |
| OptiChat | [Diagnosing Infeasible Optimization Problems Using Large Language Models](https://arxiv.org/pdf/2308.12923) | arXiv | 2023 | [Python](https://github.com/li-group/OptiChat) | Identify potential sources of infeasibility |
| GP4NLDR | [Explaining Genetic Programming Trees Using Large Language Models](https://arxiv.org/pdf/2403.03397) | arXiv | 2024 | N/A |  Provide explainability for results of EA |

## EA-enhanced LLM

### EA-based Prompt Engineering

### EA-based LLM Architecture Search

Note: Approaches discussed here primarily focus on LLM architecture search, and their techniques are based on EAs.

### EA Empower LLM for Other Capabilities


## Integrated Synergy and Application of LLM and EA

### Code Generation

### Software Engineering

### Neural Architecture Search

Note: Methods reviewed here leverage the synergistic combination of EAs and LLMs, which are more versatile and not limited to LLM architecture search alone, applicable to a broader range of NAS tasks..

### Others Generative Tasks



Hope our conclusion can help your work.

