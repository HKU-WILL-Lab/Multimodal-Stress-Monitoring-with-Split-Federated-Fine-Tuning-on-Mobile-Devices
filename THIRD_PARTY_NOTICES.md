# Third-party notices

This document lists the third-party software, models, and datasets used by this
project, together with their sources and applicable licenses.

## Third-party components

### MobileFineTuner

Used for client-side LLM training.

- **Source:** [MobileFineTuner](https://github.com/Edge-Intelligence-Lab/MobileFineTuner)
- **Version:** [`b62d3b12a597e05489e6e8ef025527c613c94837`](https://github.com/Edge-Intelligence-Lab/MobileFineTuner/tree/b62d3b12a597e05489e6e8ef025527c613c94837)
- **License:** Apache License 2.0
- **Copyright:** Mobile LLM Fine-Tuning Project Contributors
- **Paper:** [arXiv:2512.08211v2](https://arxiv.org/abs/2512.08211v2)

```bibtex
@misc{geng2026mobilefinetuner,
  title         = {{MobileFineTuner}: A Mobile-Native Framework for On-Device {LLM} Fine-Tuning in Real-World Embedded {AI} Applications},
  author        = {Jiaxiang Geng and Lunyu Zhao and Yiyi Lu and Bing Luo},
  year          = {2026},
  eprint        = {2512.08211},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Version 2},
  url           = {https://arxiv.org/abs/2512.08211v2}
}
```

### ExecuTorch

Used to run exported models on mobile devices.

- **Source:** [ExecuTorch](https://github.com/pytorch/executorch)
- **License:** [BSD 3-Clause](https://github.com/pytorch/executorch/blob/main/LICENSE)

### gRPC

Used for communication between mobile clients, the main server, and the
federated server.

- **Source:** [gRPC](https://github.com/grpc/grpc)
- **License:** [Apache License 2.0](https://github.com/grpc/grpc/blob/master/LICENSE)

### Protocol Buffers

Used to define communication messages and generate C++ and Python code for
message serialization and parsing.

- **Source:** [Protocol Buffers](https://github.com/protocolbuffers/protobuf)
- **License:** [BSD 3-Clause](https://github.com/protocolbuffers/protobuf/blob/main/LICENSE)

### Samsung Health Sensor SDK

Used to access sensor data on supported Samsung Galaxy watches. The SDK and
its `.aar` library are not included in this repository.

- **Source:** [Samsung Health Sensor SDK](https://developer.samsung.com/health/sensor/overview.html)
- **License:** Samsung Health Sensor SDK license agreement supplied with the SDK.

### Other dependencies

PyTorch, Transformers, PEFT, AndroidX, Kotlin libraries and build tools, and
Google Play services for Wear OS are external dependencies and are not bundled
in this repository. Dependency versions are specified in the Python and Gradle
build configuration files.

The Gradle Wrapper scripts and JAR included in `apps/` are provided by the
[Gradle project](https://github.com/gradle/gradle) under the Apache License 2.0.
The full Gradle distribution is downloaded by the Wrapper when needed.

### OpenTSLM

The alignment implementation follows the OpenTSLM-SP architecture. The encoder
export helper uses OpenTSLM's NormWear interface, supplied separately by the user.

- **Source:** [OpenTSLM](https://github.com/OpenTSLM/OpenTSLM)
- **License:** [MIT (OpenTSLM code)](https://github.com/OpenTSLM/OpenTSLM/blob/main/LICENSE.md); third-party components retain their own licenses.
- **Paper:** [arXiv:2510.02410v3](https://arxiv.org/abs/2510.02410v3)

```bibtex
@misc{langer2026opentslm,
  title         = {{OpenTSLM}: Time-Series Language Models for Reasoning over Multivariate Medical Text- and Time-Series Data},
  author        = {Patrick Langer and Thomas Kaar and Max Rosenblattl and Maxwell A. Xu and Winnie Chow and Martin Maritsch and Robert Jakob and Ning Wang and Juncheng Liu and Aradhana Verma and Brian Han and Daniel Seung Kim and Henry Chubb and Scott Ceresnak and Aydin Zahedivash and Alexander Tarlochan Singh Sandhu and Fatima Rodriguez and Daniel McDuff and Elgar Fleisch and Oliver Aalami and Filipe Barata and Paul Schmiedmayer},
  year          = {2026},
  eprint        = {2510.02410},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Version 3},
  url           = {https://arxiv.org/abs/2510.02410v3}
}
```

### NormWear

Provides the pretrained time-series encoder used in training and inference.
Its source code and pretrained weights are obtained separately.

- **Source:** [NormWear](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear)
- **License:** [Apache License 2.0](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/blob/main/LICENSE)
- **Paper:** [ACM Transactions on Computing for Healthcare](https://doi.org/10.1145/3803808)

```bibtex
@article{luo2026normwear,
  title   = {Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals},
  author  = {Yunfei Luo and Yuliang Chen and Asif Salekin and Tauhidur Rahman},
  journal = {ACM Transactions on Computing for Healthcare},
  year    = {2026},
  doi     = {10.1145/3803808},
  url     = {https://doi.org/10.1145/3803808}
}
```

## Model assets

### Llama 3.2

Used as the base language model for training and inference. This repository
does not include Llama 3.2 model weights, tokenizer files, or fine-tuned or
merged model weights.

- **Source:** [Llama 3.2](https://github.com/meta-llama/llama-models/tree/main/models/llama3_2)
- **License:** [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE)
- **Acceptable Use Policy:** [Llama 3.2 Acceptable Use Policy](https://www.llama.com/llama3_2/use-policy)

## Data

### K-EmoCon

K-EmoCon data and derived samples are not distributed. Users must obtain the
dataset from its official source and comply with its terms and human-subject
data restrictions.

- **Source:** [K-EmoCon dataset](https://zenodo.org/records/3931963)
- **Paper:** [Scientific Data](https://doi.org/10.1038/s41597-020-00630-y)

```bibtex
@article{park2020kemocon,
  title   = {{K-EmoCon}, a multimodal sensor dataset for continuous emotion recognition in naturalistic conversations},
  author  = {Cheul Young Park and Narae Cha and Soowon Kang and Auk Kim and Ahsan Habib Khandoker and Leontios Hadjileontiadis and Alice Oh and Yong Jeong and Uichin Lee},
  journal = {Scientific Data},
  volume  = {7},
  pages   = {293},
  year    = {2020},
  doi     = {10.1038/s41597-020-00630-y},
  url     = {https://doi.org/10.1038/s41597-020-00630-y}
}
```
