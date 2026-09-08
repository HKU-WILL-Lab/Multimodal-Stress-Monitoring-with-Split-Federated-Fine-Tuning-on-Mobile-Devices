# Third-party notices

No reuse license is currently granted for the newly authored project source.
External software and assets retain their own terms; their licenses do not
license this project's original source by implication.

## External source/build dependencies

- **MobileFineTuner** — external, unmodified checkout pinned to
  `b62d3b12a597e05489e6e8ef025527c613c94837`; Apache License 2.0; copyright
  Mobile LLM Fine-Tuning Project Contributors.
  <https://github.com/Edge-Intelligence-Lab/MobileFineTuner>
  Research reference: Jiaxiang Geng, Lunyu Zhao, Yiyi Lu, and Bing Luo.
  *MobileFineTuner: A mobile-native framework for on-device LLM fine-tuning
  in real-world embedded AI applications.*
  [Paper](https://arxiv.org/abs/2512.08211).
- **ExecuTorch** — external source/build dependency; BSD 3-Clause license. If
  ExecuTorch binaries are redistributed, reproduce its copyright, license
  conditions, and disclaimer in the accompanying materials.
  <https://github.com/pytorch/executorch>
- **gRPC** — external source/build dependency; Apache License 2.0.
  <https://github.com/grpc/grpc>
- **Protocol Buffers** — external source/build dependency; BSD 3-Clause
  license. <https://github.com/protocolbuffers/protobuf>
- **PyTorch, Transformers, PEFT, AndroidX, Kotlin, Gradle, Google Play services
  for Wear OS**, and their transitive dependencies retain their respective
  licenses. Dependency declarations identify the requested versions; a binary
  distributor must generate notices from the resolved dependency graph.
- The included Gradle Wrapper scripts and JAR originate from the Gradle project
  and retain the Gradle project's Apache-2.0 terms.

No EdgeFlowerTune or Flower source or binary is distributed or required.
SplitLoRA and SplitFed are research references, not bundled software.

## Encoder and alignment references

The alignment configuration follows the published OpenTSLM-SP architecture.
OpenTSLM source and pretrained files are not distributed. The optional export
helper requires users to supply their own lawfully obtained implementation and
checkpoint. The relevant NormWear component identifies its upstream source as
Apache-2.0, but that does not automatically license other OpenTSLM materials.

- **OpenTSLM:** Patrick Langer et al. *OpenTSLM: Time-Series Language Models
  for Reasoning over Multivariate Medical Text- and Time-Series Data.*
  [Paper](https://arxiv.org/abs/2510.02410) ·
  [Official code](https://github.com/OpenTSLM/OpenTSLM).
  The alignment implementation follows OpenTSLM-SP; the encoder export helper
  loads a separately supplied OpenTSLM NormWear interface.
- **NormWear:** Yunfei Luo, Yuliang Chen, Asif Salekin, and Tauhidur Rahman.
  *Toward Foundation Model for Multivariate Wearable Sensing of Physiological
  Signals.* ACM Transactions on Computing for Healthcare, 2026.
  [Paper](https://doi.org/10.1145/3803808) ·
  [Official code](https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear).
  NormWear provides the pretrained time-series encoder used by the documented
  training and inference workflows; its weights are obtained separately.

## Model assets

Llama 3.2 weights, tokenizer files, fine-tuned/merged weights, and model-derived
artifacts are not distributed. They are governed by the Llama 3.2 Community
License and Acceptable Use Policy.
Any distributor of Llama materials or a product containing them must satisfy
Meta's then-applicable attribution, notice, naming, and acceptable-use terms.

## Data and vendor SDKs

- K-EmoCon data and derived samples are not distributed. Users must obtain the
  dataset from its official source and comply with its terms and human-subject
  data restrictions.
  Reference: Cheul Young Park et al. *K-EmoCon, a multimodal sensor dataset for
  continuous emotion recognition in naturalistic conversations.* Scientific
  Data 7, 293 (2020).
  [Paper](https://doi.org/10.1038/s41597-020-00630-y) ·
  [Dataset](https://zenodo.org/records/3931963).
- Samsung Health Sensor SDK and its `.aar` are not distributed. Physical
  Samsung sensor builds require a separately accepted Samsung SDK agreement,
  supported hardware, and any required partner/package registration.

Research references and repository links are listed in the component
documentation. This notice is informational and does not replace the license
files supplied by each dependency.
