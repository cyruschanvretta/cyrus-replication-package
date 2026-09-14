# Model deployment adapters

## AWS SageMaker

Deploy Gemma 3 12B Instruct behind a real-time endpoint, then set:

```text
AWE_SAGEMAKER_ENDPOINT=<endpoint name>
AWS_REGION=<region>
```

The default `messages` payload is suitable for chat-compatible serving
containers. If the container accepts Hugging Face/TGI text generation instead,
set `model.sagemaker.payload_mode: tgi`; the adapter will send a rendered prompt
under `inputs` with `max_new_tokens`, temperature, and `return_full_text=false`.

SageMaker containers do not share one universal request/response schema. Before
running student responses, use synthetic text to verify that the endpoint
returns exactly the compact profile contract. The response parser accepts
common `generated_text`, `text`, `outputs[0].text`, and
`choices[0].message.content` shapes.

AWS credentials should use the normal SDK credential chain (instance role,
task role, profile, or environment). Do not put credentials in YAML or commit a
`.env` file.

## Generic HTTP

Set `model.adapter: http`, `AWE_MODEL_URL`, and optionally `AWE_MODEL_TOKEN`.
The endpoint receives the same provider-neutral chat or TGI payload. HTTPS is
strongly recommended because requests contain student text.

## Local Ollama

Set `model.adapter: ollama` or pass `--adapter ollama`. The defaults target
`http://127.0.0.1:11434` and model tag `gemma3:12b`. Override either with
`AWE_OLLAMA_URL` and `AWE_OLLAMA_MODEL`. Requests are non-streaming and disable
reasoning so the compact output contract remains the only generated content.

## Python callable or another local runtime

Set `model.adapter: callable` and
`model.callable.target: package.module:function`. The function receives
`messages` and `model` keyword arguments and returns a string or a supported
response object. This is the extension point for a local model server, a
different cloud SDK, or a batch-inference wrapper.

## Concurrency and privacy

The default concurrency is 12. Confirm endpoint autoscaling and throttling
before raising it. The runner writes features and accepted compact output but
does not copy response text to output artifacts. Endpoint/provider logging,
encryption, retention, and regional residency remain deployment
responsibilities.
