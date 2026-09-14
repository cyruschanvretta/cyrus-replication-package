# Model deployment adapters

## AWS Bedrock

The package default. There is no endpoint to provision; set:

```text
BEDROCK_MODEL_ID=<model id or inference profile id>
AWS_REGION=<region>
AWS_ACCESS_KEY_ID=<key id>
AWS_SECRET_ACCESS_KEY=<secret>
```

`BEDROCK_MODEL_ID` accepts a plain model id (`meta.llama3-70b-instruct-v1:0`), a
cross-region inference profile id (`us.meta.llama3-...`), or an application
inference profile ARN; a profile is required for the models that are only
offered that way. Model access must first be granted for that model in that
region in the Bedrock console. Until it is, runs fail with
`AccessDeniedException` rather than a scoring error.

Because a profile ARN does not name the model it routes to, set
`model.bedrock.model_family` to the model actually served. The run manifest
records that value as `model_family` in place of the package-wide
`recommended_family`, so the manifest attests to the model that did the scoring.
The shipped value is `meta.llama3-70b-instruct-v1:0`. Note the evidence boundary:
the reported metrics were obtained with Gemma 3 12B Instruct, so a different
model is a new configuration, not a replication of the recorded results.

The adapter uses the Converse API, which normalizes the request and response
across model vendors, so unlike SageMaker there is no `payload_mode` to choose
and no response-shape verification step. The system instruction is sent in
Converse's separate `system` field, and `temperature`/`max_tokens` are sent as
`inferenceConfig`.

The keys above are optional. If they are absent, the normal SDK credential chain
applies (instance role, task role, or profile), which remains the better choice
for a deployed runner. Keys in a local `.env` are supported for workstation
runs; `.env` is gitignored and must stay uncommitted. Never put credentials in
YAML.

Bedrock quotas are per-account requests and tokens per minute, not per-endpoint
capacity. The default concurrency of 5 can still exceed a low quota, so the client is
configured with botocore standard-mode retries
(`model.bedrock.botocore_max_attempts`, default 3) to absorb `ThrottlingException`
below the pipeline's own retry loop. Lower `--concurrency` if throttling still
surfaces as failed rows.

## AWS SageMaker

Set `model.adapter: sagemaker` or pass `--adapter sagemaker`. Deploy Gemma 3 12B
Instruct behind a real-time endpoint, then set:

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

The default concurrency is 5. Confirm endpoint autoscaling and throttling
before raising it. The runner writes features and accepted compact output but
does not copy response text to output artifacts. Endpoint/provider logging,
encryption, retention, and regional residency remain deployment
responsibilities.
