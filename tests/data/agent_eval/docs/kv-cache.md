# KV cache quantization

The KV cache is stored packed 4-bit with an activation rotation applied
before quantization. Quality is tracked as KL divergence against the fp16
cache, split by context depth, in bench_kv_quality. Deep-context KL is the
metric that regresses first when the rotation is wrong.
