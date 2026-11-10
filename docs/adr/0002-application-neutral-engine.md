# Keep application policy above the engine ABI

geistlib owns model loading and inference; agent loops, tool authorization,
chat templating and product behavior belong to its consumers. Embedding these
policies in the engine could simplify one application but would couple all
consumers to its lifecycle and trust decisions, so the public API instead
exposes the inference primitives needed to build them externally.

This records the existing scope boundary. The optional decision API scores
candidates, and the offline evaluation tools compare models; neither makes
the engine an application policy layer. See
[Architecture](../ARCHITECTURE.md#above-the-abi-what-the-engine-deliberately-does-not-do)
and the [API contract](../API_CONTRACT.md).
