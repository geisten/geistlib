# geistlib

geistlib is an application-neutral model inference engine. This glossary
names its domain concepts; API details and implementation rules live in the
[documentation](docs/README.md).

## Language

**Engine**:
The runtime that loads a model and drives inference through sessions.
_Avoid_: Agent runtime, application.

**Architecture**:
A model's computation structure: the operations, their order and the roles
of its tensors.
_Avoid_: Backend, model family.

**Model family**:
A group of transformer models sharing metadata conventions and attention
and feed-forward behavior.
_Avoid_: Backend, architecture when referring to a transformer family.

**Backend**:
An execution provider for model operations on a particular compute platform.
_Avoid_: Architecture, device when referring to the execution provider.

**Model**:
Loaded weights and configuration describing the computation available to
its sessions.
_Avoid_: Session, checkpoint when referring to the loaded model.

**Session**:
The evolving inference state for one model interaction, including its token
history and caches.
_Avoid_: Model, chat when referring to inference state.

**Prefill**:
Processing supplied input tokens to establish or extend a session's state.
_Avoid_: Decode when referring to supplied input.

**Decode**:
Advancing a session to produce subsequent output tokens.
_Avoid_: Prefill when referring to output generation.

**Tower**:
An audio or vision encoder that supplies representations to the decoder.
_Avoid_: Backend.

**Soft-token prefix**:
A sequence of encoder representations supplied before text as decoder input,
rather than vocabulary token IDs.
_Avoid_: Text prompt when referring to encoder representations.

**Placement**:
The assignment of model weights and computation to backends.
_Avoid_: Routing when referring to the assignment itself.

**Layer split**:
A placement in which a leading group of transformer layers runs on a second
backend and the remaining model runs on the load backend.
_Avoid_: Tensor parallelism.

**Twin**:
An alternative backend-bound copy of a weight or layer used by an offload path.
_Avoid_: Draft model.

**Hot expert / cold expert**:
An MoE expert with a resident GPU copy / without a resident GPU copy in the
chosen placement.
_Avoid_: Selected / unselected expert; selection is a separate router decision.

**Decision scoring**:
Numeric scoring of explicitly supplied candidates through the optional
decision API.
_Avoid_: Agent action, generated answer.
