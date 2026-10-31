"""Run a span of an EXL3 model's blocks on a GPU in another machine, over Ethernet. See doc/remote.md."""
from .stage import RemoteStage, attach_remote, attach_from_env
