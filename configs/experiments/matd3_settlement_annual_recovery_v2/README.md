# MATD3 annual settlement recovery V2

The previous two Union attempts stopped immediately after valid `step_end`
progress writes, including one at 103,936/140,160. V2 writes progress through
atomic replacement, keeps the watchdog armed across the inter-step gap and
saves a lightweight actor checkpoint after every training year.

The residual ONNX remains research evidence, not a standalone deployment: it
requires the external `RBCCommunityPolicy` base. The manifest states this
explicitly. A matched GPU smoke completed 1,024 updates across profiling
boundaries 512 and 1,024, wrote the checkpoint and exported the marked bundle.
