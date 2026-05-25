"""WebSocket serving layer for the SAM2Act / SAM2Act+ HistoryBench policy.

Mirrors robomme_policy_learning's serving protocol (reset / add_buffer / infer)
so the robomme `examples/sam2act` eval client can drive SAM2Act over WebSocket.
"""
