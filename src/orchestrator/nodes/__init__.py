"""Lazy exports for orchestration nodes.

Importing the package should not eagerly pull in heavyweight optional
dependencies such as matplotlib via predictor/explain modules. This keeps
utility code that only needs the generator node usable in lightweight envs.
"""

from importlib import import_module

__all__ = [
    "planner_node",
    "miner_node",
    "make_miner_node",
    "generator_node",
    "make_generator_node",
    "build_pool_index",
    "post_generation_verifier_node",
    "make_post_generation_verifier_node",
    "build_pool_stats",
    "predictor_node",
    "make_predictor_node",
    "route_after_planner",
    "route_after_post_generation_verifier",
]


_EXPORTS = {
    "build_pool_index": ("src.orchestrator.nodes.generator", "build_pool_index"),
    "generator_node": ("src.orchestrator.nodes.generator", "generator_node"),
    "make_generator_node": ("src.orchestrator.nodes.generator", "make_generator_node"),
    "miner_node": ("src.orchestrator.nodes.miner", "miner_node"),
    "make_miner_node": ("src.orchestrator.nodes.miner", "make_miner_node"),
    "planner_node": ("src.orchestrator.nodes.planner", "planner_node"),
    "predictor_node": ("src.orchestrator.nodes.predictor", "predictor_node"),
    "make_predictor_node": ("src.orchestrator.nodes.predictor", "make_predictor_node"),
    "route_after_planner": ("src.orchestrator.nodes.router", "route_after_planner"),
    "route_after_post_generation_verifier": (
        "src.orchestrator.nodes.router",
        "route_after_post_generation_verifier",
    ),
    "build_pool_stats": ("src.orchestrator.nodes.verifier", "build_pool_stats"),
    "post_generation_verifier_node": (
        "src.orchestrator.nodes.verifier",
        "post_generation_verifier_node",
    ),
    "make_post_generation_verifier_node": (
        "src.orchestrator.nodes.verifier",
        "make_post_generation_verifier_node",
    ),
}


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
