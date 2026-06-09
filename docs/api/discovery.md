# Discovery & Registration

InterpKit automatically discovers model architecture from HuggingFace configs
and module name heuristics. When auto-discovery fails, use `register()` to
manually specify the model's structure.

## Auto-discovery

::: interpkit.core.arch.discover

::: interpkit.core.arch.resolve_arch

## Data classes

::: interpkit.core.arch.ArchInfo

::: interpkit.core.arch.LayerInfo

::: interpkit.core.arch.ModuleInfo

## Manual registration

::: interpkit.core.registry.register
