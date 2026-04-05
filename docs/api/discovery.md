# Discovery & Registration

InterpKit automatically discovers model architecture from HuggingFace configs
and module name heuristics. When auto-discovery fails, use `register()` to
manually specify the model's structure.

## Auto-discovery

::: interpkit.core.discovery.discover

## Data classes

::: interpkit.core.discovery.ModelArchInfo

::: interpkit.core.discovery.LayerInfo

::: interpkit.core.discovery.ModuleInfo

## Manual registration

::: interpkit.core.registry.register
