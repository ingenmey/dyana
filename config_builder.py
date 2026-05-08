from __future__ import annotations

from dataclasses import dataclass, field

from analysis_params import AtomLabelsParam, BoolParam, ChoiceParam, CompoundParam, FloatParam, ForEach, IntParam, When


@dataclass
class PromptContext:
    owner: object
    input_provider: object
    values: dict[str, object] = field(default_factory=dict)
    scope: dict[str, object] = field(default_factory=dict)


def prompt_config_from_schema(owner, schema, config_class, provider=None):
    values = {}
    context = PromptContext(
        owner=owner,
        input_provider=provider or owner.input_provider,
        values=values,
    )

    for step in schema:
        prompt_step(step, context)

    return config_class(**values)


def prompt_step(step, context):
    if isinstance(step, (CompoundParam, AtomLabelsParam, IntParam, FloatParam, BoolParam, ChoiceParam)):
        context.values[step.name] = prompt_param(step, context)
        return
    if isinstance(step, ForEach):
        context.values[step.collect_as] = _run_for_each(step, context)
        return
    if isinstance(step, When):
        _run_when(step, context)
        return
    raise TypeError(f"Unsupported config step type: {type(step).__name__}")


def prompt_param(param, context):
    if isinstance(param, CompoundParam):
        return _prompt_compound(param, context)
    if isinstance(param, AtomLabelsParam):
        return _prompt_atom_labels(param, context)
    if isinstance(param, IntParam):
        return context.input_provider.ask_int(
            param.prompt,
            default=param.default,
            display_default=param.display_default,
            minval=param.minval,
            maxval=param.maxval,
        )
    if isinstance(param, FloatParam):
        if param.allow_none:
            return _prompt_optional_float(param, context)
        return context.input_provider.ask_float(
            param.prompt,
            default=param.default,
            display_default=param.display_default,
            minval=param.minval,
            maxval=param.maxval,
        )
    if isinstance(param, BoolParam):
        return context.input_provider.ask_bool(param.prompt, default=param.default)
    if isinstance(param, ChoiceParam):
        return context.input_provider.ask_choice(
            param.prompt,
            param.choices,
            default=param.default,
        )
    raise TypeError(f"Unsupported config parameter type: {type(param).__name__}")


def _run_for_each(step, context):
    items = _resolve_name(step.source, context)
    if step.collect_mode == "dict":
        collected = {}
    elif step.collect_mode == "list":
        collected = []
    else:
        raise ValueError(f"Unsupported ForEach collect_mode: {step.collect_mode}")

    for item in items:
        inherited_keys = set(context.values)
        child_context = PromptContext(
            owner=context.owner,
            input_provider=context.input_provider,
            values=dict(context.values),
            scope={**context.scope, step.item_name: item},
        )
        for child_step in step.steps:
            prompt_step(child_step, child_context)

        child_values = {
            key: value
            for key, value in child_context.values.items()
            if key not in inherited_keys
        }
        collected_value = _collapse_collected_values(child_values)
        if step.collect_mode == "dict":
            collected[item] = collected_value
        else:
            collected.append(collected_value)

    return collected


def _run_when(step, context):
    left = _resolve_name(step.source, context)
    right = _resolve_name(step.value_source, context) if step.value_source is not None else step.value
    if _compare_when_values(left, step.op, right):
        for child_step in step.steps:
            prompt_step(child_step, context)


def _prompt_compound(param, context):
    selection = context.owner.compound_selection(
        role=param.role,
        multi=param.multi,
        prompt_text=param.prompt,
        provider=context.input_provider,
    )
    if param.multi:
        return [idx for idx, _ in selection]
    idx, _ = selection
    return idx


def _prompt_atom_labels(param, context):
    compound = None
    if param.compound is not None:
        compound_idx = _resolve_name(param.compound, context)
        compound = context.owner.traj.topology_frame.get_compound_type_by_index(compound_idx)

    return context.owner.atom_selection(
        role=param.role,
        compound=compound,
        prompt_text=param.prompt,
        allow_empty=param.allow_empty,
        provider=context.input_provider,
    )


def _resolve_name(name, context):
    if name in context.scope:
        return context.scope[name]
    if name in context.values:
        return context.values[name]
    raise KeyError(f"Unknown schema value reference: {name}")


def _collapse_collected_values(values):
    if len(values) == 1:
        return next(iter(values.values()))
    return dict(values)


def _normalized_for_comparison(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _normalized_for_comparison(val)) for key, val in value.items()))
    if isinstance(value, list):
        return tuple(sorted(_normalized_for_comparison(item) for item in value))
    return value


def _compare_when_values(left, op, right):
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "unordered==":
        return _normalized_for_comparison(left) == _normalized_for_comparison(right)
    raise ValueError(f"Unsupported When operator: {op}")


def _prompt_optional_float(param, context):
    while True:
        answer = context.input_provider.ask_str(
            param.prompt,
            default="" if param.default is None else str(param.default),
            display_default=param.display_default,
        ).strip()
        if answer == "":
            return None
        try:
            value = float(answer)
        except ValueError:
            print("Please enter a valid number or leave blank.")
            continue
        if (param.minval is not None and value < param.minval) or (param.maxval is not None and value > param.maxval):
            if param.minval is not None and param.maxval is not None:
                print(f"Please enter a number between {param.minval} and {param.maxval}, or leave blank.")
            elif param.minval is not None:
                print(f"Please enter a number >= {param.minval}, or leave blank.")
            elif param.maxval is not None:
                print(f"Please enter a number <= {param.maxval}, or leave blank.")
            continue
        return value
