from __future__ import annotations

from dataclasses import dataclass, field

from analysis_params import AtomLabelsParam, BoolParam, ChoiceParam, CompoundParam, FloatParam, IntParam


@dataclass
class PromptContext:
    analysis: object
    input_provider: object
    values: dict[str, object] = field(default_factory=dict)


def prompt_config_from_schema(analysis, schema, config_class, provider=None):
    values = {}
    context = PromptContext(
        analysis=analysis,
        input_provider=provider or analysis.input_provider,
        values=values,
    )

    for param in schema:
        values[param.name] = prompt_param(param, context)

    return config_class(**values)


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


def _prompt_compound(param, context):
    selection = context.analysis.compound_selection(
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
        compound_idx = context.values[param.compound]
        compound = context.analysis.compound_by_index(compound_idx)

    return context.analysis.atom_selection(
        role=param.role,
        compound=compound,
        prompt_text=param.prompt,
        allow_empty=param.allow_empty,
        provider=context.input_provider,
    )
