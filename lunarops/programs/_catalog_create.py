"""Internal source selection and execution shared by the two catalog programs."""


def exclusive_source(config, path_name, file_key, inline_key):
    if (config.get(file_key) is not None) == (config.get(inline_key) is not None):
        raise ValueError(f"{path_name} requires exactly one of '{file_key}' or '{inline_key}'.")
    return config


def create_catalog(config, context, *, file_key, inline_key, output_key, load, inline, write, program, noun):
    catalog = (
        load(context.resolve_path(config[file_key])) if config.get(file_key) is not None else inline(config[inline_key])
    )
    output = write(catalog, context.resolve_path(config[output_key]))
    print(f"[{program}] {len(catalog)} {noun}(s) -> {output}")
    return output
