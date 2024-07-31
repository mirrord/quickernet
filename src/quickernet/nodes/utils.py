

def params_only(return_statement: str) -> list:
    objs = []
    innercount = 0
    begindx = 0
    return_list_str = return_statement.split('return ')[1]
    for idx, c in enumerate(return_list_str):
        if c == '(':
            innercount += 1
        elif c == ')':
            innercount -= 1
        elif c == ',' and innercount == 0:
            objs.append(return_list_str[begindx:idx].strip())
            begindx = idx + 1
    objs.append(return_list_str[begindx:].strip())
    return objs


def replace_all(text: str, replacements: dict) -> str:
    for key, val in replacements.items():
        text = text.replace(key, val)
    return text


def dprint(d: dict, indent=1):
    brace_tabs = '\t' * (indent - 1)
    print(brace_tabs, "{")
    tabs = '\t' * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{tabs}{k}:")
            dprint(v, indent + 1)
        elif isinstance(v, list):
            print(f"{tabs}{k}: [")
            for i in v:
                print(f"{tabs}\t{i}")
            print(f"{tabs}]")
        else:
            print(f"{tabs}{k}: {v}")
    print(brace_tabs, "}")
