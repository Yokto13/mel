import sys
import os


def configure_redirects() -> None:
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, "configs", "general.gin")

    # if is_qids_remap_json_configured(config_path):
    #     print("QID redirects path is already set, skipping")
    #     return

    # env path overrides.
    # If it's not set, we do nothing.
    remap_path = os.environ.get("REDIRECTS")
    if remap_path:
        print(f"Setting QID redirects path to {remap_path}")
        set_qids_remap_json_config(config_path, remap_path)


def is_qids_remap_json_configured(config_path: str) -> bool:
    with open(config_path, "r") as f:
        for line in f:
            if (
                line.strip().startswith("qids_remap_json")
                and len(line.split("=")[1].strip()) > 4
            ):
                return True
    return False


def set_qids_remap_json_config(config_path: str, remap_path: str) -> None:
    with open(config_path, "r+") as f:
        content = f.read()
        f.seek(0)
        lines = content.splitlines()
        modified = False
        for i, line in enumerate(lines):
            if line.strip().startswith("qids_remap_json="):
                lines[i] = f'qids_remap_json="{remap_path}"'
                modified = True
                break
        if not modified:
            lines.append(f'qids_remap_json="{remap_path}"')
        f.write("\n".join(lines))
        f.truncate()


configure_redirects()

print(f"{os.path.dirname(os.path.realpath(__file__))}/src")
sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/src")
print(f"Current Python path: {sys.path}")
