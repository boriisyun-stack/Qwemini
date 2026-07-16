"""Print macOS memory pressure relevant to model experiments."""

import re
import subprocess


def vm_stat() -> dict[str, int]:
    page_size = 4096
    raw = subprocess.check_output(["vm_stat"], text=True)
    result = {}
    for line in raw.splitlines():
        match = re.match(r"([^:]+):\s+(\d+)", line)
        if match:
            result[match.group(1)] = int(match.group(2)) * page_size
    return result


def main() -> None:
    mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    stats = vm_stat()
    active = stats.get("Pages active", 0) + stats.get("Pages wired down", 0)
    compressed = stats.get("Pages occupied by compressor", 0)
    swap = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    print(f"physical: {mem / 2**30:.1f} GiB")
    print(f"active+wired: {active / 2**30:.1f} GiB")
    print(f"compressed: {compressed / 2**30:.1f} GiB")
    print(swap)


if __name__ == "__main__":
    main()

