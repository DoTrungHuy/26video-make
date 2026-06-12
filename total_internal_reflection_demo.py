"""Clear entrypoint for the total internal reflection Manim demo.

The original implementation is kept in ``contrast_reflection.py`` for compatibility.
Use this file when rendering or introducing the project, because the filename
matches the actual animation topic more clearly.

Run example:
    manim -pqh total_internal_reflection_demo.py SplitScreenTIR
"""

from contrast_reflection import SplitScreenTIR


__all__ = ["SplitScreenTIR"]


if __name__ == "__main__":
    from manim import tempconfig

    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = SplitScreenTIR()
        scene.render()
