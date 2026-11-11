"""G1 sim2real lab: LeRobot recording backbone (see README.md)."""

__all__ = ["LeRobotWriter", "CAMERAS", "IMAGE_SHAPE", "JOINT_NAMES", "ROBOT_TYPE"]


def __getattr__(name):
    # Lazy, so `import g1lab.robot_io` etc. does not pull in torch/lerobot.
    if name in __all__:
        from g1lab import lerobot_writer

        return getattr(lerobot_writer, name)
    raise AttributeError(f"module 'g1lab' has no attribute {name!r}")
