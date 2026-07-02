import copy
from typing import Dict, Any, Callable


class BaseConfig(object):
    """
    Base class for Config object.
    """

    # Minimum geometric length (mm) of a cycle that may be closed during
    # tracking. Default float("inf") disables cycle closure (tree-only),
    # preserving legacy behavior for ASOCA / Aorta24. Anatomies with genuine
    # loops (e.g. Circle of Willis) should override with a finite value.
    min_cycle_length_mm: float = float("inf")

    # Radius head parameterization. "logit" = legacy r_norm=clamp(r/base_radius)
    # then logit. "log" = log(max(r, r_floor) / r_ref); requires r_ref.
    radius_param: str = "logit"
    r_ref: float = None

    def print_self(self) -> Dict[str, Any]:
        """
        Organize all class attributes as a python Dict.
        """

        # Exclude class built-in attributes and print_self() function.
        attr_names = [name for name in dir(self) if not name.startswith("__")]
        attr_dict = {
            name: getattr(self, name)
            for name in attr_names
            if not isinstance(getattr(self, name), Callable)  # methods
        }

        def stringizer(in_dict: Dict):
            """
            A bootstrapped version of stringizer.
            """

            for k, v in in_dict.items():
                if isinstance(v, Dict):
                    v = stringizer(v)
                    in_dict[k] = v
                elif isinstance(v, (str, int, float, bool, type(None))):
                    pass
                else:
                    in_dict[k] = str(v)

            return in_dict

        attr_dict = stringizer(copy.deepcopy(attr_dict))

        return attr_dict
