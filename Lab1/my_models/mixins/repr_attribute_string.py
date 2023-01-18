import inspect
from typing import Any


class ReprAttributesString:

    @staticmethod
    def is_user_defined_attribute(k_val: tuple[str, Any]) -> bool:
        return not k_val[0].startswith('_') \
            and not inspect.ismethod(k_val[0][1])

    def __repr__(self) -> str:
        result = ''
        members = inspect.getmembers(self)
        for i in members[:-1]:
            if self.is_user_defined_attribute(i):
                result += f'{i[1]},'
        last = members[-1]
        if self.is_user_defined_attribute(last):
            result += f'{last}'
        return result