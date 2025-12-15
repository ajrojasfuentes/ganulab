# ganulab/utils/catalog.py

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, List, TypeVar, Union, overload

T = TypeVar("T")


class _Section:
    """
    Represents a category within the catalog.
    """
    # -----------------------------
    # Initialization
    # -----------------------------
    def __init__(self, name: str, description: str = ""):
        """
        Initializes a new section.

        :param name: The name of the section.
        :param description: Optional description of the section.
        """
        self.name = name
        self.description = description
        self._items: Dict[str, Any] = {}

    # -----------------------------
    # Internal Registration
    # -----------------------------
    def _register(self, obj: T, name: Optional[str] = None) -> T:
        """
        Internal registration logic.

        :param obj: The object to register.
        :param name: Optional name for the object (defaults to obj.__name__ or str(obj)).
        :return: The registered object.
        """
        key = name if name else getattr(obj, "__name__", str(obj))
        self._items[key] = obj
        return obj

    # -----------------------------
    # Decorator Overloads
    # -----------------------------
    @overload
    def __call__(self, obj: T) -> T: ...
    
    @overload
    def __call__(self, *, name: str = ...) -> Callable[[T], T]: ...

    def __call__(self, obj: Union[T, None] = None, *, name: str = None) -> Union[T, Callable[[T], T]]:
        """
        Allows using the section as a decorator in two modes:
        1. Simple: @catalog.section
        2. Configured: @catalog.section(name="alias")

        :param obj: The object to decorate (if provided).
        :param name: Optional name for the object.
        :return: The decorated object or a wrapper decorator.
        """
        if obj is None:
            def wrapper(func: T) -> T:
                return self._register(func, name=name)
            return wrapper
        
        return self._register(obj, name=name)

    # -----------------------------
    # Access Methods
    # -----------------------------
    def __getattr__(self, key: str) -> Any:
        """
        Provides attribute access to registered items.

        :param key: The key of the item to access.
        :return: The registered item.
        :raises AttributeError: If the key does not exist in the section.
        """
        try:
            return self._items[key]
        except KeyError:
            raise AttributeError(f"'{self.name}' does not contain '{key}'. Available: {list(self._items.keys())}")

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns a copy of the internal items dictionary.

        :return: A copy of the registered items.
        """
        return self._items.copy()

    # -----------------------------
    # Special Methods
    # -----------------------------
    def __repr__(self) -> str:
        """
        Returns a string representation of the section.

        :return: String representation.
        """
        return f"<Section '{self.name}': {len(self._items)} items>"

    def __dir__(self) -> List[str]:
        """
        Returns a list of available attributes for autocompletion,
        including the registered items.

        :return: List of attribute names.
        """
        return list(self._items.keys()) + list(super().__dir__())

    # -----------------------------
    # Description Methods
    # -----------------------------
    def describe(self) -> None:
        """
        Prints the registered items in this section.
        """
        desc = f": {self.description}" if self.description else ""
        print(f"➔ Section [{self.name}]{desc}")
        
        if not self._items:
            print("   (Empty)")
            return

        # Sort for consistent output
        for key in sorted(self._items.keys()):
            # Attempt to get the docstring of the function/class
            item = self._items[key]
            doc = getattr(item, "__doc__", "")
            doc_line = f" -> {doc.strip().splitlines()[0]}" if doc else ""
            
            # Truncate docstring if too long
            if len(doc_line) > 60:
                doc_line = doc_line[:57] + "..."
            
            print(f"   ├── {key}{doc_line}")


class Catalog:
    """
    The main 'Catalog' that groups multiple sections.
    Example: fllib (containing lossfunc, penalty, etc.)
    """
    # -----------------------------
    # Initialization
    # -----------------------------
    def __init__(self, name: str, description: str = ""):
        """
        Initializes a new catalog.

        :param name: The name of the catalog.
        :param description: Optional description of the catalog.
        """
        self.name = name
        self.description = description
        self._sections: Dict[str, _Section] = {}

    # -----------------------------
    # Section Management
    # -----------------------------
    def create(self, name: str, description: str = "") -> _Section:
        """
        Creates a new section in the catalog.

        :param name: The name of the section to create.
        :param description: Optional description for the section.
        :return: The created section.
        :raises ValueError: If the section already exists.
        """
        if name in self._sections:
            raise ValueError(f"Section '{name}' already exists in catalog '{self.name}'.")
        
        section = _Section(name, description)
        self._sections[name] = section
        return section

    # -----------------------------
    # Access Methods
    # -----------------------------
    def __getattr__(self, key: str) -> _Section:
        """
        Provides attribute access to sections.

        :param key: The key of the section to access.
        :return: The section.
        :raises AttributeError: If the section does not exist.
        """
        try:
            return self._sections[key]
        except KeyError:
            raise AttributeError(f"Catalog '{self.name}' does not have section '{key}'.")

    # -----------------------------
    # Special Methods
    # -----------------------------
    def __repr__(self) -> str:
        """
        Returns a string representation of the catalog.

        :return: String representation.
        """
        sections_list = ", ".join(self._sections.keys())
        return f"<Catalog '{self.name}': [{sections_list}]>"

    def __dir__(self) -> List[str]:
        """
        Returns a list of available attributes for autocompletion,
        including the section names.

        :return: List of attribute names.
        """
        return list(self._sections.keys()) + list(super().__dir__())

    # -----------------------------
    # Description Methods
    # -----------------------------
    def describe(self) -> None:
        """
        Prints the complete structure of the catalog (sections and items).
        """
        print(f"\n▣ CATALOG: {self.name.upper()}")
        if self.description:
            print(f"   {self.description}")
        print("=" * 40)
        
        if not self._sections:
            print("   (Empty Catalog)")
            return

        for section_name in sorted(self._sections.keys()):
            section = self._sections[section_name]
            section.describe()
            print("")  # Space between sections
        print("=" * 40)